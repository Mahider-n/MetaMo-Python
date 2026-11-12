"""Generate accuracy comparisons and a correctness heatmap from saved STEU runs."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from benchmarks.steu_data import audit, load_items


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = ROOT / "benchmarks/data/steu_items.jsonl"
CASES = {
    "no_memory": ("No memory", "#667085"),
    "memory": ("Memory", "#2563A6"),
    "emotion_only": ("MetaMo: emotion only", "#159570"),
    "magus_only": ("MetaMo: Magus only", "#D18018"),
    "magus_with_emotion": ("MetaMo: combined", "#8653A5"),
}
SCORING = {"source": "source_correct", "reference_compatible": "reference_compatible_correct"}
SCORING_LABELS = {"source": "Source scoring key", "reference_compatible": "Reference-compatible scoring key"}


def _settings(manifest: dict) -> dict:
    """Selector/default-weight changes are intentional; other run settings must match."""
    config = manifest.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError("Run manifest has an invalid configuration snapshot")
    keys = ("model", "temperature", "seed", "protocol", "items_sha256", "item_order",
            "max_output_tokens", "max_response_attempts", "feeling_model")
    return {**{key: manifest.get(key) for key in keys},
            "core_config": config.get("core.config")}


def load_comparison(runs: list[Path], items_path: Path) -> tuple[dict, dict]:
    """Validate complete saved runs and select one series per condition/selector."""
    if not runs:
        raise ValueError("Provide at least one completed run directory")
    items = {item.id: item for item in load_items(items_path)}
    series, provenance, settings = {}, [], None
    versions = set()
    for run in runs:
        run = run.resolve()
        checked = audit(run=run, items_path=items_path)
        manifest = json.loads((run / "manifest.json").read_text())
        current = _settings(manifest)
        if settings is None:
            settings = current
        differences = [key for key in current if current[key] != settings[key]]
        if differences:
            raise ValueError(f"Incompatible runs: {', '.join(differences)} differ in {run}")
        records = [json.loads(line) for line in (run / "responses.jsonl").read_text().splitlines() if line.strip()]
        for condition in manifest.get("conditions", checked["scoring"]["source"]["conditions"]):
            key = manifest.get("selector") if condition == "full_metamo" else condition
            if key not in CASES:
                raise ValueError(f"Unsupported selector {key!r} in {run}")
            if key in series:
                raise ValueError(f"Duplicate series {key}: supply exactly one run for each case. "
                                 "Reuse the combined case from the primary run instead of supplying it twice.")
            selected = {r["item_id"]: r for r in records if r["condition"] == condition}
            ordered = [selected[item_id] for item_id in manifest["item_order"]]
            for record in ordered:
                version = (record.get("response") or {}).get("model_version")
                if version:
                    versions.add(version)
            label, color = CASES[key]
            if key == "magus_with_emotion":
                label += f" (w={manifest.get('feeling_weight')})"
            series[key] = {"label": label, "color": color, "run": str(run), "records": ordered,
                           "condition": condition, "selector": manifest.get("selector") if condition == "full_metamo" else "gemini",
                           "feeling_weight": manifest.get("feeling_weight") if condition == "full_metamo" else None}
        provenance.append({"run": str(run), "manifest_sha256": checked["manifest_sha256"],
                           "responses_sha256": checked["responses_sha256"]})
    if len(versions) > 1:
        raise ValueError("Runs contain different recorded model versions; compare runs using the same model version")
    series = {key: series[key] for key in CASES if key in series}
    order = settings["item_order"]
    result = {
        "kind": "steu_saved_results_analysis", "item_order": order, "item_count": len(order),
        "not_evaluated_item_ids": [item_id for item_id in items if item_id not in order],
        "available_cases": list(series), "missing_cases": [key for key in CASES if key not in series],
        "settings": settings, "unrecorded_settings": [key for key, value in settings.items() if value is None],
        "model_versions": sorted(versions), "inputs": provenance,
        # Uniform guessing can exceed 1/5 when a scoring key accepts more than
        # one letter. Compute its expectation from each evaluated answer set.
        "chance_accuracy": {
            scoring: sum(len(items[item_id].source_answers if scoring == "source" else items[item_id].reference_answers)
                         / len(items[item_id].options) for item_id in order) / len(order)
            for scoring in SCORING
        },
        "conditions": {},
    }
    for key, entry in series.items():
        result["conditions"][key] = {name: entry[name] for name in ("label", "run", "condition", "selector", "feeling_weight")}
        result["conditions"][key]["scoring"] = {
            scoring: {"correct": sum(r[field] for r in entry["records"]), "total": len(order),
                      "accuracy": sum(r[field] for r in entry["records"]) / len(order),
                      "incorrect_ids": [r["item_id"] for r in entry["records"] if not r[field]]}
            for scoring, field in SCORING.items()
        }
    return series, result


def _save(fig, output: Path, name: str, plt) -> str:
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return name


def _accuracy(series, data, keys, output, name, title, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.2, len(keys) * 0.65 + 1.5)), layout="constrained")
    for ax, scoring in zip(axes, SCORING):
        values = [data["conditions"][key]["scoring"][scoring] for key in keys]
        percentages = [100 * value["accuracy"] for value in values]
        ax.barh(range(len(keys)), percentages, color=[series[key]["color"] for key in keys])
        for i, (percentage, value) in enumerate(zip(percentages, values)):
            inside = percentage >= 75
            ax.text(percentage - 1 if inside else percentage + 1, i,
                    f"{value['correct']}/{value['total']} ({percentage:.1f}%)", va="center",
                    ha="right" if inside else "left", color="white" if inside else "#222222", fontsize=8)
        ax.set_yticks(range(len(keys)), [series[key]["label"] for key in keys])
        ax.invert_yaxis()
        chance = 100 * data["chance_accuracy"][scoring]
        ax.set(xlim=(0, 100), xlabel="Accuracy (%)",
               title=f"{SCORING_LABELS[scoring]}\nUniform random choice: {chance:.1f}% (dashed)")
        ax.axvline(chance, color="#888888", linestyle="--", linewidth=0.8)
    fig.suptitle(f"{title} — {data['item_count']} shared questions")
    return _save(fig, output, name, plt)


def _correctness(series, data, output, plt):
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 1, figsize=(max(11, data["item_count"] * 0.3), 6), layout="constrained")
    for ax, (scoring, field) in zip(axes, SCORING.items()):
        values = np.array([[record[field] for record in entry["records"]] for entry in series.values()])
        chart = ax.imshow(values, cmap=ListedColormap(["#E9A6A6", "#75C8A5"]), vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(series)), [entry["label"] for entry in series.values()])
        ax.set_xticks(range(data["item_count"]), data["item_order"], fontsize=7)
        ax.set(xlabel="STEU question ID", title=SCORING_LABELS[scoring])
        for (row, column), value in np.ndenumerate(values):
            ax.text(column, row, str(value), ha="center", va="center", fontsize=6)
    colorbar = fig.colorbar(chart, ax=axes, ticks=[0, 1], shrink=0.7)
    colorbar.ax.set_yticklabels(["Incorrect", "Correct"])
    fig.suptitle("Question-level correctness")
    return _save(fig, output, "question_correctness", plt)


def _write_data(series, data, items_path, output):
    items = {item.id: item for item in load_items(items_path)}
    fields = ("case", "item_id", "question", "answer", "answer_text", "source_answers", "reference_answers",
              "source_correct", "reference_compatible_correct")
    with (output / "question_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, entry in series.items():
            for record in entry["records"]:
                item = items[record["item_id"]]
                writer.writerow({"case": key, "question": item.question, "answer_text": item.options[record["answer"]],
                                 "source_answers": "/".join(item.source_answers), "reference_answers": "/".join(item.reference_answers),
                                 **{field: record[field] for field in ("item_id", "answer", *SCORING.values())}})
    (output / "analysis.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def analyze_runs(runs: list[Path], items_path: Path, output: Path) -> dict:
    """Save comparative PNG figures, JSON metrics, and question-level CSV rows."""
    series, data = load_comparison(runs, items_path)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "metamo-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    with matplotlib.rc_context({"font.size": 9}):
        graphs = [_accuracy(series, data, list(series), output, "accuracy_all", f"Overall accuracy ({len(series)}/5 cases)", plt)]
        selectors = [key for key, entry in series.items() if entry["condition"] == "full_metamo"]
        if selectors:
            graphs.append(_accuracy(series, data, selectors, output, "accuracy_selectors",
                                    f"MetaMo selector accuracy ({len(selectors)}/3 selectors)", plt))
        graphs.append(_correctness(series, data, output, plt))
    data["graphs"] = [name for name in graphs if name]
    _write_data(series, data, items_path, output)
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Completed primary run and additional selector run folders")
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--output", type=Path, default=None, help="New directory for PNG graphs and analysis data")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "benchmarks/results/analysis" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        result = analyze_runs(args.runs, args.items, output)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        print(f"Analysis stopped: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(result['graphs'])} PNG graphs, analysis.json, and question_results.csv: {output}")
    if result["missing_cases"]:
        print(f"Partial comparison; missing cases: {', '.join(result['missing_cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
