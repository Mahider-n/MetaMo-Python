"""Generate accuracy comparisons and a correctness heatmap from saved STEU runs."""

from collections import Counter
import math
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from textwrap import fill

import numpy as np

from benchmarks.steu_data import ARCHIVED_CONDITIONS, audit, load_items


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = ROOT / "benchmarks/data/steu_items.jsonl"
CASES = {
    "no_memory": ("No memory", "#667085"),
    "memory": ("Memory", "#2563A6"),
    "plausibility": ("Semantic plausibility", "#418C96"),
    "emotion_then_plausibility": ("Emotion with semantic tie-break", "#367B61"),
    "rank_fusion": ("Emotion + plausibility ranks", "#637C38"),
    "emotion_only": ("MetaMo: emotion only", "#159570"),
    "magus_only": ("MetaMo: Magus only", "#D18018"),
    "magus_with_emotion": ("MetaMo: combined", "#8653A5"),
}
SCORING = {"source": "source_correct", "reference_compatible": "reference_compatible_correct"}
SCORING_LABELS = {"source": "Source scoring key", "reference_compatible": "Reference-compatible scoring key"}
VARIANT_SETTINGS = ("state_policy", "history_policy", "feeling_scorer",
                    "tie_breaker", "tie_tolerance", "feeling_weight")


def _settings(manifest: dict) -> dict:
    """Hold model/data/core settings fixed; retain experimental axes per series."""
    config = manifest.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError("Run manifest has an invalid configuration snapshot")
    keys = ("model", "temperature", "seed", "protocol", "items_sha256", "item_order",
            "max_output_tokens", "max_response_attempts", "feeling_model")
    return {**{key: manifest.get(key) for key in keys},
            "core_config": config.get("core.config")}


def load_comparison(runs: list[Path], items_path: Path) -> tuple[dict, dict]:
    """Validate saved runs and distinguish selector variants while rejecting duplicate configurations."""
    if not runs:
        raise ValueError("Provide at least one completed run directory")
    items = {item.id: item for item in load_items(items_path)}
    series, provenance, settings = {}, [], None
    entries, configurations = [], set()
    case_styles = {**CASES, **{name: (label + " (archived)", "#BD4670")
                              for name, label in ARCHIVED_CONDITIONS.items()}}
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
            base = manifest.get("selector") if condition == "full_metamo" else condition
            if base not in case_styles:
                raise ValueError(f"Unsupported selector {base!r} in {run}")
            variants = {name: manifest.get(name) for name in VARIANT_SETTINGS}
            identity = (condition, base, tuple(variants[name]
                        for name in VARIANT_SETTINGS) if condition == "full_metamo" else ())
            if identity in configurations:
                raise ValueError(f"Duplicate configuration for {base}: supply one run per selector and variant settings.")
            configurations.add(identity)
            selected = {r["item_id"]: r for r in records if r["condition"] == condition}
            ordered = [selected[item_id] for item_id in manifest["item_order"]]
            for record in ordered:
                version = (record.get("response") or {}).get("model_version")
                if version:
                    versions.add(version)
            entries.append({"case_id": manifest.get("case_id") if condition == "full_metamo" else None,
                           "variant_settings": variants, "base_case": base, "run": str(run), "records": ordered,
                           "condition": condition, "selector": manifest.get("selector") if condition == "full_metamo" else "gemini",
                           "feeling_weight": manifest.get("feeling_weight") if condition == "full_metamo" else None})
        provenance.append({"run": str(run), "manifest_sha256": checked["manifest_sha256"],
                           "responses_sha256": checked["responses_sha256"]})
    if len(versions) > 1:
        raise ValueError("Runs contain different recorded model versions; compare runs using the same model version")
    for base, (label, color) in case_styles.items():
        group = [entry for entry in entries if entry["base_case"] == base]
        varying = [name for name in VARIANT_SETTINGS
                   if len({entry["variant_settings"][name] for entry in group}) > 1]
        for entry in group:
            suffix = ", ".join(f"{name}={entry['variant_settings'][name]}" for name in varying)
            key = entry.pop("case_id") or (f"{base} [{suffix}]" if suffix else base)
            if key in series:
                raise ValueError(f"Duplicate series identifier {key}: use distinct case IDs.")
            description = label
            if base == "magus_with_emotion" and "feeling_weight" not in varying:
                description += f" (w={entry['feeling_weight']})"
            if suffix:
                description += f" [{suffix}]"
            series[key] = {**entry, "label": description, "color": color}
    order = settings["item_order"]
    result = {
        "kind": "steu_saved_results_analysis", "item_order": order, "item_count": len(order),
        "not_evaluated_item_ids": [item_id for item_id in items if item_id not in order],
        "available_cases": list(series), "missing_cases": [key for key in CASES if key not in {entry["base_case"] for entry in series.values()}],
        "settings": settings, "unrecorded_settings": [key for key, value in settings.items() if value is None],
        "model_versions": sorted(versions), "inputs": provenance,
        "chance_accuracy": {
            scoring: sum(len(items[item_id].source_answers if scoring == "source" else items[item_id].reference_answers)
                         / len(items[item_id].options) for item_id in order) / len(order)
            for scoring in SCORING
        },
        "conditions": {},
    }
    for key, entry in series.items():
        result["conditions"][key] = {name: entry[name] for name in ("label", "run", "condition", "selector", "feeling_weight", "variant_settings")}
        result["conditions"][key]["scoring"] = {
            scoring: {"correct": sum(r[field] for r in entry["records"]), "total": len(order),
                      "accuracy": sum(r[field] for r in entry["records"]) / len(order),
                      "incorrect_ids": [r["item_id"] for r in entry["records"] if not r[field]]}
            for scoring, field in SCORING.items()
        }
    result["varying_settings"] = [name for name in VARIANT_SETTINGS
                                 if len({entry["variant_settings"][name] for entry in series.values()
                                         if entry["condition"] == "full_metamo"}) > 1]
    result["selection_diagnostics"] = {key: selection_diagnostics(entry["records"], items)
                                       for key, entry in series.items()}
    return series, result


def selection_diagnostics(records, items):
    """Inspect saved scores; counterfactual picks share each record's appraisal.

    This works for legacy logs too and makes no model requests. Margins and ties
    describe the classifier, not proof that an appraisal itself was correct.
    """
    rows = []
    for record in records:
        choice = record.get("feeling_choice")
        if not choice:
            continue
        scores = choice.get("emotion_scores") or {}
        valid = {k: v for k, v in scores.items() if v is not None}
        ranked = sorted(valid.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else None
        ties = [k for k, v in valid.items() if math.isclose(v, ranked[0], abs_tol=1e-9)] if ranked else []
        item = items[record["item_id"]]
        picks = {"selected": record["answer"], "magus": choice.get("magus_pick"),
                 "emotion": choice.get("emotion_pick"), "plausibility": choice.get("plausibility_pick")}
        rows.append({"item_id": item.id, "emotion_margin": margin, "emotion_top_ties": ties,
                     "picks": picks, "correct": {
                         scoring: {name: pick in accepted if pick is not None else None for name, pick in picks.items()}
                         for scoring, accepted in (("source", item.source_answers),
                                                   ("reference_compatible", item.reference_answers))}})
    return {"items": rows, "emotion_top_tie_count": sum(len(r["emotion_top_ties"]) > 1 for r in rows),
            "selected_label_counts": dict(Counter(items[r["item_id"]].options[r["answer"]] for r in records)),
            "selection_changes": {
                scoring: {"overturned_correct_magus": sum(r["correct"][scoring]["magus"] is True and
                                                          r["correct"][scoring]["selected"] is False for r in rows),
                          "rescued_incorrect_magus": sum(r["correct"][scoring]["magus"] is False and
                                                         r["correct"][scoring]["selected"] is True for r in rows)}
                for scoring in SCORING}}


def _save(fig, output: Path, name: str, plt) -> str:
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return name


def _accuracy(series, data, keys, output, name, title, plt):
    labels = [fill(series[key]["label"], width=36) for key in keys]
    row_height = max(.65, .15 * max(label.count("\n") + 1 for label in labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.2, len(keys) * row_height + 1.5)), layout="constrained")
    for ax, scoring in zip(axes, SCORING):
        values = [data["conditions"][key]["scoring"][scoring] for key in keys]
        percentages = [100 * value["accuracy"] for value in values]
        ax.barh(range(len(keys)), percentages, color=[series[key]["color"] for key in keys])
        for i, (percentage, value) in enumerate(zip(percentages, values)):
            inside = percentage >= 75
            ax.text(percentage - 1 if inside else percentage + 1, i,
                    f"{value['correct']}/{value['total']} ({percentage:.1f}%)", va="center",
                    ha="right" if inside else "left", color="white" if inside else "#222222", fontsize=8)
        ax.set_yticks(range(len(keys)), labels)
        ax.invert_yaxis()
        chance = 100 * data["chance_accuracy"][scoring]
        ax.set(xlim=(0, 100), xlabel="Accuracy (%)",
               title=f"{SCORING_LABELS[scoring]}\nUniform random choice: {chance:.1f}% (dashed)")
        ax.axvline(chance, color="#888888", linestyle="--", linewidth=0.8)
    fig.suptitle(f"{title} — {data['item_count']} shared questions")
    return _save(fig, output, name, plt)


def _correctness(series, data, output, plt):
    from matplotlib.colors import ListedColormap

    labels = [fill(entry["label"], width=40) for entry in series.values()]
    row_height = max(.35, .15 * max(label.count("\n") + 1 for label in labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(11, data["item_count"] * 0.3), max(6, 2 * len(series) * row_height + 3)), layout="constrained")
    for ax, (scoring, field) in zip(axes, SCORING.items()):
        values = np.array([[record[field] for record in entry["records"]] for entry in series.values()])
        chart = ax.imshow(values, cmap=ListedColormap(["#E9A6A6", "#75C8A5"]), vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(series)), labels)
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
        graphs = []
        for name, keys, title in [("accuracy_all", list(series), "Overall accuracy"),
                                  ("accuracy_selectors", [k for k,e in series.items() if e["condition"] == "full_metamo"],
                                   "Selector and configuration accuracy")]:
            for offset in range(0, len(keys), 24):
                subset = keys[offset:offset + 24]
                suffix = f"_{offset // 24 + 1:02d}" if len(keys) > 24 else ""
                graphs.append(_accuracy(series, data, subset, output, name + suffix, title, plt))
        keys = list(series)
        for offset in range(0, len(keys), 24):
            subset = {k: series[k] for k in keys[offset:offset + 24]}
            name = _correctness(subset, data, output, plt)
            if len(keys) > 24:
                new_name = name + f"_{offset // 24 + 1:02d}"
                (output / (name + ".png")).rename(output / (new_name + ".png"))
                name = new_name
            graphs.append(name)
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
