"""Read STEU materials without exposing reference answers to model prompts."""

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


SOURCE_URL = "https://www.researchgate.net/publication/233864930_MacCann_Roberts_STEU"
CONDITIONS = {"no_memory": "No Memory", "memory": "Memory", "full_metamo": "Appraisal"}
LETTERS = "ABCDE"


@dataclass(frozen=True)
class Item:
    """Store a question, five choices, and separate scoring keys."""

    id: int
    question: str
    options: dict[str, str]
    source_answers: list[str]
    reference_answers: list[str]

    def public(self) -> dict:
        """Return question inputs without scoring keys."""
        return {"id": self.id, "question": self.question, "options": self.options}


def sha256(path: Path) -> str:
    """Fingerprint a file's contents for reproducibility."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_reference(path: Path) -> list[dict]:
    """Load and validate the 42 published answer and score records."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"STEU", "Output"}
        for column in CONDITIONS.values():
            required.update({column, column + " Score"})
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Expected the original experiment_1_STEU.csv columns")
        rows = list(reader)
    if len(rows) != 42 or sorted(int(r["STEU"]) for r in rows) != list(range(1, 43)):
        raise ValueError("Expected exactly one reference row for each STEU item 1–42")
    for row in rows:
        for column in CONDITIONS.values():
            if row[column] not in LETTERS or len(row[column]) != 1:
                raise ValueError(f"Invalid answer at item {row['STEU']}")
            if row[column + " Score"] not in {"0", "1"}:
                raise ValueError(f"Invalid score at item {row['STEU']}")
        positives = {row[c] for c in CONDITIONS.values() if row[c + " Score"] == "1"}
        negatives = {row[c] for c in CONDITIONS.values() if row[c + " Score"] == "0"}
        if len(positives) > 1 or positives & negatives:
            raise ValueError(f"Contradictory reference scores at item {row['STEU']}")
    return sorted(rows, key=lambda r: int(r["STEU"]))


def prepare(source: Path, reference: Path, destination: Path) -> list[Item]:
    """Build question records, both scoring keys, and provenance from local files."""
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"SITUATIONAL TEST OF EMOTIONAL UNDERSTANDING Page \d+ of \d+", "", text)
    text = text.replace("[Type text]", "")
    if "Test items" not in text or "Scoring the STEU in SPSS" not in text:
        raise ValueError("Source must contain the 42 test items and SPSS scoring syntax")
    questions = text.split("Test items", 1)[1].split("Scoring the STEU in SPSS", 1)[0]
    blocks = re.findall(r"(?:^|\n)(\d+)\.\s+(.*?)(?=\n\d+\.\s|\Z)", questions, re.S)
    recodes = dict(re.findall(r"RECODE\s+STEU(\d+)\s+(.*?)\s+INTO", text, re.S))
    rows = read_reference(reference)
    items = []
    for item_id, block in blocks:
        item_id = int(item_id)
        split = re.split(r"\(([a-e])\)\s*", block)
        if len(split) != 11 or split[1::2] != list("abcde"):
            raise ValueError(f"Expected five ordered answer choices at item {item_id}")
        normalize = lambda s: " ".join(s.split())
        options = {letter.upper(): normalize(value) for letter, value in zip(split[1::2], split[2::2])}
        syntax = recodes.get(f"{item_id:02d}", "")
        pairs = re.findall(r"\(([1-5])=([01])\)", syntax)
        if len(pairs) != 5 or {p[0] for p in pairs} != set("12345"):
            raise ValueError(f"Incomplete source scoring syntax at item {item_id}")
        source_answers = [LETTERS[int(index) - 1] for index, score in pairs if score == "1"]
        row = rows[item_id - 1]
        positives = {row[c] for c in CONDITIONS.values() if row[c + " Score"] == "1"}
        negatives = {row[c] for c in CONDITIONS.values() if row[c + " Score"] == "0"}
        reference_answers = sorted(positives or (set(source_answers) - negatives))
        if not source_answers or not reference_answers:
            raise ValueError(f"Cannot establish provisional scoring for item {item_id}")
        items.append(Item(item_id, normalize(split[0]), options, source_answers, reference_answers))
    if [item.id for item in items] != list(range(1, 43)):
        raise ValueError("Source must contain all 42 test items in order")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in items))
    manifest = {
        "source_url": SOURCE_URL, "source_sha256": sha256(source),
        "reference_sha256": sha256(reference), "items_sha256": sha256(destination),
        "source_scoring": "Literal numeric SPSS RECODE rules, including multiple accepted answers",
        "reference_scoring": "Provisional CSV-compatible key; not independently validated",
    }
    destination.with_suffix(".provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return items


def load_items(path: Path) -> list[Item]:
    """Load and validate all 42 prepared question records."""
    items = [Item(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    if [item.id for item in items] != list(range(1, 43)):
        raise ValueError("Expected all 42 items in numerical order")
    for item in items:
        if set(item.options) != set(LETTERS) or not item.question.strip():
            raise ValueError(f"Invalid question/options at item {item.id}")
        for answers in [item.source_answers, item.reference_answers]:
            if not answers or not set(answers).issubset(LETTERS):
                raise ValueError(f"Invalid scoring key at item {item.id}")
    return items


def _summarize_scores(scores: dict[str, dict[int, int]], order: list[int]) -> dict:
    """Summarize correctness and paired changes for one scoring key."""
    conditions = {}
    for condition in CONDITIONS:
        correct = sum(scores[condition][item_id] for item_id in order)
        conditions[condition] = {
            "correct": correct, "total": len(order), "accuracy": correct / len(order),
            "incorrect_ids": [item_id for item_id in order if scores[condition][item_id] == 0],
        }
    paired = {}
    labels = {(0, 1): "improved_ids", (1, 0): "regressed_ids",
              (1, 1): "unchanged_correct_ids", (0, 0): "unchanged_incorrect_ids"}
    for left, right in [("no_memory", "memory"), ("memory", "full_metamo"), ("no_memory", "full_metamo")]:
        groups = {label: [] for label in labels.values()}
        for item_id in order:
            outcome = (scores[left][item_id], scores[right][item_id])
            groups[labels[outcome]].append(item_id)
        paired[f"{right}_vs_{left}"] = {
            **groups, "net_correct_change": len(groups["improved_ids"]) - len(groups["regressed_ids"]),
        }
    return {"conditions": conditions, "paired_changes": paired}


def audit(reference: Path | None = None, items: list[Item] | None = None, *,
          run: Path | None = None, items_path: Path | None = None) -> dict:
    """Audit either a published CSV or a saved experimental run."""
    if run is not None:
        if reference is not None or items is not None or items_path is None:
            raise ValueError("Experimental audit requires run and items_path only")
        return _audit_experiment(run, items_path)
    if reference is None or items_path is not None:
        raise ValueError("Published audit requires a reference CSV and optional loaded items")
    return _audit_reference(reference, items)


def _audit_reference(reference: Path, items: list[Item] | None) -> dict:
    """Load published scores and check their source-key consistency."""
    rows = read_reference(reference)
    summary = _summarize_scores({
        condition: {int(row["STEU"]): int(row[column + " Score"]) for row in rows}
        for condition, column in CONDITIONS.items()
    }, [int(row["STEU"]) for row in rows])
    result = {
        "kind": "published_reference_audit_not_a_new_metamo_run",
        "reference_sha256": sha256(reference), "item_count": len(rows),
        "columns": list(rows[0]), "conditions": summary["conditions"],
        "missing_inputs_in_csv": ["original questions", "five answer choices", "complete independent answer key"],
        "unrecoverable_key_ids_from_csv_alone": [
            int(row["STEU"]) for row in rows
            if not any(row[c + " Score"] == "1" for c in CONDITIONS.values())
        ],
        "paired_changes": {
            comparison: {key: changes[key] for key in ("improved_ids", "regressed_ids")}
            for comparison, changes in summary["paired_changes"].items()
        },
    }
    for condition, column in CONDITIONS.items():
        entry = result["conditions"][condition]
        if items:
            entry["source_rule_correct"] = sum(row[column] in item.source_answers for row, item in zip(rows, items))
            entry["source_score_disagreements"] = [
                item.id for row, item in zip(rows, items)
                if int(row[column + " Score"]) != int(row[column] in item.source_answers)
            ]
    if items:
        result["source_url"] = SOURCE_URL
        result["source_vs_provisional_key_differences"] = [i.id for i in items if i.source_answers != i.reference_answers]
        result["source_multiple_accepted_answers"] = {str(i.id): i.source_answers for i in items if len(i.source_answers) > 1}
    return result


def write_audit(result: dict, destination: Path) -> None:
    """Save a published or experimental audit as JSON and Markdown."""
    if result["kind"] == "experimental_results_audit":
        name, lines = "experiment_audit", _experimental_audit_lines(result)
    elif result["kind"] == "published_reference_audit_not_a_new_metamo_run":
        name, lines = "reference_audit", _reference_audit_lines(result)
    else:
        raise ValueError("Unknown audit report kind")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / f"{name}.md").write_text("\n".join(lines))


def _reference_audit_lines(result: dict) -> list[str]:
    """Format the published-reference Markdown report."""
    lines = [
        "# Phase 1: published STEU reference audit", "",
        "These are the downloaded paper's recorded results, not new MetaMo measurements.", "",
        "| Published condition | Correct | Accuracy |", "| --- | ---: | ---: |",
    ]
    for condition, entry in result["conditions"].items():
        lines.append(f"| {CONDITIONS[condition]} | {entry['correct']}/{entry['total']} | {entry['accuracy']:.2%} |")
    lines += ["", "The CSV contains responses, binary scores, and appraisal outputs. It does not contain original questions, complete options, or an independent answer key.", "",
              "No correct letter can be recovered from the CSV alone for items " + ", ".join(map(str, result["unrecoverable_key_ids_from_csv_alone"])) + ". Outputs include chosen answers and must never be used as fresh test questions.", ""]
    if "source_url" in result:
        lines += [f"The [author-uploaded STEU instrument]({result['source_url']}) supplies the questions and literal SPSS scoring rules. This source has scoring issues and is not treated as an adjudicated key.", "",
                  "| Published answers re-scored with source rules | Correct | Score disagreement IDs |", "| --- | ---: | --- |"]
        for condition, entry in result["conditions"].items():
            lines.append(f"| {CONDITIONS[condition]} | {entry['source_rule_correct']}/42 | {', '.join(map(str, entry['source_score_disagreements']))} |")
        lines += ["", "Source rules accept multiple answers at: " + json.dumps(result["source_multiple_accepted_answers"]) + ".",
                  "The source also says D=5 in its prose coding example; numeric RECODE rules use 4 for D. The parser uses numeric indices 1–5 in those rules.", "",
                  "Fresh runs report both literal-source scores and a provisional CSV-compatible sensitivity score. The latter uses the 37 positively scored CSV letters and source rules consistent with negative CSV scores for the five remaining items. It is not an independently validated gold key.", ""]
    lines += ["The original paper tested GPT-3.5 prompting strategies. Its 35/42 result does not establish the performance of this MetaMo implementation or Gemini. See [Experiment 1, sections 4.1–4.2](https://doi.org/10.1371/journal.pone.0301033).", ""]
    return lines


def _audit_experiment(run: Path, items_path: Path) -> dict:
    """Validate a completed run and compare conditions under both saved keys."""
    manifest_path = run / "manifest.json"
    responses_path = run / "responses.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Experimental audit requires a completed run; resume the run first")
    if manifest.get("items_sha256") != sha256(items_path):
        raise ValueError("Prepared dataset does not match this run's items_sha256; use --items with the original dataset")
    items = {item.id: item for item in load_items(items_path)}
    order = manifest.get("item_order")
    if (not isinstance(order, list) or not order
            or any(type(item_id) is not int or item_id not in items for item_id in order)
            or len(set(order)) != len(order)):
        raise ValueError("Run manifest has an invalid item_order")
    expected = {(condition, item_id) for condition in CONDITIONS for item_id in order}
    records = {condition: {} for condition in CONDITIONS}
    seen = set()
    score_keys = {"source": "source_correct", "reference_compatible": "reference_compatible_correct"}
    for line in responses_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Saved responses must be JSON objects")
        condition, item_id = record.get("condition"), record.get("item_id")
        if (not isinstance(condition, str) or type(item_id) is not int
                or (condition, item_id) not in expected):
            raise ValueError("Saved response does not match the run's conditions and item_order")
        pair = (condition, item_id)
        if pair in seen:
            raise ValueError(f"Duplicate response at {condition} item {item_id}")
        answer = record.get("answer")
        if not isinstance(answer, str) or answer not in list(LETTERS):
            raise ValueError(f"Invalid answer at {condition} item {item_id}")
        item = items[item_id]
        for field, answers in [("source_correct", item.source_answers),
                               ("reference_compatible_correct", item.reference_answers)]:
            score = record.get(field)
            if type(score) is not int or score not in (0, 1) or score != int(answer in answers):
                raise ValueError(f"Invalid or inconsistent {field} at {condition} item {item_id}")
        records[condition][item_id] = record
        seen.add(pair)
    if seen != expected:
        raise ValueError("Completed run is missing responses for some conditions or items")
    selected = [items[item_id] for item_id in order]
    result = {
        "kind": "experimental_results_audit",
        "run_directory": str(run.resolve()), "model": manifest.get("model"),
        "protocol": manifest.get("protocol"), "item_count": len(order), "item_order": order,
        "scope": "42-item benchmark" if len(order) == 42 else "partial pilot, not a full STEU score",
        "scoring_provisional": True, "items_sha256": manifest["items_sha256"],
        "manifest_sha256": sha256(manifest_path), "responses_sha256": sha256(responses_path),
        "source_vs_provisional_key_differences": [i.id for i in selected if i.source_answers != i.reference_answers],
        "source_multiple_accepted_answers": {str(i.id): i.source_answers for i in selected if len(i.source_answers) > 1},
        "scoring": {},
    }
    for scoring, field in score_keys.items():
        result["scoring"][scoring] = _summarize_scores({
            condition: {item_id: records[condition][item_id][field] for item_id in order}
            for condition in CONDITIONS
        }, order)
    return result


def _experimental_audit_lines(result: dict) -> list[str]:
    """Format the experimental Markdown report."""
    lines = [
        "# Phase 1: experimental results audit", "",
        "These are your saved experimental responses, not the paper's published results.", "",
        f"Run: `{result['run_directory']}`  ",
        f"Model: `{result['model']}`  ",
        f"Scope: {result['scope']} ({result['item_count']} questions per configuration).", "",
        "Scoring remains provisional: source uses literal SPSS rules; reference_compatible uses the CSV-compatible key.",
        "The prepared dataset fingerprint and each saved correctness flag were checked", "",
        "## Scoring-key checks", "",
        "Key differences among the evaluated items: " + json.dumps(result["source_vs_provisional_key_differences"]) + ".", "",
        "Source rules accepting multiple answers among the evaluated items: "
        + json.dumps(result["source_multiple_accepted_answers"]) + ".", "",
        "For X_vs_Y, improved means Y was wrong and X correct; regressed means Y was correct and X wrong.",
        "These describe differences between configurations, not learning during a run or proof of causation.", "",
    ]
    for scoring, entry in result["scoring"].items():
        lines += [f"## {scoring} scoring", "",
                  "| Configuration | Correct | Accuracy | Incorrect IDs |",
                  "| --- | ---: | ---: | --- |"]
        for condition, scores in entry["conditions"].items():
            lines.append(f"| {condition} | {scores['correct']}/{scores['total']} | {scores['accuracy']:.2%} | {scores['incorrect_ids']} |")
        lines += ["", "| Comparison | Improved IDs | Regressed IDs | Net correct change |",
                  "| --- | --- | --- | ---: |"]
        for comparison, changes in entry["paired_changes"].items():
            lines.append(f"| {comparison} | {changes['improved_ids']} | {changes['regressed_ids']} | {changes['net_correct_change']:+d} |")
        lines += ["", "| Comparison | Unchanged correct IDs | Unchanged incorrect IDs |", "| --- | --- | --- |"]
        for comparison, changes in entry["paired_changes"].items():
            lines.append(f"| {comparison} | {changes['unchanged_correct_ids']} | {changes['unchanged_incorrect_ids']} |")
        lines.append("")
    return lines
