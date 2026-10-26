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


def audit(reference: Path, items: list[Item] | None = None) -> dict:
    """Summarize published scores and flag source-key disagreements."""
    rows = read_reference(reference)
    result = {
        "kind": "published_reference_audit_not_a_new_metamo_run",
        "reference_sha256": sha256(reference), "item_count": len(rows),
        "columns": list(rows[0]), "conditions": {},
        "missing_inputs_in_csv": ["original questions", "five answer choices", "complete independent answer key"],
        "unrecoverable_key_ids_from_csv_alone": [
            int(row["STEU"]) for row in rows
            if not any(row[c + " Score"] == "1" for c in CONDITIONS.values())
        ],
        "paired_changes": {},
    }
    for condition, column in CONDITIONS.items():
        correct = sum(int(row[column + " Score"]) for row in rows)
        entry = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows),
                 "incorrect_ids": [int(row["STEU"]) for row in rows if row[column + " Score"] == "0"]}
        if items:
            entry["source_rule_correct"] = sum(row[column] in item.source_answers for row, item in zip(rows, items))
            entry["source_score_disagreements"] = [
                item.id for row, item in zip(rows, items)
                if int(row[column + " Score"]) != int(row[column] in item.source_answers)
            ]
        result["conditions"][condition] = entry
    for left, right in [("no_memory", "memory"), ("memory", "full_metamo"), ("no_memory", "full_metamo")]:
        a, b = CONDITIONS[left] + " Score", CONDITIONS[right] + " Score"
        result["paired_changes"][f"{right}_vs_{left}"] = {
            "improved_ids": [int(r["STEU"]) for r in rows if r[a] == "0" and r[b] == "1"],
            "regressed_ids": [int(r["STEU"]) for r in rows if r[a] == "1" and r[b] == "0"],
        }
    if items:
        result["source_url"] = SOURCE_URL
        result["source_vs_provisional_key_differences"] = [i.id for i in items if i.source_answers != i.reference_answers]
        result["source_multiple_accepted_answers"] = {str(i.id): i.source_answers for i in items if len(i.source_answers) > 1}
    return result


def write_audit(result: dict, destination: Path) -> None:
    """Save the reference audit as JSON and a readable report."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "reference_audit.json").write_text(json.dumps(result, indent=2) + "\n")
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
    (destination / "reference_audit.md").write_text("\n".join(lines))
