"""Deterministic in-code sample data and the labelled-dataset contract for ESM-2 sequence classification.

The tutorial task is synthetic and deliberately order-sensitive: every sequence carries the same
number of strongly hydrophobic residues, but in class `segment` they form one contiguous stretch
(18-22 residues, the length range of a membrane-spanning helix) and in class `scattered` they are
spread out so that no hydrophobic run is longer than 5. A model that only counts residues cannot
separate the classes; a model that reads the sequence can. This is sanity evidence for the
fine-tuning contract, not a biological benchmark (NOTEBOOK_SPEC 2.0 DAT8).
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import ALPHABET, MAX_RESIDUES, STANDARD_AMINO_ACIDS

DATASET_REPRESENTATION = "io.github.kurtvalcorza.dataset.protein.sequence-labels.v1"
SAMPLE_CLASSES: tuple[str, ...] = ("scattered", "segment")
SAMPLE_SEED = 20260918
SAMPLE_SIZE = 96  # 48 per class
MIN_RECORDS = 12
MAX_RECORDS = 5_000
MAX_CLASSES = 20
MIN_RECORDS_PER_CLASS = 3
MAX_ID_CHARS = 64
MAX_LABEL_CHARS = 64
REQUIRED_COLUMNS = ("id", "sequence", "label")

# Residues counted as strongly hydrophobic for the synthetic rule (Kyte-Doolittle > 1.8 plus W).
HYDROPHOBIC = "AILMFVW"
# Background composition (approximate UniProt frequencies, %); the generator samples from it.
_BACKGROUND = {
    "A": 8.3,
    "R": 5.5,
    "N": 4.1,
    "D": 5.5,
    "C": 1.4,
    "Q": 3.9,
    "E": 6.7,
    "G": 7.1,
    "H": 2.3,
    "I": 5.9,
    "L": 9.7,
    "K": 5.8,
    "M": 2.4,
    "F": 3.9,
    "P": 4.7,
    "S": 6.6,
    "T": 5.4,
    "W": 1.1,
    "Y": 2.9,
    "V": 6.9,
}
_POLAR = "".join(ch for ch in STANDARD_AMINO_ACIDS if ch not in HYDROPHOBIC)
_SEGMENT_LENGTH = (18, 22)
_SEQUENCE_LENGTH = (60, 120)
_MAX_SCATTERED_RUN = 5


def _draw_background(rng: random.Random, n: int) -> list[str]:
    letters = list(_BACKGROUND)
    weights = [_BACKGROUND[ch] for ch in letters]
    return rng.choices(letters, weights=weights, k=n)


def longest_hydrophobic_run(sequence: str) -> int:
    """Length of the longest contiguous run of residues in HYDROPHOBIC."""
    best = run = 0
    for ch in sequence:
        run = run + 1 if ch in HYDROPHOBIC else 0
        best = max(best, run)
    return best


def hydrophobic_fraction(sequence: str) -> float:
    return sum(ch in HYDROPHOBIC for ch in sequence) / len(sequence) if sequence else 0.0


def _make_segment(rng: random.Random) -> str:
    length = rng.randint(*_SEQUENCE_LENGTH)
    seg_len = rng.randint(*_SEGMENT_LENGTH)
    body = _draw_background(rng, length)
    # Polar background outside the segment keeps the hydrophobic count equal between classes.
    body = [ch if ch not in HYDROPHOBIC else rng.choice(_POLAR) for ch in body]
    start = rng.randint(2, length - seg_len - 2)
    segment = rng.choices(HYDROPHOBIC, k=seg_len)
    body[start : start + seg_len] = segment
    return "".join(body)


def _make_scattered(rng: random.Random, n_hydrophobic: int, length: int) -> str:
    for _ in range(1000):
        body = [rng.choice(_POLAR) for _ in range(length)]
        positions = rng.sample(range(length), n_hydrophobic)
        for pos in positions:
            body[pos] = rng.choice(HYDROPHOBIC)
        seq = "".join(body)
        if longest_hydrophobic_run(seq) <= _MAX_SCATTERED_RUN:
            return seq
    raise RuntimeError("could not place scattered hydrophobic residues without a long run")


def generate_sample_dataset(seed: int = SAMPLE_SEED, size: int = SAMPLE_SIZE) -> list[dict[str, Any]]:
    """`size` labelled records (half `segment`, half `scattered`), deterministic for a given seed.

    Each `scattered` record mirrors one `segment` record's length and hydrophobic residue count,
    so the two classes have identical composition statistics by construction.
    """
    if size < 2 or size % 2:
        raise ValueError("size must be an even number >= 2 (one scattered record per segment record)")
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for i in range(size // 2):
        seg = _make_segment(rng)
        n_h = sum(ch in HYDROPHOBIC for ch in seg)
        sca = _make_scattered(rng, n_h, len(seg))
        records.append({"id": f"seg-{i:03d}", "sequence": seg, "label": "segment"})
        records.append({"id": f"sca-{i:03d}", "sequence": sca, "label": "scattered"})
    return records


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 over the canonical (id, sequence, label) rows; recorded in provenance (OUT9)."""
    canon = json.dumps([[r["id"], r["sequence"], r["label"]] for r in records], separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    classes: Sequence[str] | None = None,
    min_records: int = MIN_RECORDS,
    min_per_class: int = MIN_RECORDS_PER_CLASS,
) -> dict[str, Any]:
    """Check a labelled dataset against the sequence-classification contract; return its manifest.

    Every error names the record and the violated rule (VAL4/DAT19). When `classes` is given the
    labels must be drawn from exactly that list (used for validation/test splits and BYOD inference
    against an adapted head); otherwise the sorted set of observed labels becomes the class list.
    """
    if isinstance(records, str | bytes | Mapping) or not isinstance(records, Sequence):
        raise TypeError("records must be a list of {'id', 'sequence', 'label'} mappings")
    if len(records) < min_records:
        raise ValueError(f"dataset has {len(records)} records; at least {min_records} are required")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"dataset has {len(records)} records; ceiling is {MAX_RECORDS}")
    seen_ids: set[str] = set()
    seen_sequences: dict[str, str] = {}
    counts: dict[str, int] = {}
    lengths: list[int] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise TypeError(f"record[{i}] must be a mapping, got {type(rec).__name__}")
        missing = [c for c in REQUIRED_COLUMNS if c not in rec]
        if missing:
            raise ValueError(
                f"record[{i}] is missing required column(s) {missing}; required: {list(REQUIRED_COLUMNS)}"
            )
        rid = str(rec["id"]).strip()
        if not rid or len(rid) > MAX_ID_CHARS:
            raise ValueError(f"record[{i}] id must be 1..{MAX_ID_CHARS} characters")
        if rid in seen_ids:
            raise ValueError(f"record[{i}] duplicates id {rid!r}")
        seen_ids.add(rid)
        seq = rec["sequence"]
        if not isinstance(seq, str) or not seq:
            raise ValueError(f"record[{i}] ({rid}) sequence must be a non-empty string")
        if len(seq) > MAX_RESIDUES:
            raise ValueError(f"record[{i}] ({rid}) has {len(seq)} residues; ceiling is {MAX_RESIDUES}")
        bad = sorted({ch for ch in seq if ch not in ALPHABET})
        if bad:
            raise ValueError(
                f"record[{i}] ({rid}) sequence contains characters outside the amino-acid alphabet: {bad!r}"
            )
        if seq in seen_sequences:
            raise ValueError(f"record[{i}] ({rid}) duplicates the sequence of {seen_sequences[seq]!r}")
        seen_sequences[seq] = rid
        label = rec["label"]
        if not isinstance(label, str) or not label.strip() or len(label) > MAX_LABEL_CHARS:
            raise ValueError(
                f"record[{i}] ({rid}) label must be a non-empty string of at most {MAX_LABEL_CHARS} chars"
            )
        counts[label] = counts.get(label, 0) + 1
        lengths.append(len(seq))
    if classes is None:
        class_list = sorted(counts)
    else:
        class_list = [str(c) for c in classes]
        unknown = sorted(set(counts) - set(class_list))
        if unknown:
            raise ValueError(f"labels {unknown} are not in the class list {class_list}")
    if len(class_list) < 2:
        raise ValueError(f"classification needs at least 2 classes, found {class_list}")
    if len(class_list) > MAX_CLASSES:
        raise ValueError(f"{len(class_list)} classes exceeds the ceiling of {MAX_CLASSES}")
    thin = [c for c in class_list if counts.get(c, 0) < min_per_class]
    if thin:
        raise ValueError(f"classes {thin} have fewer than {min_per_class} records each (class coverage rule)")
    return {
        "verdict": "accepted",
        "representation": DATASET_REPRESENTATION,
        "n_records": len(records),
        "classes": class_list,
        "class_counts": {c: counts.get(c, 0) for c in class_list},
        "residues": {"min": min(lengths), "max": max(lengths), "mean": round(sum(lengths) / len(lengths), 1)},
        "ceilings": {
            "max_residues": MAX_RESIDUES,
            "max_records": MAX_RECORDS,
            "max_classes": MAX_CLASSES,
            "min_records": min_records,
            "min_records_per_class": min_per_class,
        },
        "digest": dataset_digest(records),
        "findings": [],
    }


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Stratified random train/validation/test split (assumes rows are independent, SPL3).

    Each class is shuffled with `seed` and cut in the given proportions, so every split keeps the
    class proportions and every class has at least one record per split when the data allow it.
    """
    if not (0.0 < val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must be in (0, 1) and sum to less than 1")
    manifest = validate_dataset(records)
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = {c: [] for c in manifest["classes"]}
    for rec in records:
        by_class[rec["label"]].append(dict(rec))
    out: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for cls in manifest["classes"]:
        rows = by_class[cls]
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * val_fraction))
        n_test = max(1, round(len(rows) * test_fraction))
        if len(rows) - n_val - n_test < 1:
            raise ValueError(f"class {cls!r} has {len(rows)} records; too few to leave one per split")
        out["validation"].extend(rows[:n_val])
        out["test"].extend(rows[n_val : n_val + n_test])
        out["train"].extend(rows[n_val + n_test :])
    for part in out.values():
        rng.shuffle(part)
    return out


def load_byod_dataset(source: str | Path) -> list[dict[str, Any]]:
    """Read a user-supplied CSV (`id,sequence,label` header), JSON array, or JSONL file into records.

    Sequences are stripped of surrounding whitespace only; nothing else is rewritten (VAL7).
    The records are then validated with `validate_dataset`, whose errors name the offending row.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"BYOD dataset file not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"BYOD dataset file is empty: {path}")
    suffix = path.suffix.lower()
    records: list[dict[str, Any]] = []
    if suffix == ".csv":
        reader = csv.DictReader(text.splitlines())
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"CSV header {header} is missing required column(s) {missing}")
        for row in reader:
            records.append({c: (row.get(c) or "").strip() for c in REQUIRED_COLUMNS})
    elif suffix == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no} is not valid JSON: {exc}") from exc
            records.append(item)
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"file is not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise TypeError("JSON dataset must be a top-level array of objects")
        records = data
    else:
        raise ValueError(f"unsupported BYOD file type {suffix!r}; use .csv, .json or .jsonl")
    for rec in records:
        if isinstance(rec, Mapping) and isinstance(rec.get("sequence"), str):
            rec["sequence"] = rec["sequence"].strip()
    validate_dataset(records)
    return [dict(r) for r in records]


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write records as the BYOD CSV shape (`id,sequence,label`), so users have a template."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(REQUIRED_COLUMNS)
        for r in records:
            writer.writerow([r["id"], r["sequence"], r["label"]])
    return out
