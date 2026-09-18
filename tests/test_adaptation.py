"""Offline tests for the dataset contract, splits, metrics, baselines and the artifact manifest checks."""

from __future__ import annotations

import csv
import json

import pytest

from esm2_protein_pipeline import (
    ARTIFACT_FORMAT,
    ARTIFACT_MANIFEST_NAME,
    DATASET_REPRESENTATION,
    HYDROPHOBIC,
    MAX_RESIDUES,
    MIN_RECORDS,
    MIN_RECORDS_PER_CLASS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_CLASSES,
    SAMPLE_SEED,
    SAMPLE_SIZE,
    ESM2Pipeline,
    auroc,
    classification_metrics,
    composition_baseline,
    dataset_digest,
    generate_sample_dataset,
    hydrophobic_fraction,
    load_byod_dataset,
    longest_hydrophobic_run,
    majority_baseline,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)

# --- sample dataset ---------------------------------------------------------------------------


def test_sample_dataset_is_deterministic_balanced_and_valid():
    a = generate_sample_dataset()
    b = generate_sample_dataset(seed=SAMPLE_SEED, size=SAMPLE_SIZE)
    assert a == b and len(a) == SAMPLE_SIZE
    manifest = validate_dataset(a)
    assert manifest["verdict"] == "accepted"
    assert manifest["representation"] == DATASET_REPRESENTATION
    assert manifest["classes"] == list(SAMPLE_CLASSES)
    assert manifest["class_counts"] == {"scattered": SAMPLE_SIZE // 2, "segment": SAMPLE_SIZE // 2}
    assert manifest["digest"] == dataset_digest(a)
    assert generate_sample_dataset(seed=7) != a


def test_sample_classes_share_composition_but_differ_in_order():
    records = generate_sample_dataset()
    by_label = {label: [r["sequence"] for r in records if r["label"] == label] for label in SAMPLE_CLASSES}
    for seg, sca in zip(by_label["segment"], by_label["scattered"], strict=True):
        assert len(seg) == len(sca)
        assert sum(c in HYDROPHOBIC for c in seg) == sum(c in HYDROPHOBIC for c in sca)
        assert hydrophobic_fraction(seg) == pytest.approx(hydrophobic_fraction(sca))
        assert 18 <= longest_hydrophobic_run(seg) <= 22
        assert longest_hydrophobic_run(sca) <= 5


# --- validation -------------------------------------------------------------------------------


def _records(n_per_class=4, classes=("x", "y")):
    out = []
    for c_i, c in enumerate(classes):
        for i in range(n_per_class):
            out.append({"id": f"{c}-{i}", "sequence": "MKT" + "A" * (i + 1) + "LV" * (c_i + 1), "label": c})
    return out


def test_validate_dataset_rejections_are_actionable():
    good = _records(n_per_class=MIN_RECORDS)
    assert validate_dataset(good)["classes"] == ["x", "y"]
    with pytest.raises(TypeError, match="list of"):
        validate_dataset({"id": 1})
    with pytest.raises(ValueError, match=f"at least {MIN_RECORDS}"):
        validate_dataset(good[:3])
    bad = [dict(r) for r in good]
    del bad[0]["label"]
    with pytest.raises(ValueError, match=r"missing required column\(s\) \['label'\]"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[1]["id"] = bad[0]["id"]
    with pytest.raises(ValueError, match="duplicates id"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[2]["sequence"] = "MKT a"
    with pytest.raises(ValueError, match="outside the amino-acid alphabet"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[3]["sequence"] = "A" * (MAX_RESIDUES + 1)
    with pytest.raises(ValueError, match=f"ceiling is {MAX_RESIDUES}"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[4]["sequence"] = bad[5]["sequence"]
    with pytest.raises(ValueError, match="duplicates the sequence"):
        validate_dataset(bad)
    one_class = [dict(r, label="x") for r in good]
    with pytest.raises(ValueError, match="at least 2 classes"):
        validate_dataset(one_class)
    thin = good + [{"id": "z-0", "sequence": "MKTWWW", "label": "z"}]
    with pytest.raises(ValueError, match=f"fewer than {MIN_RECORDS_PER_CLASS}"):
        validate_dataset(thin)
    with pytest.raises(ValueError, match="not in the class list"):
        validate_dataset(good, classes=["x"])


def test_split_is_stratified_disjoint_and_seeded():
    records = generate_sample_dataset()
    s1 = split_dataset(records, seed=42)
    s2 = split_dataset(records, seed=42)
    assert s1 == s2
    assert split_dataset(records, seed=1) != s1
    ids = [r["id"] for part in s1.values() for r in part]
    assert len(ids) == len(set(ids)) == len(records)
    for part in s1.values():
        labels = [r["label"] for r in part]
        assert labels.count("segment") == labels.count("scattered")
    assert {k: len(v) for k, v in s1.items()} == {"train": 56, "validation": 20, "test": 20}
    with pytest.raises(ValueError, match="sum to less than 1"):
        split_dataset(records, val_fraction=0.6, test_fraction=0.5)


# --- metrics ----------------------------------------------------------------------------------


def test_auroc_handles_ties_and_degenerate_labels():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert auroc([1, 1], [0.1, 0.2]) is None


def test_classification_metrics_binary():
    y = ["scattered", "segment", "segment", "scattered"]
    p = ["scattered", "segment", "scattered", "scattered"]
    scores = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.5, 0.5]]  # one negative outranks one positive
    m = classification_metrics(y, p, scores, ["scattered", "segment"])
    assert m["n"] == 4 and m["accuracy"] == 0.75
    assert m["per_class"]["segment"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
        "support": 2,
        "predicted": 1,
    }
    assert m["macro_f1"] == pytest.approx((0.8 + 0.6667) / 2, abs=1e-4)
    assert m["auroc"] == 0.75 and "positive class 'segment'" in m["auroc_definition"]
    with pytest.raises(ValueError, match="outside the class list"):
        classification_metrics(y, ["other"] * 4, None, ["scattered", "segment"])
    with pytest.raises(ValueError, match="one column per class"):
        classification_metrics(y, p, [[0.5]] * 4, ["scattered", "segment"])


def test_classification_metrics_multiclass_macro_auroc():
    y = ["a", "b", "c", "a", "b", "c"]
    scores = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ]
    m = classification_metrics(y, y, scores, ["a", "b", "c"])
    assert m["accuracy"] == 1.0 and m["auroc"] == 1.0
    assert "one-vs-rest" in m["auroc_definition"]


def test_baselines_fit_on_train_only():
    records = generate_sample_dataset()
    splits = split_dataset(records, seed=42)
    classes = validate_dataset(records)["classes"]
    maj = majority_baseline(splits["train"], splits["test"], classes)
    assert maj["baseline"] == "majority-class" and maj["accuracy"] == 0.5
    comp = composition_baseline(splits["train"], splits["test"], classes)
    assert comp["baseline"] == "hydrophobic-fraction threshold"
    assert 0.0 <= comp["accuracy"] <= 1.0 and comp["auroc"] is not None
    # By construction the classes share their composition, so the baseline cannot separate them well.
    assert comp["accuracy"] <= 0.7
    with pytest.raises(ValueError, match="binary"):
        composition_baseline(splits["train"], splits["test"], ["a", "b", "c"])


# --- BYOD loaders -----------------------------------------------------------------------------


def test_byod_csv_roundtrip_and_rejections(tmp_path):
    records = generate_sample_dataset(size=24)
    path = write_dataset_csv(records, tmp_path / "data.csv")
    with open(path, encoding="utf-8", newline="") as fh:
        assert next(csv.reader(fh)) == ["id", "sequence", "label"]
    loaded = load_byod_dataset(path)
    assert loaded == records
    (tmp_path / "bad.csv").write_text("id,seq,label\n1,MKT,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing required column\(s\) \['sequence'\]"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_byod_dataset(tmp_path / "empty.csv")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "nope.csv")
    tsv = tmp_path / "data.tsv"
    tsv.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported BYOD file type"):
        load_byod_dataset(tsv)


def test_byod_json_and_jsonl(tmp_path):
    records = generate_sample_dataset(size=24)
    (tmp_path / "d.json").write_text(json.dumps(records), encoding="utf-8")
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    assert load_byod_dataset(tmp_path / "d.json") == records
    assert load_byod_dataset(tmp_path / "d.jsonl") == records
    (tmp_path / "obj.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(TypeError, match="top-level array"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "broken.jsonl").write_text('{"id": 1}\n{bad\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        load_byod_dataset(tmp_path / "broken.jsonl")


# --- artifact manifest checks (no model needed to reject) --------------------------------------


def _fake_loaded():
    pipe = ESM2Pipeline(lambda sequences: [[0.0] * 640 for _ in sequences], "cpu")
    pipe.model = type("M", (), {"config": type("C", (), {"_name_or_path": "x"})()})()
    pipe.tokenizer = object()
    return pipe


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path):
    pipe = _fake_loaded()
    art = tmp_path / "adapter"
    art.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        pipe.load_artifact(art)
    base = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    (art / ARTIFACT_MANIFEST_NAME).write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({"format": ARTIFACT_FORMAT, "base_model": {**base, "model_revision": "0" * 40}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="this package pins"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({"format": ARTIFACT_FORMAT, "base_model": base, "classes": ["only"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="at least two unique classes"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format": ARTIFACT_FORMAT,
                "base_model": base,
                "classes": ["a", "b"],
                "files": [{"path": "adapter.safetensors", "bytes": 1, "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="artifact file missing"):
        pipe.load_artifact(art)
    (art / "adapter.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        pipe.load_artifact(art)
