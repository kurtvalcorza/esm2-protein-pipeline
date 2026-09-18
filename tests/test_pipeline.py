"""Offline pipeline tests: identity, snapshot verification/staging, input checks, injected backends."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from esm2_protein_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    HIDDEN_SIZE,
    MANIFEST_NAME,
    MAX_RESIDUES,
    MAX_SEQUENCES_PER_CALL,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    ESM2Pipeline,
    stage_missing_files,
    validate_inputs,
    verify_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
SEQS = ["MKTAYIAKQRQISFVKSHFSRQ", "GGGSSSLLLVVVAAAIIIFFF"]


def _fake(classes=None, logits=None):
    def embedder(sequences):
        return [[float(len(s))] * HIDDEN_SIZE for s in sequences]

    pipe = ESM2Pipeline(embedder, "cpu")
    if classes is not None:
        pipe.classes = list(classes)
        rows = logits or [[0.0, 1.0] for _ in range(MAX_SEQUENCES_PER_CALL)]
        pipe._classifier = lambda sequences: [rows[i] for i in range(len(sequences))]
    return pipe


def test_identity_constants_are_40_hex_and_match_manifest():
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["modelId"] == MODEL_ID
        assert manifest["revision"] == MODEL_REVISION
        assert {f["path"] for f in manifest["files"]} >= {"config.json", "model.safetensors", "vocab.txt"}
    cfg = REPO / "weights" / MODEL_KEY / "config.json"
    if cfg.is_file():
        config = json.loads(cfg.read_text(encoding="utf-8"))
        assert config["hidden_size"] == HIDDEN_SIZE
        assert config["max_position_embeddings"] >= MAX_RESIDUES + 2


def _write_snapshot(tmp_path: Path, content: bytes, sha256: str, revision: str = MODEL_REVISION) -> Path:
    (tmp_path / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [{"path": "config.json", "bytes": len(content), "sha256": sha256}],
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    content = b'{"hidden_size": 640}'
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    assert verify_snapshot(root)["revision"] == MODEL_REVISION


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"hidden_size": 640}'
    digest = hashlib.sha256(content).hexdigest()
    flipped = ("1" if digest[0] != "1" else "2") + digest[1:]
    root = _write_snapshot(tmp_path, content, flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)


def test_verify_snapshot_rejects_wrong_revision_and_missing_file(tmp_path):
    content = b"{}"
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest(), revision="0" * 40)
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    (root / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path):
    content = b"{}"
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["files"].append({"path": "vocab.txt", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()})
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    fetched = []

    def downloader(rel, dst):
        fetched.append(rel)
        (dst / rel).write_bytes(b"abc")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == ["vocab.txt"]
    assert fetched == ["vocab.txt"]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    assert verify_snapshot(root)["files"][1]["path"] == "vocab.txt"


def test_stage_refuses_manifest_for_another_model(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"modelId": "other/model", "revision": MODEL_REVISION, "files": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True)


def test_validate_inputs_rejections_name_the_rule():
    with pytest.raises(TypeError, match="list of str"):
        validate_inputs("MKT")
    with pytest.raises(TypeError, match="must be str"):
        validate_inputs([["MKT"]])
    with pytest.raises(ValueError, match="empty"):
        validate_inputs([""])
    with pytest.raises(ValueError, match=f"ceiling is {MAX_RESIDUES}"):
        validate_inputs(["A" * (MAX_RESIDUES + 1)])
    with pytest.raises(ValueError, match="outside the amino-acid alphabet"):
        validate_inputs(["mkt"])
    with pytest.raises(ValueError, match="outside the amino-acid alphabet"):
        validate_inputs(["MK-T"])
    with pytest.raises(ValueError, match=f"1..{MAX_SEQUENCES_PER_CALL}"):
        validate_inputs(["A"] * (MAX_SEQUENCES_PER_CALL + 1))
    with pytest.raises(ValueError, match="exactly one id"):
        validate_inputs(SEQS, names=["a"])
    with pytest.raises(ValueError, match="unique"):
        validate_inputs(SEQS, names=["a", "a"])


def test_validate_inputs_accepts_extended_codes_and_records_them():
    manifest = validate_inputs(["MKTXBUZO"], names=["p1"])
    assert manifest["verdict"] == "accepted"
    assert manifest["inputs"] == [{"id": "p1", "residues": 8, "non_standard_residues": 5}]


def test_embed_contract_with_injected_backend():
    pipe = _fake()
    out = pipe.embed(SEQS, names=["a", "b"])
    assert out["ids"] == ["a", "b"]
    assert out["dimension"] == HIDDEN_SIZE
    assert len(out["embeddings"]) == 2 and len(out["embeddings"][0]) == HIDDEN_SIZE
    assert out["embeddings"][0][0] == float(len(SEQS[0]))
    assert out["model_revision"] == MODEL_REVISION


def test_embed_rejects_backend_shape_drift():
    pipe = ESM2Pipeline(lambda sequences: [[0.0] * 3 for _ in sequences], "cpu")
    with pytest.raises(RuntimeError, match="wrong shape"):
        pipe.embed(SEQS)


def test_classify_requires_adaptation():
    with pytest.raises(RuntimeError, match="adapt"):
        _fake().classify(SEQS)


def test_classify_contract_preserves_class_order_and_scores():
    pipe = _fake(classes=["scattered", "segment"], logits=[[2.0, 0.0], [0.0, 2.0]])
    out = pipe.classify(SEQS, names=["a", "b"])
    labels = [p["label"] for p in out["predictions"]]
    assert labels == ["scattered", "segment"]
    scores = out["predictions"][0]["scores"]
    assert list(scores) == ["scattered", "segment"]
    assert scores["scattered"] == pytest.approx(0.8808, abs=1e-3)
    assert out["predictions"][0]["score"] == pytest.approx(scores["scattered"])
    assert "not calibrated" in out["decision_rule"]


def test_classify_rejects_logits_that_do_not_match_classes():
    pipe = _fake(classes=["a", "b", "c"], logits=[[0.0, 1.0]] * 4)
    with pytest.raises(RuntimeError, match="does not match the class list"):
        pipe.classify(SEQS)


def test_save_and_load_artifact_require_an_adapted_model(tmp_path):
    pipe = _fake()
    with pytest.raises(RuntimeError, match="adapt"):
        pipe.save_artifact(tmp_path / "adapter")
    with pytest.raises(RuntimeError, match="from_pretrained"):
        pipe.load_artifact(tmp_path / "adapter")
