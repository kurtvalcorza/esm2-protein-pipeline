"""Offline tests for the public validation-stage helpers (input manifest and dataset manifest)."""

from __future__ import annotations

from esm2_protein_pipeline import (
    INPUT_SCHEMA,
    MAX_RESIDUES,
    MAX_SEQUENCES_PER_CALL,
    MODEL_ID,
    MODEL_REVISION,
    generate_sample_dataset,
    validate_dataset,
    validate_inputs,
)


def test_validate_inputs_returns_manifest_with_schema_and_identity():
    seqs = ["MKTAYIAKQR", "GGSSLLVVAA"]
    manifest = validate_inputs(seqs, names=["p1", "p2"])
    assert manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["sequences"] == [1, MAX_SEQUENCES_PER_CALL]
    assert manifest["schema"]["residues"] == [1, MAX_RESIDUES]
    assert manifest["inputs"] == [
        {"id": "p1", "residues": 10, "non_standard_residues": 0},
        {"id": "p2", "residues": 10, "non_standard_residues": 0},
    ]
    assert manifest["n_sequences"] == 2 and manifest["max_residues_observed"] == 10
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids():
    manifest = validate_inputs(["MKT"])
    assert [entry["id"] for entry in manifest["inputs"]] == ["seq-0"]


def test_dataset_manifest_reports_ceilings_and_digest():
    manifest = validate_dataset(generate_sample_dataset())
    assert manifest["ceilings"]["max_residues"] == MAX_RESIDUES
    assert manifest["residues"]["min"] >= 60 and manifest["residues"]["max"] <= 120
    assert len(manifest["digest"]) == 64
