"""Regression tests for the esm2_protein_colab review fixes (ESM-M1..M4, ESM-m1..m3).

They run within CI's install budget (pytest only: no torch, no NumPy, no weights). Adaptation is exercised up to the
point where it would import the model libraries (the `forbid_model_imports` fixture turns that import into an
assertion), evaluation runs on an injected classifier, and the notebook's own Section 4 cell is executed with the
carried package functions and a stub pipeline. None of this is model or clean-runtime evidence.
"""
# ruff: noqa: E501  -- test cases quote notebook source lines and refusal messages in full

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import esm2_protein_pipeline as package
from esm2_protein_pipeline import (
    HIDDEN_SIZE,
    ESM2Pipeline,
    composition_baseline,
    generate_sample_dataset,
    load_byod_dataset,
    longest_hydrophobic_run,
    longest_run_baseline,
    residue_composition,
    residue_composition_baseline,
    split_dataset,
    validate_dataset,
    validate_splits,
    write_dataset_csv,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "esm2_protein_colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _cell(nb: dict, marker: str) -> str:
    found = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(found) == 1, marker
    return found[0]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _three_class(n_per_class: int = 10, seed: int = 5) -> list[dict[str, Any]]:
    """`n_per_class` records each of segment, scattered and a third, charged class (the review's P4 shape)."""
    rng = random.Random(seed)
    records = [dict(r) for r in generate_sample_dataset(seed=11, size=2 * n_per_class)]
    for i in range(n_per_class):
        body = "".join(rng.choice("KREDKREDSTNQG") for _ in range(rng.randint(60, 100)))
        records.append({"id": f"chg-{i:03d}", "sequence": "M" + body, "label": "charged"})
    return records


def _stub_classifier_pipe(classes: list[str]) -> ESM2Pipeline:
    pipe = ESM2Pipeline(lambda seqs: [[0.0] * HIDDEN_SIZE for _ in seqs], "cpu")
    pipe.classes = list(classes)
    pipe._classifier = lambda seqs: [[float(k == 0) for k in range(len(classes))] for _ in seqs]
    return pipe


# --- ESM-M1: isolated runtime, no in-kernel install ---------------------------------------------------------------


def test_exactly_two_kernel_cells_and_a_hash_locked_isolated_install(nb: dict) -> None:
    kernel = [_src(c) for c in _code_cells(nb) if "# dimer: kernel cell" in _src(c)]
    assert len(kernel) == 2
    install = kernel[0]
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256", 'platform.machine() != "x86_64"'):
        assert needed in install
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in kernel[1]
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    for pin in ("torch==2.14.0", "transformers==4.57.6", "numpy==2.5.3", "safetensors==0.8.0", "huggingface-hub==0.36.2"):
        assert pin in lock
    assert "esm2-protein-pipeline (pyproject.toml)" in lock and "ast-audio" not in lock


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(nb: dict, monkeypatch: pytest.MonkeyPatch, real_google: bool) -> None:
    """find_spec("google.colab") (accelerate does this) must not raise on the worker's stubs (fleet Colab failure)."""
    namespace: dict[str, Any] = {"SKIP_INSTALL": True, "__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_src(_code_cells(nb)[1]), "router", "exec"), namespace)
    worker = namespace["_WORKER_SOURCE"]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    for name in ("google", "google.colab", "google.colab.files"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    shim_globals = {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None}
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), shim_globals)
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google.colab", "google.colab.files"):
            sys.modules.pop(name, None)


def test_release_record_no_longer_counts_the_restarted_run_as_a_pass() -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "after the expected fresh-process restart" not in text
    assert "One expected fresh-process restart followed the install cell" not in text
    assert text.count("**Passed only after a manual restart** — not a one-pass Run all, not promotion evidence.") == 2
    assert "`restarted: false`" in text
    assert "Current status: **Candidate" in (ROOT / "STATUS.md").read_text(encoding="utf-8")
    for name in ("README.md", "STATUS.md", "tutorials/README.md"):
        assert "Release-grade** —" not in (ROOT / name).read_text(encoding="utf-8"), name


# --- ESM-M2: baselines that show what the comparison can and cannot say ---------------------------------------------


def test_order_blind_and_order_aware_baselines_on_the_default_split() -> None:
    records = generate_sample_dataset()
    splits = split_dataset(records, seed=42)
    classes = validate_dataset(records)["classes"]
    fraction = composition_baseline(splits["train"], splits["test"], classes)
    centroid = residue_composition_baseline(splits["train"], splits["test"], classes)
    rule = longest_run_baseline(splits["train"], splits["test"], classes)
    # The equalised feature carries nothing; the full composition does; the order rule defines the classes.
    assert (fraction["accuracy"], fraction["auroc"], fraction["rule"]) == (0.45, 0.47, "predict 'segment' when fraction < 0.2065")
    assert centroid["accuracy"] == 0.75 and centroid["baseline"] == "20-residue composition nearest centroid"
    assert rule["accuracy"] == 1.0 and rule["train_accuracy"] == 1.0
    assert rule["rule"] == "predict 'segment' when longest hydrophobic run >= 18"
    by_label = {c: [longest_hydrophobic_run(r["sequence"]) for r in records if r["label"] == c] for c in classes}
    assert max(by_label["scattered"]) <= 5 and min(by_label["segment"]) >= 18


def test_residue_composition_baseline_is_order_blind_and_multiclass() -> None:
    records = _three_class()
    splits = split_dataset(records, seed=42)
    classes = validate_dataset(records)["classes"]
    base = residue_composition_baseline(splits["train"], splits["test"], classes)
    shuffled = []
    rng = random.Random(0)
    for r in splits["test"]:
        letters = list(r["sequence"])
        rng.shuffle(letters)
        shuffled.append({**r, "sequence": "".join(letters)})
    again = residue_composition_baseline(splits["train"], shuffled, classes)
    assert again["accuracy"] == base["accuracy"] and again["auroc"] == base["auroc"]
    assert len(residue_composition("ACDA")) == 20 and residue_composition("AA")[0] == 1.0
    with pytest.raises(ValueError, match="binary"):
        longest_run_baseline(splits["train"], splits["test"], classes)


def test_learner_text_no_longer_claims_equal_composition_or_credits_pretraining(nb: dict) -> None:
    md = _markdown(nb)
    for stale in ("share their residue composition", "share their composition", "pretrained to represent", "A residue-counting baseline cannot separate the classes"):
        assert stale not in md, stale
    assert "it does **not** show what ESM-2's pretraining contributes" in md
    assert "`longest_hydrophobic_run >= 18`" in md
    code = _cell(nb, "baseline_majority = majority_baseline(")
    assert "residue_composition_baseline(train_records, test_records, CLASSES)" in code
    assert "longest_run_baseline(train_records, test_records, CLASSES)" in code
    for module in ("samples.py", "metrics.py"):
        text = (ROOT / "src" / "esm2_protein_pipeline" / module).read_text(encoding="utf-8")
        assert "identical composition" not in text and "share\ntheir composition by construction" not in text


# --- ESM-M3: one BYOD size contract; no stale head after a dataset change -------------------------------------------


def test_thirty_record_three_class_dataset_meets_every_split_minimum(forbid_model_imports) -> None:
    records = _three_class()
    manifest = validate_dataset(records)
    splits = split_dataset(records, seed=42)
    assert {k: len(v) for k, v in splits.items()} == {"train": 18, "validation": 6, "test": 6}
    manifests = validate_splits(splits, manifest["classes"])
    assert all(m["classes"] == ["charged", "scattered", "segment"] for m in manifests.values())
    pipe = ESM2Pipeline(lambda seqs: [[0.0] * HIDDEN_SIZE for _ in seqs], "cpu")
    pipe.model, pipe.tokenizer = object(), object()
    # Validation passes, so adapt reaches the model import (which the fixture turns into an assertion).
    with pytest.raises(AssertionError, match="model dependency imported"):
        pipe.adapt(splits["train"], splits["validation"], classes=manifest["classes"])
    # The smallest dataset the contract accepts (12 records, 2 classes) also reaches adaptation.
    small = generate_sample_dataset(seed=13, size=12)
    small_splits = split_dataset(small, seed=42)
    assert {k: len(v) for k, v in small_splits.items()} == {"train": 8, "validation": 2, "test": 2}
    validate_splits(small_splits)
    with pytest.raises(AssertionError, match="model dependency imported"):
        pipe.adapt(small_splits["train"], small_splits["validation"])


def test_split_refusals_name_the_split(forbid_model_imports) -> None:
    splits = split_dataset(_three_class(), seed=42)
    no_charged = [r for r in splits["validation"] if r["label"] != "charged"]
    with pytest.raises(ValueError, match=r"^validation split: classes \['charged'\] have fewer than 1"):
        validate_splits({**splits, "validation": no_charged})
    pipe = ESM2Pipeline(lambda seqs: [[0.0] * HIDDEN_SIZE for _ in seqs], "cpu")
    pipe.model, pipe.tokenizer = object(), object()
    with pytest.raises(ValueError, match=r"^validation split: "):
        pipe.adapt(splits["train"], no_charged)
    with pytest.raises(ValueError, match=r"^train split: "):
        pipe.adapt(splits["train"][:1], splits["validation"])


def test_split_message_names_the_size_a_class_needs() -> None:
    records = generate_sample_dataset(size=12)  # 6 per class: 3 validation + 3 test leave no training record
    with pytest.raises(ValueError, match=r"class 'scattered' has 6 records; .* it needs at least 7 to leave one per split"):
        split_dataset(records, val_fraction=0.45, test_fraction=0.45)


def test_evaluate_accepts_a_small_split_and_refuses_without_a_head(tmp_path: Path) -> None:
    splits = split_dataset(generate_sample_dataset(seed=13, size=12), seed=42)
    pipe = _stub_classifier_pipe(["scattered", "segment"])
    metrics = pipe.evaluate(splits["test"])  # 2 records: used to fail the 12-record dataset minimum
    assert metrics["n"] == 2 and metrics["accuracy"] == 0.5
    pipe.reset_adaptation()
    assert pipe.classes == [] and pipe._classifier is None and pipe.adaptation == {}
    with pytest.raises(RuntimeError, match="requires an adapted head"):
        pipe.evaluate(splits["test"])
    with pytest.raises(RuntimeError, match="requires an adapted head"):
        pipe.save_artifact(tmp_path / "never-written")
    assert not (tmp_path / "never-written").exists()


def _section4_namespace(pipe: Any) -> dict[str, Any]:
    ns = {name: getattr(package, name) for name in package.__all__}
    ns.update({"pipe": pipe, "__name__": "__main__"})
    return ns


def _run_section4(nb: dict, ns: dict, *, use_byod: bool, byod_path: str = "") -> str:
    src = _cell(nb, "USE_BYOD = False  # @param")
    if use_byod:
        src = src.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param")
        src = src.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {byod_path!r}  # @param")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(src, "section4", "exec"), ns)
    return out.getvalue()


def test_section4_byod_path_runs_outside_colab_and_drops_the_old_head(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    pipe = _stub_classifier_pipe(["scattered", "segment"])
    pipe.adaptation = {"trainable_layers": 2}
    path = tmp_path / "my_proteins.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _three_class()) + "\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    ns = _section4_namespace(pipe)
    printed = _run_section4(nb, ns, use_byod=True, byod_path=str(path))
    assert "google.colab" not in sys.modules
    assert ns["data_source"] == "BYOD (my_proteins.jsonl)"
    assert ns["dataset_csv"] == "outputs/esm2_protein_byod_dataset.csv"
    assert (tmp_path / "outputs" / "esm2_protein_byod_dataset.csv").is_file()
    assert not (tmp_path / "outputs" / "esm2_protein_sample_dataset.csv").exists()
    assert [len(ns[k]) for k in ("train_records", "val_records", "test_records")] == [18, 6, 6]
    assert pipe.classes == [] and pipe._classifier is None, "the sample head must be dropped when the dataset changes"
    assert "'validation': {'n': 6" in printed


def test_section4_default_path_writes_the_sample_template(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns = _section4_namespace(_stub_classifier_pipe(["a", "b"]))
    _run_section4(nb, ns, use_byod=False)
    assert ns["dataset_csv"] == "outputs/esm2_protein_sample_dataset.csv"
    assert [len(ns[k]) for k in ("train_records", "val_records", "test_records")] == [56, 20, 20]


# --- ESM-m2: actionable BYOD intake errors ----------------------------------------------------------------------------


def test_cancelled_upload_and_utf16_file_get_actionable_messages(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    replies: list[dict] = [{}]
    google = types.ModuleType("google")
    google.__path__ = []
    colab = types.ModuleType("google.colab")
    colab.__path__ = []
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: replies.pop(0)
    colab.files = files
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)
    ns = _section4_namespace(_stub_classifier_pipe(["a", "b"]))
    with pytest.raises(ValueError, match=r"Upload exactly one \.csv, \.json or \.jsonl file \(got 0"):
        _run_section4(nb, ns, use_byod=True)
    utf16 = tmp_path / "export.csv"
    records = generate_sample_dataset(size=24)
    write_dataset_csv(records, tmp_path / "plain.csv")
    utf16.write_bytes((tmp_path / "plain.csv").read_text(encoding="utf-8").encode("utf-16"))
    with pytest.raises(ValueError, match=r"export\.csv is not UTF-8 text .*save it as UTF-8"):
        load_byod_dataset(utf16)


# --- ESM-M4 / ESM-m1: guided layer and the structured activity ------------------------------------------------------


def test_guided_layer_and_infrastructure_cells(nb: dict) -> None:
    md = _markdown(nb)
    for marker, least in (
        ("**Who this is for.**", 1),
        ("**Input → Model → Output.**", 1),
        ("**How to use this notebook.**", 1),
        ("**Roadmap:**", 1),
        ("**Predict before running:**", 8),
        ("**What to notice:**", 8),
        ("<summary>Check your reasoning</summary>", 9),
        ("## Troubleshooting", 1),
        ("## Glossary", 1),
        ("## Conclusion (your notes)", 1),
        ("> **Infrastructure.**", 3),
    ):
        assert md.count(marker) >= least, marker
    titled = [c for c in _code_cells(nb) if _src(c).startswith("# @title Infrastructure:")]
    assert len(titled) == 7 and all(c["metadata"].get("cellView") == "form" for c in titled)
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"


def test_activity_is_structured_and_rerun_exports_describe_one_model(nb: dict) -> None:
    md = _markdown(nb)
    assert "## 12. Your turn — change one thing: train the head alone" in md
    assert "**Predict → Change one thing → Run → Observe → Explain.**" in md
    assert "select the Section 7 cell and choose **Runtime → Run after**" in md
    assert "accuracy sits near" not in md, "the EPOCHS = 1 hint must not promise a fixed outcome"
    assert "run_history.append(" in _cell(nb, "val_metrics = pipe.evaluate(val_records)")
    assert "assert evaluation_report['adaptation']['trainable_layers'] == artifact_manifest['adaptation']['trainable_layers'] == TRAINABLE_LAYERS" in _cell(nb, "result_payload = {")
    assert "for row in run_history:" in _cell(nb, "columns = ['run', 'data'")


# --- ESM-m3: the release procedure carries the BYOD gate ------------------------------------------------------------


def test_release_procedure_has_a_byod_gate() -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "the **BYOD gate (REL12)**" in text
    assert "record one refused input" in text
