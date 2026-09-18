"""ESM-2 150M (`facebook/esm2_t30_150M_UR50D`) DIMER pipeline: verified snapshot, residue-level
embeddings, and bounded sequence-classification fine-tuning with a portable adapter artifact.

Everything model-related is imported lazily so that snapshot verification and input validation run
(and can refuse) before `torch` or `transformers` are imported (fleet RTM-001).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "facebook/esm2_t30_150M_UR50D"
MODEL_REVISION = "a695f6045e2e32885fa60af20c13cb35398ce30c"
MODEL_LICENSE = "mit"
MODEL_KEY = "esm2-t30-150m-ur50d"
ARTIFACT_FORMAT = "org.valcorza.esm2-protein.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Ceilings. ESM-2 was trained on windows of 1,024 tokens; `<cls>` and `<eos>` take two of them, so
# 1,022 residues is the longest sequence that fits one training-length window. Rotary positions
# do not fail beyond that, they degrade silently, so the pipeline refuses longer sequences.
MAX_RESIDUES = 1_022
MAX_SEQUENCES_PER_CALL = 64
HIDDEN_SIZE = 640  # config.json hidden_size; the embedding width every `embed` row has
# The 20 standard amino acids plus the ambiguity/rare codes ESM-2's vocabulary carries. Anything
# else (lowercase, gaps, whitespace, digits) is a validation error, never a silent `<unk>`.
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
EXTENDED_AMINO_ACIDS = "BUZOX"  # B=Asx, U=selenocysteine, Z=Glx, O=pyrrolysine, X=unknown
ALPHABET = frozenset(STANDARD_AMINO_ACIDS + EXTENDED_AMINO_ACIDS)


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest.hexdigest()} != manifest {entry['sha256']}")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the ESM-2 snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _verify_manifest(root, MODEL_ID, MODEL_REVISION)


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


INPUT_SCHEMA: dict[str, Any] = {
    "input": "1..MAX_SEQUENCES_PER_CALL protein sequences as uppercase one-letter strings",
    "sequences": [1, MAX_SEQUENCES_PER_CALL],
    "residues": [1, MAX_RESIDUES],
    "alphabet": "".join(sorted(ALPHABET)),
    "preprocessing": (
        "each sequence is tokenized one residue per token by the pinned EsmTokenizer, wrapped in "
        "<cls> ... <eos>, and right-padded to the longest sequence in the batch; embeddings are the "
        "mean of the last hidden state over residue tokens only (cls, eos and pad excluded)"
    ),
}


def _check_sequences(sequences: Any, names: Any = None) -> tuple[list[str], list[str]]:
    """Raise TypeError/ValueError naming the first violated ceiling; return (sequences, ids).

    ``embed``, ``classify`` and ``validate_inputs`` all route through this function so their
    acceptance criteria cannot diverge.
    """
    if isinstance(sequences, str | bytes) or not isinstance(sequences, Sequence):
        raise TypeError("sequences must be a list of str (one protein sequence per item)")
    if not 1 <= len(sequences) <= MAX_SEQUENCES_PER_CALL:
        raise ValueError(f"sequences must hold 1..{MAX_SEQUENCES_PER_CALL} items, got {len(sequences)}")
    checked: list[str] = []
    for i, seq in enumerate(sequences):
        if not isinstance(seq, str):
            raise TypeError(f"sequences[{i}] must be str, got {type(seq).__name__}")
        if not seq:
            raise ValueError(f"sequences[{i}] is empty")
        if len(seq) > MAX_RESIDUES:
            raise ValueError(f"sequences[{i}] has {len(seq)} residues; ceiling is {MAX_RESIDUES}")
        bad = sorted({ch for ch in seq if ch not in ALPHABET})
        if bad:
            raise ValueError(
                f"sequences[{i}] contains characters outside the amino-acid alphabet: {bad!r} "
                f"(allowed: uppercase {STANDARD_AMINO_ACIDS} plus {EXTENDED_AMINO_ACIDS}; strip gaps, "
                "whitespace and lowercase first)"
            )
        checked.append(seq)
    if names is None:
        ids = [f"seq-{i}" for i in range(len(checked))]
    else:
        if isinstance(names, str | bytes) or not isinstance(names, Sequence) or len(names) != len(checked):
            raise ValueError("names must be a list with exactly one id per sequence")
        ids = [str(n) for n in names]
        if len(set(ids)) != len(ids):
            raise ValueError("names must be unique")
    return checked, ids


def validate_inputs(sequences: Sequence[str], *, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, verdict).

    Rejection is reported by raising exactly as ``embed``/``classify`` would; a caller that wants
    the finding recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    checked, ids = _check_sequences(sequences, names)
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": sid,
                "residues": len(seq),
                "non_standard_residues": sum(ch in EXTENDED_AMINO_ACIDS for ch in seq),
            }
            for sid, seq in zip(ids, checked, strict=True)
        ],
        "n_sequences": len(checked),
        "max_residues_observed": max(len(seq) for seq in checked),
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def _softmax(logits: Sequence[float]) -> list[float]:
    import math

    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


@dataclass
class ESM2Pipeline:
    """ESM-2 150M pipeline: `embed` (representations) always; `classify` after `adapt` or `from_artifact`.

    `_embedder(sequences)` returns one HIDDEN_SIZE-wide mean-pooled vector per sequence;
    `_classifier(sequences)` returns one logits row per sequence (None until adapted).
    """

    _embedder: Callable[[list[str]], list[list[float]]]
    device: str
    load_warnings: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    _classifier: Callable[[list[str]], list[list[float]]] | None = None
    model: Any = None
    tokenizer: Any = None
    classifier_model: Any = None
    adaptation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ESM2Pipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(f"no snapshot manifest at {root} and allow_download={allow_download}")
        # Stage and verify before importing model libraries (RTM-001).
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        import torch
        from transformers import AutoTokenizer, EsmModel

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tokenizer = AutoTokenizer.from_pretrained(
                str(root), local_files_only=True, trust_remote_code=False
            )
            model = EsmModel.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                add_pooling_layer=False,
            )
        model = model.to(resolved_device).eval()
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]
        pipe = cls(cls._make_embedder(model, tokenizer, resolved_device), resolved_device, messages)
        pipe.model, pipe.tokenizer = model, tokenizer
        return pipe

    # -- backends ---------------------------------------------------------------------------------

    @staticmethod
    def _encode(tokenizer: Any, sequences: list[str], device: str) -> dict[str, Any]:
        batch = tokenizer(sequences, return_tensors="pt", padding=True, add_special_tokens=True)
        return {k: v.to(device) for k, v in batch.items()}

    @staticmethod
    def _residue_mask(batch: dict[str, Any], tokenizer: Any) -> Any:
        """1 for residue tokens, 0 for <cls>, <eos> and padding (the pooling window)."""
        ids = batch["input_ids"]
        mask = batch["attention_mask"].clone()
        for special in (tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
            if special is not None:
                mask = mask * (ids != special)
        return mask

    @classmethod
    def _make_embedder(
        cls, model: Any, tokenizer: Any, device: str
    ) -> Callable[[list[str]], list[list[float]]]:
        import torch

        def embedder(sequences: list[str]) -> list[list[float]]:
            batch = cls._encode(tokenizer, sequences, device)
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state
            mask = cls._residue_mask(batch, tokenizer).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return pooled.float().cpu().tolist()

        return embedder

    @classmethod
    def _make_classifier(
        cls, model: Any, tokenizer: Any, device: str
    ) -> Callable[[list[str]], list[list[float]]]:
        import torch

        def classifier(sequences: list[str]) -> list[list[float]]:
            batch = cls._encode(tokenizer, sequences, device)
            # no_grad, not inference_mode: the ESM rotary layers cache cos/sin tables per sequence
            # length, and a cache built under inference_mode cannot be reused by the next training
            # epoch ("Inference tensors cannot be saved for backward").
            with torch.no_grad():
                logits = model(**batch).logits
            return logits.float().cpu().tolist()

        return classifier

    # -- public stages ----------------------------------------------------------------------------

    def embed(self, sequences: Sequence[str], *, names: Sequence[str] | None = None) -> dict[str, Any]:
        """Mean-pooled last-hidden-state representation per sequence (HIDDEN_SIZE floats each)."""
        checked, ids = _check_sequences(sequences, names)
        vectors = self._embedder(checked)
        if len(vectors) != len(checked) or any(len(v) != HIDDEN_SIZE for v in vectors):
            raise RuntimeError("backend returned embeddings of the wrong shape")
        return {
            "ids": ids,
            "embeddings": [[float(x) for x in v] for v in vectors],
            "dimension": HIDDEN_SIZE,
            "pooling": "mean of the last hidden state over residue tokens (cls/eos/pad excluded)",
            "unit": "one vector per whole sequence; representations, not predictions",
            "n_sequences": len(checked),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def classify(self, sequences: Sequence[str], *, names: Sequence[str] | None = None) -> dict[str, Any]:
        """Class scores and argmax label per sequence; requires a prior `adapt` or `from_artifact`."""
        if self._classifier is None or not self.classes:
            raise RuntimeError(
                "classify requires an adapted head: call adapt(...) or load from_artifact(...) first"
            )
        checked, ids = _check_sequences(sequences, names)
        logits = self._classifier(checked)
        predictions = []
        for sid, seq, row in zip(ids, checked, logits, strict=True):
            if len(row) != len(self.classes):
                raise RuntimeError("backend returned a logits row that does not match the class list")
            scores = _softmax(row)
            best = max(range(len(scores)), key=scores.__getitem__)
            predictions.append(
                {
                    "id": sid,
                    "residues": len(seq),
                    "label": self.classes[best],
                    "score": scores[best],
                    "scores": dict(zip(self.classes, scores, strict=True)),
                }
            )
        return {
            "predictions": predictions,
            "classes": list(self.classes),
            "decision_rule": (
                "argmax over softmax(logits); scores are softmax outputs, not calibrated probabilities"
            ),
            "n_sequences": len(checked),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adaptation": dict(self.adaptation),
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Held-out classification metrics over labelled records (see metrics.classification_metrics)."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        validate_dataset(records, classes=self.classes)
        sequences = [r["sequence"] for r in records]
        ids = [r["id"] for r in records]
        predicted: list[str] = []
        scores: list[list[float]] = []
        for start in range(0, len(sequences), MAX_SEQUENCES_PER_CALL):
            chunk = self.classify(
                sequences[start : start + MAX_SEQUENCES_PER_CALL],
                names=ids[start : start + MAX_SEQUENCES_PER_CALL],
            )
            for p in chunk["predictions"]:
                predicted.append(p["label"])
                scores.append([p["scores"][c] for c in self.classes])
        return classification_metrics([r["label"] for r in records], predicted, scores, self.classes)

    def adapt(
        self,
        train_records: Sequence[Mapping[str, Any]],
        val_records: Sequence[Mapping[str, Any]] | None = None,
        *,
        classes: Sequence[str] | None = None,
        epochs: int = 4,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        trainable_layers: int = 2,
        weight_decay: float = 0.01,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Bounded gradient fine-tuning of a sequence-classification head on top of the verified base.

        Builds `EsmForSequenceClassification` from the pinned base weights (the classifier head is
        newly initialised), freezes every parameter except the head and the last `trainable_layers`
        encoder layers, and runs AdamW for `epochs` passes. Validation records are monitored per
        epoch only; the final epoch's weights are kept (no selection).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("adapt requires a pipeline built by from_pretrained (no loaded base model)")
        from .samples import validate_dataset

        if not 1 <= int(epochs) <= 50:
            raise ValueError("epochs must be in 1..50 (tutorial-scale adaptation)")
        if not 1 <= int(batch_size) <= MAX_SEQUENCES_PER_CALL:
            raise ValueError(f"batch_size must be in 1..{MAX_SEQUENCES_PER_CALL}")
        if not 0 <= int(trainable_layers) <= 30:
            raise ValueError("trainable_layers must be in 0..30")
        train_manifest = validate_dataset(train_records, classes=classes)
        class_list = list(train_manifest["classes"])
        if val_records is not None:
            validate_dataset(val_records, classes=class_list)

        import random

        import torch
        from transformers import EsmForSequenceClassification

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        root = Path(getattr(self.model.config, "_name_or_path", DEFAULT_WEIGHTS_DIR))
        clf = EsmForSequenceClassification.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            num_labels=len(class_list),
        )
        clf = clf.to(self.device)
        for p in clf.parameters():
            p.requires_grad = False
        layers = clf.esm.encoder.layer
        for layer in layers[len(layers) - int(trainable_layers) :] if trainable_layers else []:
            for p in layer.parameters():
                p.requires_grad = True
        for p in clf.classifier.parameters():
            p.requires_grad = True
        if trainable_layers:
            for p in clf.esm.encoder.emb_layer_norm_after.parameters():
                p.requires_grad = True
        trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in clf.parameters())
        optimizer = torch.optim.AdamW(
            [p for p in clf.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay
        )
        label_index = {c: i for i, c in enumerate(class_list)}
        examples = [(r["sequence"], label_index[r["label"]]) for r in train_records]
        self.classes = class_list
        self.classifier_model = clf
        self._classifier = self._make_classifier(clf, self.tokenizer, self.device)

        history: list[dict[str, Any]] = []
        for epoch in range(1, int(epochs) + 1):
            clf.train()
            order = list(range(len(examples)))
            random.shuffle(order)
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(order), int(batch_size)):
                batch_examples = [examples[i] for i in order[start : start + int(batch_size)]]
                batch = self._encode(self.tokenizer, [s for s, _ in batch_examples], self.device)
                labels = torch.tensor([y for _, y in batch_examples], device=self.device)
                optimizer.zero_grad()
                out = clf(**batch, labels=labels)
                out.loss.backward()
                optimizer.step()
                total_loss += float(out.loss.item())
                n_batches += 1
            clf.eval()
            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(total_loss / max(1, n_batches), 6),
                "n_batches": n_batches,
            }
            if val_records:
                val = self.evaluate(val_records)
                entry["val_accuracy"] = val["accuracy"]
                entry["val_macro_f1"] = val["macro_f1"]
            history.append(entry)
        clf.eval()
        self.adaptation = {
            "method": "gradient fine-tuning (AdamW) of the classification head"
            + (f" and the last {int(trainable_layers)} encoder layer(s)" if trainable_layers else ""),
            "classes": class_list,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
            "trainable_layers": int(trainable_layers),
            "seed": int(seed),
            "precision": "float32",
            "trainable_parameters": int(n_trainable),
            "total_parameters": int(n_total),
            "trainable_parameter_names": trainable,
            "train_records": len(train_records),
            "val_records": len(val_records) if val_records else 0,
            "selection": "final epoch kept; validation metrics are monitoring only",
            "history": history,
        }
        return dict(self.adaptation)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Export the trainable tensors as safetensors plus a JSON manifest binding them to the base."""
        if self.classifier_model is None or not self.classes:
            raise RuntimeError("save_artifact requires an adapted head (call adapt first)")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adaptation.get("trainable_parameter_names", []))
        tensors = {
            k: v.detach().cpu().contiguous()
            for k, v in self.classifier_model.state_dict().items()
            if k in names
        }
        if not tensors:
            raise RuntimeError("no trainable tensors recorded; nothing to export")
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path))
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "license": MODEL_LICENSE},
            "classes": list(self.classes),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": digest}
            ],
            "tensors": sorted(tensors),
            "adaptation": {k: v for k, v in self.adaptation.items() if k != "trainable_parameter_names"},
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Rebuild the classification head from an exported artifact (manifest verified before loading)."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("load_artifact requires a pipeline built by from_pretrained")
        art = Path(artifact_dir)
        manifest_path = art / ARTIFACT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("model_id"), base.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError(f"artifact was trained on {base}, this package pins {MODEL_ID}@{MODEL_REVISION}")
        classes = [str(c) for c in manifest.get("classes", [])]
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("artifact manifest must list at least two unique classes")
        for entry in manifest["files"]:
            fp = art / entry["path"]
            if not fp.is_file():
                raise FileNotFoundError(f"artifact file missing: {fp}")
            if fp.stat().st_size != entry["bytes"]:
                raise ValueError(f"{entry['path']}: size {fp.stat().st_size} != manifest {entry['bytes']}")
            if hashlib.sha256(fp.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"{entry['path']}: sha256 mismatch against the artifact manifest")
        from safetensors.torch import load_file
        from transformers import EsmForSequenceClassification

        root = Path(getattr(self.model.config, "_name_or_path", DEFAULT_WEIGHTS_DIR))
        clf = EsmForSequenceClassification.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            num_labels=len(classes),
        )
        tensors = load_file(str(art / ARTIFACT_WEIGHTS_NAME))
        expected = set(manifest.get("tensors", []))
        if set(tensors) != expected:
            raise ValueError("artifact tensors do not match the names listed in its manifest")
        missing, unexpected = clf.load_state_dict(tensors, strict=False)
        if unexpected:
            raise ValueError(
                f"artifact carries tensors the base architecture does not have: {sorted(unexpected)[:5]}"
            )
        clf = clf.to(self.device).eval()
        self.classes = classes
        self.classifier_model = clf
        self._classifier = self._make_classifier(clf, self.tokenizer, self.device)
        self.adaptation = {**manifest.get("adaptation", {}), "loaded_from_artifact": str(art)}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ESM2Pipeline:
        """Verified base snapshot + exported adapter, ready for `classify`."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
