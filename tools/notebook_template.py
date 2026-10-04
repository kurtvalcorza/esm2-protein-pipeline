"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E protein sequence-classification adaptation: ESM-2 150M is
verified from its pinned snapshot, a synthetic dataset defined by a residue-order rule is validated
and split, four trivial baselines are measured, a bounded AdamW fine-tuning of the classification
head plus the last encoder layers runs, and the adapter is exported as safetensors and reloaded.

Review fixes (Notebook Review Framework v1, review PR #7, ESM-M1..M4 / ESM-m1..m3): the runtime is the fleet's uv
isolated environment (no in-kernel install, no restart); the baselines add a 20-residue composition
nearest-centroid and the order rule `longest_hydrophobic_run` that defines the classes, and the prose no longer
claims equal composition or credits pretraining; BYOD splits are validated with the minimums `adapt` and
`evaluate` apply, a dataset change drops any earlier head, `BYOD_PATH` bypasses the upload dialog, an empty upload
and a non-UTF-8 file get actionable messages, and BYOD data is written under its own name; the optional experiment
is a structured activity whose rerun leaves every export describing one model; and the guided layer (who it is
for, how to use, roadmap, predictions, worked answers, troubleshooting, glossary, conclusion) is added.

Code cells are written with single braces and escaped by ``_py`` for the generator's ``str.format`` pass; ``@STEM@``
becomes the output stem. Markdown cells are formatted too, so they contain no literal braces.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "esm2-protein-pipeline"


def _py(code: str) -> str:
    """Escape a code cell for the generator's ``str.format`` pass; ``@STEM@`` stands for ``{stem}``."""
    return code.replace("{", "{{").replace("}", "}}").replace("@STEM@", "{stem}")


BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/esm2_protein_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fesm2__t30__150M__UR50D-ffcc4d?style=flat",
        "https://huggingface.co/facebook/esm2_t30_150M_UR50D",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-facebookresearch%2Fesm-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/facebookresearch/esm",
    ),
    (
        "bioRxiv",
        "https://img.shields.io/badge/bioRxiv-2022.07.20.500902-b31b1b.svg",
        "https://doi.org/10.1101/2022.07.20.500902",
    ),
]

_DATA_CODE = _py(
    """import json
import os
from pathlib import Path

USE_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ''  # @param {type:"string"}
VAL_FRACTION = 0.2  # @param {type:"number"}
TEST_FRACTION = 0.2  # @param {type:"number"}
SEED = 42  # @param {type:"integer"}

os.makedirs('outputs', exist_ok=True)
# A new dataset needs a new head: drop any head an earlier run trained, so no later cell can evaluate,
# classify or export it under this dataset's name.
pipe.reset_adaptation()
if USE_BYOD:
    if BYOD_PATH:
        byod_path = Path(BYOD_PATH)
        file_name = byod_path.name
    else:
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError(f'Upload exactly one .csv, .json or .jsonl file (got {len(uploaded)}; an empty or cancelled upload gives 0). Outside Colab, set BYOD_PATH to the file instead.')
        file_name, payload = next(iter(uploaded.items()))
        byod_path = Path('work') / file_name
        byod_path.parent.mkdir(parents=True, exist_ok=True)
        byod_path.write_bytes(payload)
    records = load_byod_dataset(byod_path)
    data_source = 'BYOD (' + file_name + ')'
    dataset_csv = 'outputs/@STEM@_byod_dataset.csv'
else:
    records = generate_sample_dataset()
    data_source = f'synthetic hydrophobic-segment dataset (seed {SAMPLE_SEED}, {SAMPLE_SIZE} records)'
    dataset_csv = 'outputs/@STEM@_sample_dataset.csv'

dataset_manifest = validate_dataset(records)
CLASSES = dataset_manifest['classes']
splits = split_dataset(records, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED)
# Every split must hold every class at least once: the minimum adapt and evaluate apply. A refusal names the split.
split_manifests = validate_splits(splits, CLASSES)
train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']
write_dataset_csv(records, dataset_csv)

print({'data_source': data_source, 'n_records': dataset_manifest['n_records'], 'classes': CLASSES, 'class_counts': dataset_manifest['class_counts']})
print({'residues': dataset_manifest['residues'], 'ceilings': dataset_manifest['ceilings'], 'digest': dataset_manifest['digest'][:16] + '...'})
print({name: {'n': m['n_records'], 'class_counts': m['class_counts']} for name, m in split_manifests.items()})
print({'written': dataset_csv})
example = train_records[0]
print({'example_id': example['id'], 'label': example['label'], 'residues': len(example['sequence']), 'longest_hydrophobic_run': longest_hydrophobic_run(example['sequence']), 'hydrophobic_fraction': round(hydrophobic_fraction(example['sequence']), 3)})
print(example['sequence'])"""
)

_EMBED_CODE = _py(
    """import csv
import math

embed_records = val_records[:8]
embedding_result = pipe.embed([r['sequence'] for r in embed_records], names=[r['id'] for r in embed_records])
vectors = embedding_result['embeddings']
print({'n_sequences': embedding_result['n_sequences'], 'dimension': embedding_result['dimension'], 'pooling': embedding_result['pooling']})

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

within, between = [], []
for i in range(len(embed_records)):
    for j in range(i + 1, len(embed_records)):
        sim = cosine(vectors[i], vectors[j])
        (within if embed_records[i]['label'] == embed_records[j]['label'] else between).append(sim)
print({'mean_cosine_within_class': round(sum(within) / len(within), 4) if within else None, 'mean_cosine_between_classes': round(sum(between) / len(between), 4) if between else None, 'note': 'inspection only; embeddings are unlabelled representations'})

with open('outputs/@STEM@_embeddings.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'label'] + [f'dim_{k}' for k in range(embedding_result['dimension'])])
    for r, vec in zip(embed_records, vectors):
        writer.writerow([r['id'], r['label']] + [f'{x:.6f}' for x in vec])
print('wrote outputs/@STEM@_embeddings.csv')"""
)

_BASELINE_CODE = _py(
    """baseline_majority = majority_baseline(train_records, test_records, CLASSES)
baselines = {'majority': baseline_majority, 'residue_composition': residue_composition_baseline(train_records, test_records, CLASSES)}
if len(CLASSES) == 2:
    baseline_composition = composition_baseline(train_records, test_records, CLASSES)
    baselines['hydrophobic_fraction'] = baseline_composition
    baselines['longest_hydrophobic_run'] = longest_run_baseline(train_records, test_records, CLASSES)
else:
    baseline_composition = None
    print('the two threshold baselines are defined for binary tasks only; skipped for', len(CLASSES), 'classes')
for name, b in baselines.items():
    print({'baseline': name, 'kind': 'order-aware' if name == 'longest_hydrophobic_run' else 'order-blind', 'rule': b.get('rule', b.get('predicted_label')), 'train_accuracy': b.get('train_accuracy'), 'test_accuracy': b['accuracy'], 'test_macro_f1': b['macro_f1'], 'test_auroc': b['auroc']})"""
)

_ADAPT_CODE = _py(
    """import time

EPOCHS = 4  # @param {type:"integer"}
LEARNING_RATE = 1e-4  # @param {type:"number"}
BATCH_SIZE = 8  # @param {type:"integer"}
TRAINABLE_LAYERS = 2  # @param {type:"integer"}

started = time.perf_counter()
adapt_result = pipe.adapt(
    train_records,
    val_records,
    classes=CLASSES,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    trainable_layers=TRAINABLE_LAYERS,
    seed=SEED,
)
adapt_seconds = round(time.perf_counter() - started, 1)
print({'method': adapt_result['method'], 'trainable_parameters': adapt_result['trainable_parameters'], 'total_parameters': adapt_result['total_parameters'], 'precision': adapt_result['precision'], 'device': pipe.device, 'seconds': adapt_seconds})
for step in adapt_result['history']:
    print({k: step[k] for k in step})"""
)

_EVAL_CODE = _py(
    """val_metrics = pipe.evaluate(val_records)
test_metrics = pipe.evaluate(test_records)
print({'split': 'validation', **{k: val_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}})
print({'split': 'test', **{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}})
for cls_name, row in test_metrics['per_class'].items():
    print({'class': cls_name, **row})

comparison = {name: {'accuracy': b['accuracy'], 'macro_f1': b['macro_f1'], 'auroc': b['auroc']} for name, b in baselines.items()}
comparison['fine_tuned'] = {k: test_metrics[k] for k in ('accuracy', 'macro_f1', 'auroc')}
for name, row in comparison.items():
    print({'test_split': name, **row})

evaluation_report = {
    'task': 'protein sequence classification (bounded fine-tuning of ESM-2 150M)',
    'evidence': 'tutorial sample-sanity metrics on one stratified holdout; not a benchmark',
    'estimation': 'single train/validation/test split, seed ' + str(SEED) + ', no dispersion estimate',
    'data_source': data_source,
    'dataset_digest': dataset_manifest['digest'],
    'classes': CLASSES,
    'splits': {'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)},
    'baselines': baselines,
    'validation_metrics': val_metrics,
    'test_metrics': test_metrics,
    'comparison': comparison,
    'delta_vs_majority': {k: round(test_metrics[k] - baseline_majority[k], 4) for k in ('accuracy', 'macro_f1')},
    'delta_vs_baselines': {name: round(test_metrics['accuracy'] - b['accuracy'], 4) for name, b in baselines.items()},
    'adaptation': {k: v for k, v in adapt_result.items() if k != 'trainable_parameter_names'},
    'adaptation_seconds': adapt_seconds,
}
with open('outputs/@STEM@_evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_report, f, indent=2)
print({'delta_vs_baselines': evaluation_report['delta_vs_baselines'], 'report': 'outputs/@STEM@_evaluation_report.json'})

# One row per Section 7 run in this session, so a changed setting is read next to the default run (Section 12).
run_history = globals().get('run_history', [])
run_history.append({'run': len(run_history) + 1, 'data': 'BYOD' if USE_BYOD else 'sample', 'trainable_layers': TRAINABLE_LAYERS, 'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'trainable_parameters': adapt_result['trainable_parameters'], 'final_val_accuracy': adapt_result['history'][-1].get('val_accuracy'), 'test_accuracy': test_metrics['accuracy'], 'test_macro_f1': test_metrics['macro_f1'], 'test_auroc': test_metrics['auroc']})"""
)

_INFER_CODE = _py(
    """if USE_BYOD:
    new_records = test_records[:6]
    new_source = 'first six BYOD test-split records'
else:
    new_records = generate_sample_dataset(seed=7, size=6)
    new_source = 'freshly generated sequences (seed 7)'
input_manifest = validate_inputs([r['sequence'] for r in new_records], names=[r['id'] for r in new_records])
print({'new_source': new_source, 'verdict': input_manifest['verdict'], 'n_sequences': input_manifest['n_sequences'], 'max_residues_observed': input_manifest['max_residues_observed']})
inference_result = pipe.classify([r['sequence'] for r in new_records], names=[r['id'] for r in new_records])
predictions = inference_result['predictions']
print({'decision_rule': inference_result['decision_rule']})
n_match = 0
for p, r in zip(predictions, new_records):
    n_match += p['label'] == r['label']
    print({'id': p['id'], 'predicted': p['label'], 'score': round(p['score'], 4), 'true_label': r['label']})
print({'matches': n_match, 'of': len(new_records), 'note': 'sanity check on generated labels, not an evaluation'})

with open('outputs/@STEM@_predictions.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'residues', 'predicted_label', 'score'] + [f'score_{c}' for c in CLASSES])
    for p in predictions:
        writer.writerow([p['id'], p['residues'], p['label'], f"{p['score']:.6f}"] + [f"{p['scores'][c]:.6f}" for c in CLASSES])
print('wrote outputs/@STEM@_predictions.csv')"""
)

_EXPORT_CODE = _py(
    """artifact_dir = Path('outputs/@STEM@_adapter')
pipe.save_artifact(artifact_dir, metadata={'data_source': data_source, 'dataset_digest': dataset_manifest['digest'], 'test_metrics': {k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}})
with open(artifact_dir / ARTIFACT_MANIFEST_NAME, encoding='utf-8') as f:
    artifact_manifest = json.load(f)
print({'format': artifact_manifest['format'], 'base_model': artifact_manifest['base_model'], 'classes': artifact_manifest['classes'], 'trainable_layers': artifact_manifest['adaptation']['trainable_layers'], 'n_tensors': len(artifact_manifest['tensors']), 'files': artifact_manifest['files']})

reloaded_pipe = ESM2Pipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR)
reloaded_result = reloaded_pipe.classify([r['sequence'] for r in new_records], names=[r['id'] for r in new_records])
max_score_diff = 0.0
for before, after in zip(predictions, reloaded_result['predictions']):
    assert before['id'] == after['id'] and before['label'] == after['label'], f'reload parity failure on {before["id"]}'
    max_score_diff = max(max_score_diff, abs(before['score'] - after['score']))
assert max_score_diff < 1e-5, f'reload score drift {max_score_diff}'
print({'reload_parity': 'PASS', 'labels_equal': True, 'max_abs_score_diff': max_score_diff, 'loaded_from': reloaded_pipe.adaptation.get('loaded_from_artifact')})
del reloaded_pipe"""
)

_RESULT_CODE = _py(
    """import platform

result_payload = {
    'task': 'protein sequence classification adaptation (ESM-2 150M)',
    'pipeline_class': 'ESM2Pipeline',
    'model_id': MODEL_ID,
    'model_revision': MODEL_REVISION,
    'model_license': MODEL_LICENSE,
    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],
    'notebook_source': NOTEBOOK_SOURCE,
    'data_source': data_source,
    'dataset_manifest': dataset_manifest,
    'embedding_summary': {'n_sequences': embedding_result['n_sequences'], 'dimension': embedding_result['dimension'], 'pooling': embedding_result['pooling']},
    'evaluation_report': evaluation_report,
    'inference': {'new_source': new_source, 'decision_rule': inference_result['decision_rule'], 'predictions': predictions},
    'artifact_format': ARTIFACT_FORMAT,
    'artifact_format_version': ARTIFACT_FORMAT_VERSION,
    'artifact_manifest': artifact_manifest,
    'reload_parity': {'labels_equal': True, 'max_abs_score_diff': max_score_diff},
    'run_history': run_history,
    'runtime': {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'safetensors': safetensors.__version__,
        'device': pipe.device,
        'precision': 'float32',
    },
}
with open('outputs/@STEM@_result.json', 'w', encoding='utf-8') as f:
    json.dump(result_payload, f, indent=2)
# Every export of this run describes one model: the report, the artifact and this file.
assert evaluation_report['adaptation']['trainable_layers'] == artifact_manifest['adaptation']['trainable_layers'] == TRAINABLE_LAYERS

print('outputs/:')
for path in sorted(Path('outputs').rglob('*')):
    if path.is_file():
        print(f'  - {path.as_posix()} ({path.stat().st_size / 1024:.1f} KB)')"""
)

_ACTIVITY_CODE = _py(
    """# Section 8 adds one row per Section 7 run in this session; this cell only prints them.
columns = ['run', 'data', 'trainable_layers', 'epochs', 'learning_rate', 'trainable_parameters', 'final_val_accuracy', 'test_accuracy', 'test_macro_f1', 'test_auroc']
print(' | '.join(columns))
for row in run_history:
    print(' | '.join(str(row[column]) for column in columns))
if len(run_history) == 1:
    print('One run so far. Set TRAINABLE_LAYERS = 0 in Section 7, select that cell and choose Runtime > Run after; this table then shows both runs.')"""
)

TEMPLATE = {
    "package": "esm2_protein_pipeline",
    "repo_name": REPO,
    "stem": "esm2_protein",
    "notebook_name": "esm2_protein_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "infrastructure_labels": True,
    "collapse_model_cell": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline):
    # managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`. The pins equal ast-audio-classification-pipeline's,
    # so the lock is that repository's (16eee39) with the requesting project renamed.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (the "
        "kernel's own packages are left alone, so no restart is needed), stages and digest-verifies the pinned ESM-2 150M "
        "snapshot (6 files, ~595 MB), generates the deterministic 96-sequence tutorial dataset in code (no download), "
        "validates it against the sequence-classification contract, splits it into stratified train/validation/test sets, "
        "computes mean-pooled sequence embeddings, measures four trivial baselines on the test split (majority class, "
        "hydrophobic fraction, 20-residue composition and the longest-hydrophobic-run rule), runs a bounded AdamW "
        "fine-tuning of the classification head and the last two encoder layers, evaluates accuracy, macro-F1 and AUROC "
        "on the held-out test split beside those baselines, classifies six freshly generated sequences, exports the "
        "adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify prediction "
        "parity. The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog "
        "and no configuration edit (NOTEBOOK_SPEC 2.2 §5). Measured times come from the previous notebook version, whose "
        "model stages are the same: a Kaggle Tesla T4 run took 243.7 s of cell time (2026-09-18, including a pinned "
        "in-kernel install this version no longer does) and a local Windows CPU check took 65 s with the install skipped "
        "and the files pre-staged (2026-10-03 review). Building the isolated environment (PyTorch with its CUDA libraries) "
        "and downloading the checkpoint come on top and usually take a few minutes (an estimate; no run of this version is "
        "recorded yet)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4, select that cell and choose "
        "**Runtime → Run after** to supply your own labelled protein dataset as a CSV (`id,sequence,label` header), JSON "
        "array or JSONL file — through the upload dialog on Colab, or as a path in `BYOD_PATH` on any runtime. It passes "
        "through the same validation, stratified split, baselines, adaptation, held-out evaluation, inference, artifact "
        "export and reload-parity cells as the synthetic sample; Section 4 first drops the head trained on the sample, so "
        "nothing trained on the sample can be evaluated or exported under your file's name. The expected schema, the "
        "alphabet, the minimum size and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files "
        "stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "ESM2Pipeline",
    "weights_key": "esm2-t30-150m-ur50d",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers", "safetensors"],
    "title": "ESM-2 150M — DIMER E2E protein sequence-classification adaptation tutorial (standalone)",
    "badges": BADGES,
    "capability": "protein sequence embeddings and bounded sequence-classification fine-tuning on labelled protein sequences",
    "intro": (
        "ESM-2 is a protein language model: a 30-layer transformer encoder with rotary position embeddings, pretrained by "
        "masked-token prediction on UniRef50 protein sequences (Lin et al., 2022). The pinned checkpoint `facebook/esm2_t30_150M_UR50D` "
        "reads one amino-acid residue per token and produces a 640-dimensional hidden state per residue. This tutorial uses it in "
        "two ways: as a **representation model** (mean-pooled residue states give one embedding per sequence) and as the **base of a "
        "sequence classifier** (`EsmForSequenceClassification` adds a newly initialised head on the `<cls>` position, and a bounded "
        "gradient fine-tuning adapts that head together with the last encoder layers).\n\n"
        "The tutorial dataset is synthetic and its classes are defined by a **residue-order rule**. Every `scattered` sequence has "
        "the length and the number of strongly hydrophobic residues of one `segment` sequence, but in `segment` those residues form "
        "one contiguous 18–22-residue stretch and in `scattered` no hydrophobic run is longer than 5. Only that hydrophobic count is "
        "matched: the other residues are drawn differently in the two classes, so their full composition is not equal. Section 6 "
        "therefore measures four baselines, two that ignore order and two that do not — and the one-line rule "
        "`longest_hydrophobic_run >= 18` already separates the classes perfectly. A fine-tuned model can at best equal it here. What "
        "the run shows is that the adaptation contract (verified base, bounded fine-tuning, held-out evaluation, export, reload "
        "parity) learns a residue-order rule from 56 training sequences; it does **not** show what ESM-2's pretraining contributes, "
        "which would need the same fine-tuning from a randomly initialised encoder.\n\n"
        "**Who this is for.** A learner who knows basic Python, has met the idea of a train/validation/test split and of a "
        "classifier, and wants to see how a pretrained protein language model is fine-tuned and measured honestly: baselines "
        "first, a bounded fine-tune, one look at the held-out split, an exported adapter that reloads. No prior experience with "
        "ESM-2, transformers or fine-tuning is assumed, and no biology beyond the one-letter amino-acid codes; each term is "
        "explained where it is first used and again in the **Glossary** at the end. No GPU is required (CPU works; a GPU is used "
        "automatically).\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Embedding (Section 5) | Classification (Sections 6–10) |\n"
        "|---|---|---|\n"
        "| Input | protein sequences, uppercase one-letter codes, up to 1,022 residues | labelled records `id`, `sequence`, `label`, split into training, validation and test |\n"
        "| Model | ESM-2 150M encoder, residue states mean-pooled; no head | the same encoder + a new classification head, the head and the last `TRAINABLE_LAYERS` encoder layers trained |\n"
        "| Output | one 640-d vector per sequence; no label, no score | held-out accuracy, macro-F1 and AUROC beside four baselines, argmax labels with per-class scores, and a safetensors adapter that reloads with identical predictions |\n\n"
        "**How to use this notebook.** Choose a runtime (Colab, Kaggle or Linux Jupyter; a GPU is faster, CPU works), then "
        "**Runtime → Run all**. Sections 1–3 are **infrastructure** — the isolated environment, the carried code and the model "
        "verification — and can be run without study; their code is collapsed. The learning path starts in Section 4. Form "
        "fields (`# @param`) are the only values meant to be edited; the defaults reproduce the default path. Each learner "
        "section asks you to **predict** before it runs; the next section opens with **What to notice** and a collapsible "
        "**Check your reasoning** block with a worked answer. Each optional experiment names the field to change and the cell to "
        "re-run from (**Runtime → Run after**). Section 12 is a **change-one-thing activity**. **Troubleshooting**, a **Glossary** "
        "and a **Conclusion** template are at the end. Your notes are optional.\n\n"
        "**Roadmap:** *core concepts* — 4 the dataset and its split → 5 embeddings as representations → 6 four baselines, "
        "order-blind and order-aware; *evaluation practice* — 7 bounded fine-tuning → 8 the held-out comparison; "
        "*engineering* — 9 inference on new sequences → 10 export and reload → 11 provenance → 12 **change one thing: train the "
        "head alone** → conclude."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) say what a mean-pooled ESM-2 embedding is and why a cosine between two embeddings "
        "is an inspection, not an evaluation (Section 5); (2) read a fine-tuned model's test accuracy beside an order-blind and an "
        "order-aware baseline and say what each one rules out (Section 6); (3) state which parameters a bounded fine-tuning "
        "trains and why validation is monitoring only here (Section 7); (4) judge what a perfect score on 20 test sequences does "
        "and does not show (Section 8); (5) check that an exported adapter reproduces the evaluated model (Section 10); and (6) "
        "predict, measure and explain what changes when the head is trained alone (Section 12)."
    ),
    "exclusions": (
        "residue-level (token) classification, contact or structure prediction, masked-token scoring, full-parameter "
        "fine-tuning of all 30 layers, a pretrained-versus-randomly-initialised comparison, or arbitrary unvalidated dataset "
        "formats. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python; amino-acid one-letter codes; what a train/validation/test split protects against. The notebook explains embeddings, mean pooling, encoder layers, the classification head, fine-tuning, accuracy, macro-F1, AUROC, the baselines, the adapter and reload parity where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab, Kaggle or Linux Jupyter. Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter, and a Windows or macOS kernel is not supported (Section 1 stops with that message). The default path runs on CPU in about a minute of model time on a workstation (65 s for all learner cells in a local CPU check with the files pre-staged); CUDA is used automatically when present and shortens adaptation. The locked install (PyTorch 2.14.0 with its CUDA libraries) and the ~595 MB checkpoint are the large downloads.",
        "- **Data contract:** records are `{id, sequence, label}`; sequences are uppercase one-letter strings over `ACDEFGHIKLMNPQRSTVWY` plus the ambiguity codes `BUZOX`, at most 1,022 residues (`MAX_RESIDUES`); ids of 1..64 characters, unique; unique sequences; labels of 1..64 characters; 2..20 classes; at most 5,000 records. **Minimum size:** at least 12 records and 3 per class. That is enough: the stratified split puts every class at least once into each of training, validation and test (with the default 20 % / 20 % fractions), which is all `adapt` and `evaluate` require of a split, and a refusal names the split and the number of records a class needs.",
        "- **BYOD file (Section 4):** CSV with an `id,sequence,label` header, a JSON array of objects, or JSONL, saved as UTF-8 (a spreadsheet's 'CSV UTF-8'). On Colab the upload dialog opens; on any runtime set `BYOD_PATH` to the file instead. Your records are written to `outputs/esm2_protein_byod_dataset.csv`; the sample is written to `outputs/esm2_protein_sample_dataset.csv`, which is also a template of the expected shape. Validation is structural, not semantic: nothing checks that a label is right for its sequence, and random splitting of related proteins leaks (see Interpretation and limits).",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — unpublished, proprietary, patient-derived or export-controlled sequences are exactly that. The default path uploads nothing.",
        "- **Expected log lines:** loading `EsmForSequenceClassification` prints that the classifier head weights are newly initialised — that is the head this tutorial trains, not a defect.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample dataset, validation and split\n\n"
                "The default dataset is generated in code with a fixed seed (`SAMPLE_SEED`): 48 `segment` and 48 `scattered` sequences of 60–120 residues, "
                "paired so that each `scattered` record has exactly the length and hydrophobic residue count of one `segment` record. `validate_dataset` "
                "checks the schema, the alphabet, the residue ceiling, duplicate ids/sequences and class coverage before any model runs, and returns a "
                "manifest with the class counts, the ceilings and a SHA-256 digest of the rows. `split_dataset` then shuffles **within each class** with "
                "`SEED` and cuts 20 % validation / 20 % test, so all three splits keep the class balance, and `validate_splits` checks each split holds "
                "every class (the minimum the fine-tuning and evaluation cells apply; a refusal names the split). Random splitting assumes the records "
                "are independent, which holds for generated data and must be checked for real proteins (homologous sequences across splits leak). The "
                "cell first calls `pipe.reset_adaptation()`: a new dataset needs a new head, so any head an earlier run trained is dropped and no later "
                "cell can evaluate or export it under this dataset's name.\n\n"
                "With `USE_BYOD = True`, your file (upload dialog, or `BYOD_PATH`) goes through `load_byod_dataset` instead; see the Prerequisites for "
                "the format and the minimum size.\n\n"
                "**Predict before running:** the test split will hold 20 sequences, 10 per class. By how many percentage points does one sequence "
                "moved from wrong to right change the test accuracy?"
            ),
            "code": _DATA_CODE,
        },
        {
            "md": (
                "**What to notice:** 96 records, classes `['scattered', 'segment']`, splits 56 / 20 / 20 with both classes in each, and the written "
                "`outputs/esm2_protein_sample_dataset.csv`. The example shows a `longest_hydrophobic_run` of 18–22 for a `segment` record or at most 5 "
                "for a `scattered` one.\n\n"
                "<details><summary>Check your reasoning</summary>5 points: one sequence is 1/20 of the test split. Keep that in mind in Section 8 — "
                "a difference of one or two sequences between two models is within what one seeded split of 20 can show.</details>\n\n"
                "## 5. Sequence embeddings (representation, not prediction)\n\n"
                "`pipe.embed` runs the verified base encoder and returns one 640-dimensional vector per sequence: the mean of the last hidden state over "
                "residue tokens only (`<cls>`, `<eos>` and padding are excluded) — **mean pooling**. Embeddings are representations — they carry no label and no metric of "
                "their own; a downstream labelled task is what gives them meaning (EVAL9). The cell embeds eight validation sequences, writes them with "
                "their ids to `outputs/{stem}_embeddings.csv` (OUT4), and prints the mean cosine similarity within and between classes as an inspection, "
                "not an evaluation: the pretrained model has never seen this synthetic rule, so do not expect a clean separation before adaptation.\n\n"
                "**Predict before running:** before any training, will two sequences of the same class have clearly more similar embeddings than two "
                "sequences of different classes?"
            ),
            "code": _EMBED_CODE,
        },
        {
            "md": (
                "**What to notice:** 8 sequences, 640 dimensions, the pooling description, and two mean cosines.\n\n"
                "<details><summary>Check your reasoning</summary>Barely. In the local CPU check of this notebook's code (2026-10-03 review, same pins) the "
                "within-class mean cosine was 0.961 and the between-class mean 0.943: every sequence looks protein-like to the pretrained encoder, and "
                "averaging over all residues dilutes where the hydrophobic residues sit. A small gap on eight sequences proves nothing about "
                "separability; that is measured with labels, after training, in Section 8.</details>\n\n"
                "## 6. Baselines on the test split\n\n"
                "Four trivial predictors set the reference points before any training (EVAL10/EVAL11), each fitted on the training split only (SPL8) "
                "and scored on the test split with the same `classification_metrics` fields as the model:\n\n"
                "- `majority_baseline` predicts the most frequent training class for every record — 0.5 on a balanced split.\n"
                "- `composition_baseline` fits one threshold on the **hydrophobic fraction**, the one composition feature the generator equalises. "
                "Order-blind.\n"
                "- `residue_composition_baseline` assigns each sequence to the class whose mean **20-residue composition** (the fraction of each amino "
                "acid) is nearest. Order-blind: shuffling a sequence does not change its prediction.\n"
                "- `longest_run_baseline` fits one threshold on `longest_hydrophobic_run`. **Order-aware**, and it is the rule the generator uses to "
                "define the classes.\n\n"
                "The two threshold baselines are binary only and are skipped for a BYOD dataset with more classes.\n\n"
                "**Predict before running:** every `scattered` sequence has the same hydrophobic count as a `segment` sequence. Which of the four "
                "baselines will sit near 0.5 on the test split, and which will not?"
            ),
            "code": _BASELINE_CODE,
        },
        {
            "md": (
                "**What to notice:** the four test accuracies, and which baselines are order-blind.\n\n"
                "<details><summary>Check your reasoning</summary>Only two sit near chance. In the local CPU check: majority 0.50, hydrophobic "
                "fraction 0.45 (AUROC 0.47) — the equalised feature carries nothing — but the 20-residue composition centroid reached **0.75**, "
                "because the two classes draw their other residues from different distributions, and the order rule scored **1.0** (train 1.0; "
                "`segment` runs are 18–22, `scattered` runs at most 5). So the classes are not equal in composition, and a one-line rule already "
                "solves the task. Whatever the fine-tuned model scores in Section 8, it cannot beat 1.0, and matching the rule cannot tell you "
                "whether pretraining helped.</details>\n\n"
                "## 7. Bounded fine-tuning\n\n"
                "`pipe.adapt` drops any earlier head, builds `EsmForSequenceClassification` from the verified base weights (the head is newly "
                "initialised — the log line says so), **freezes** every parameter except the classification head, the final layer norm and the last "
                "`TRAINABLE_LAYERS` encoder layers (each **encoder layer** is one transformer block of the 30), and runs AdamW with the hyperparameters "
                "below (FT4/FT6): these are tutorial values chosen for a one-minute CPU run, not production settings. Validation metrics are computed "
                "after each **epoch** (one pass over the training split) for **monitoring only**; the final epoch's weights are kept, so no selection "
                "happens on the validation split (EVAL14). Training loss going down is optimisation evidence, not task-quality evidence (FT7) — "
                "Section 8 is where quality is measured. Every run starts again from the pretrained base, so a rerun with other settings never "
                "builds on an earlier one.\n\n"
                "**Predict before running:** about 10.3 M of 148 M parameters are trained with two layers. Will validation accuracy reach 1.0 within "
                "four epochs, and could the fine-tuned model beat the order rule of Section 6 on the test split?"
            ),
            "code": _ADAPT_CODE,
        },
        {
            "md": (
                "**What to notice:** the trainable/total parameter counts, the train loss per epoch and the validation accuracy per epoch.\n\n"
                "<details><summary>Check your reasoning</summary>In the local CPU check 10,259,842 of 148,140,123 parameters were trained and "
                "validation accuracy went 0.85 → 1.0 → 1.0 → 1.0. It cannot beat the order rule on the test split: 1.0 is the ceiling. The most "
                "the model can show is that it learns the same rule from 56 examples.</details>\n\n"
                "## 8. Held-out evaluation\n\n"
                "`pipe.evaluate` classifies every record of a split and reports `accuracy` (discrete correctness under the argmax rule), `macro_f1` "
                "(the unweighted mean of per-class F1, which exposes a model that ignores a minority class), per-class precision/recall/F1 with support, "
                "and `auroc` (ranking quality of the positive-class score, independent of the argmax threshold; for more than two classes it is the "
                "macro one-vs-rest average). The **test split** was never used for training or monitoring, so its numbers are the independent evidence "
                "(SPL6/SPL7); the validation split is shown alongside for comparison. These are tutorial metrics on a synthetic 20-record split "
                "(EVAL6): a single holdout, no dispersion estimate. The cell prints the fine-tuned model next to every baseline and writes the report, "
                "with the baselines and the deltas against them, to `outputs/{stem}_evaluation_report.json`. It also adds this run to `run_history`, "
                "which Section 12 prints.\n\n"
                "**Predict before running:** what test accuracy, macro-F1 and AUROC do you expect, and what would a perfect score here tell you "
                "about ESM-2?"
            ),
            "code": _EVAL_CODE,
        },
        {
            "md": (
                "**What to notice:** the test row beside the four baseline rows, and `delta_vs_baselines` — positive against the order-blind ones, "
                "zero against the order rule.\n\n"
                "<details><summary>Check your reasoning</summary>In the local CPU check and in the recorded Kaggle T4 run of the previous version "
                "the test split scored accuracy, macro-F1 and AUROC 1.0 (n = 20) — equal to the order rule, 0.25 above the composition centroid. "
                "That tells you the bounded fine-tuning learned the rule that defines the classes; it does not tell you that pretraining is what "
                "made it possible (a randomly initialised encoder fine-tuned the same way might learn it too — this notebook does not run that), "
                "and 20 synthetic sequences say nothing about real proteins.</details>\n\n"
                "## 9. Inference on new sequences\n\n"
                "`pipe.classify` returns, per sequence, the argmax `label`, its `score` and the full `scores` dictionary in class order. The scores are "
                "softmax outputs of a head trained on a few dozen sequences — **not calibrated probabilities** (UNC2); the only decision rule is argmax "
                "(UNC3), and a deployment that must trade false positives against false negatives owns its own threshold. On the default path the "
                "new sequences are generated with a different seed, so their true labels are known and shown as a check; on the BYOD path the first "
                "six test-split records stand in as new data (INF2). Predictions are written to `outputs/{stem}_predictions.csv` with ids and per-class scores.\n\n"
                "**Predict before running:** the model was perfect on the test split. Will its scores on six new sequences be close to 1.0?"
            ),
            "code": _INFER_CODE,
        },
        {
            "md": (
                "**What to notice:** six predictions with their true labels, the scores, and `matches`.\n\n"
                "<details><summary>Check your reasoning</summary>Not necessarily. In the local CPU check all 6 were right, with scores between 0.57 "
                "and 0.85: correct argmax labels from a head that is far from saturated after four epochs. A score is a softmax output, not a "
                "probability that the label is right.</details>\n\n"
                "## 10. Export the adapter and verify a fresh reload\n\n"
                "`pipe.save_artifact` writes only the trained tensors (head, final layer norm and the unfrozen encoder layers, about 41 MB for two "
                "layers) as `adapter.safetensors` — **safetensors** is a file format that stores tensors without executable code — plus a "
                "`manifest.json` that records the artifact format, the exact base model id and revision the tensors belong to (ART4), the class order, "
                "the tensor names, the file size and SHA-256, and the adaptation configuration (OUT8). `ESM2Pipeline.from_artifact` then re-verifies "
                "the base snapshot, checks the artifact manifest and digests **before** deserialising, rebuilds the classifier and overlays the "
                "tensors — a fresh object from files, not the in-memory model (VER2). The cell compares its predictions on the same new sequences with "
                "the pre-export ones: labels must match exactly and scores within `1e-5` (VER4).\n\n"
                "**Predict before running:** will the reloaded model's scores be identical to the in-memory ones, or only close?"
            ),
            "code": _EXPORT_CODE,
        },
        {
            "md": (
                "**What to notice:** the manifest summary (classes, `trainable_layers`, tensor count, file size and SHA-256), `reload_parity: PASS` and "
                "`max_abs_score_diff`.\n\n"
                "<details><summary>Check your reasoning</summary>Identical on the same device: the local CPU check measured a maximum difference of "
                "0.0. On a GPU a difference in the last digits is possible, which is why the check allows `1e-5`. Loading a file is not reproducing "
                "a result; the parity check is what shows the export is the model that was evaluated.</details>\n\n"
                "## 11. Result export and provenance\n\n"
                "The last output, `outputs/{stem}_result.json`, gathers everything a reader needs to interpret the files above: the notebook source "
                "revision, the model id, immutable revision and licence, the dataset source and digest, the adaptation configuration, baseline and "
                "held-out metrics, the new-sequence predictions, the artifact manifest, the reload-parity result, the run history, and the runtime "
                "versions and device (OUT6/OUT7). The cell asserts that the report, the artifact and this file describe the same model. No credential "
                "is involved anywhere in this notebook, so none can leak into it (OUT10).\n\n"
                "**Predict before running:** how many files will `outputs/` hold after a default run?"
            ),
            "code": _RESULT_CODE,
        },
        {
            "md": (
                "**What to notice:** seven files — the sample CSV, the embeddings, the evaluation report, the predictions and the result file at "
                "the top level, and `adapter.safetensors` plus `manifest.json` in the adapter folder.\n\n"
                "<details><summary>Check your reasoning</summary>Seven: five files at the top level and two inside "
                "`outputs/esm2_protein_adapter/` (the local CPU check listed exactly these; the adapter file is about 40 MB). After a BYOD run "
                "`esm2_protein_byod_dataset.csv` is added, and a rerun overwrites the others, so they always describe the latest run.</details>\n\n"
                "## 12. Your turn — change one thing: train the head alone\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.**\n\n"
                "1. **Predict:** with `TRAINABLE_LAYERS = 0` only the classification head is trained (about 0.4 M parameters instead of 10.3 M), "
                "reading the frozen pretrained features. Will the test accuracy stay at 1.0? Will the AUROC? Write your guess down.\n"
                "2. **Change one thing:** in Section 7 set `TRAINABLE_LAYERS = 0` and nothing else.\n"
                "3. **Run:** select the Section 7 cell and choose **Runtime → Run after** (it re-runs Sections 7–12, so the report, the predictions, "
                "the adapter and the result file are all rewritten for this run and describe one model). `adapt` starts again from the pretrained "
                "base.\n"
                "4. **Observe:** this cell prints one row per Section 7 run in this session — the setting, the trained parameter count, the final "
                "validation accuracy and the test accuracy, macro-F1 and AUROC.\n"
                "5. **Explain:** what did training the last two encoder layers buy over the head alone on this task, and can 20 test sequences "
                "tell you?\n\n"
                "<details><summary>Check your reasoning</summary>In the local CPU check the head alone trained 411,522 parameters and scored test "
                "accuracy 0.90 with AUROC 1.0, against 1.0 / 1.0 for two layers. AUROC 1.0 means the head still ranks every `segment` sequence above "
                "every `scattered` one; only the argmax cut is not yet in the right place for two of the 20 — the frozen features carry the order "
                "signal, and the trained layers sharpen the decision. Two sequences is 10 points on this split, so the difference is real for this "
                "run but small evidence. Other experiments in the same pattern: `EPOCHS = 1`, `LEARNING_RATE = 3e-5`, `TRAINABLE_LAYERS = 4` — each "
                "adds a row.</details>"
            ),
            "code": _ACTIVITY_CODE,
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The fine-tuned model separates `segment` from `scattered` sequences on a 20-record test split — and so does the one-line order rule "
        "`longest_hydrophobic_run >= 18` that defines the classes, while the order-blind 20-residue composition centroid gets part of the way "
        "(0.75 in the local check) because the classes are not equal in composition. That is what the run shows: the adaptation contract — "
        "verified base, bounded fine-tuning, held-out evaluation against baselines, export and reload parity — learns a residue-order rule from "
        "56 training sequences. It does **not** show that ESM-2's pretraining is what makes this possible: that would need the same fine-tuning "
        "from a randomly initialised encoder, which this notebook does not run. The metrics come from one seeded holdout with no dispersion "
        "estimate, and the classes are defined by a generator rule rather than by biology — so a perfect score here says the adaptation "
        "contract works, not that ESM-2 predicts membrane segments, localisation, function or anything else about real proteins. On real data "
        "the same workflow needs homology-aware splits (random splitting of related sequences leaks), labels from experiments or curated "
        "databases, and enough records per class for the numbers to mean something. The softmax scores are uncalibrated; embeddings are "
        "representations that need a labelled downstream task before any quality can be stated.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone notebook, can acquire "
        "and digest-verify the pinned model, validate the demonstrated dataset contract, execute bounded fine-tuning, evaluate against trivial "
        "baselines on an independent split, and emit the shown machine-readable artifacts — without the repository being reachable. "
        "It does **not** establish benchmark superiority, production fitness, the contribution of pretraining, or biological validity of the classes.\n\n"
        "**Next experiments** (each starts after the default Run all; every adaptation starts again from the pretrained base):\n\n"
        "1. **Trainable layers:** the Section 12 activity (`TRAINABLE_LAYERS = 0`, or 4), **Run after** from Section 7.\n"
        "2. **Fewer epochs:** in Section 7 set `EPOCHS = 1`, then **Run after** from Section 7, and compare accuracy with AUROC in the new "
        "Section 12 row. An under-trained head can rank sequences well (high AUROC) before its argmax cut is right (lower accuracy); see "
        "whether your run shows that.\n"
        "3. **Another split:** in Section 4 change `SEED`, then **Run after** from Section 4; the baselines and the fine-tuning are recomputed "
        "on the new split.\n"
        "4. **Your own labelled sequences:** in Section 4 set `USE_BYOD = True` (and upload, or set `BYOD_PATH`), then **Run after** from "
        "Section 4. Read the composition baselines first — if they already score well, your labels may be predictable from composition alone.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment is built from manylinux wheels; use Colab, "
        "Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 fails to download `uv`, Python or a package.** The runtime needs `pypi.org`, `files.pythonhosted.org` and the "
        "python-build-standalone release host. Re-run the cell; a size or SHA-256 mismatch is refused on purpose.\n"
        "- **\"The isolated environment's Python process exited\".** Usually out of memory. Restart the session and choose **Run all** "
        "again; on a small CPU runtime lower `BATCH_SIZE` in Section 7.\n"
        "- **Section 3 reports a size or SHA-256 mismatch.** A snapshot file was altered or truncated; delete "
        "`weights/esm2-t30-150m-ur50d/model.safetensors` and run Section 3 again.\n"
        "- **Section 4: \"Upload exactly one … file (got 0)\".** The upload was cancelled or empty. Run Section 4 again and pick one file, "
        "or set `BYOD_PATH`.\n"
        "- **Section 4: \"… is not UTF-8 text\".** Save the file as UTF-8 (a spreadsheet's 'CSV UTF-8') and run Section 4 again.\n"
        "- **Section 4 refuses a BYOD dataset.** The message names the record and the rule (alphabet, length, duplicate id or sequence, "
        "missing column, too few records or classes), or the split and the number of records a class needs. Fix the file and run "
        "Section 4 again.\n"
        "- **Section 8, 9 or 10: \"requires an adapted head\".** Section 7 has not completed since the dataset last changed; run Section 7 "
        "(**Run after**) first.\n"
        "- **Section 10's parity assertion fails.** The export or reload is broken; run Sections 7–11 again. Do not use that artifact.\n"
        "- **Slow on CPU.** Each epoch trains on 56 sequences through the 30-layer encoder; a GPU runtime is several times faster. Lower "
        "`EPOCHS` for a quicker experiment.\n"
        "- **CUDA out of memory.** Lower `BATCH_SIZE` in Section 7, or switch the runtime to CPU.\n\n"
        "## Glossary\n\n"
        "- **Protein language model / ESM-2:** a transformer trained to predict masked amino acids in millions of protein sequences, so its "
        "hidden states encode which residues fit where (Lin et al., 2022).\n"
        "- **Residue / token:** one amino acid in a sequence; ESM-2 reads one residue per token, plus the special `<cls>` and `<eos>` tokens.\n"
        "- **Embedding / mean pooling:** a fixed-length vector for a whole sequence; here the average of the 640-d states of its residues.\n"
        "- **Encoder layer:** one of the 30 transformer blocks; fine-tuning the last few adapts the highest-level features.\n"
        "- **Classification head:** a small new layer that maps the `<cls>` state to one score per class.\n"
        "- **Fine-tuning / frozen parameters:** continuing training of a pretrained model on labelled data; frozen parameters are left as "
        "they are. **AdamW** is the optimiser; an **epoch** is one pass over the training split.\n"
        "- **Accuracy / macro-F1:** the share of correct labels / the unweighted mean of per-class F1, which punishes a forgotten class.\n"
        "- **AUROC:** the probability that a random positive is scored above a random negative; it measures ranking, not the argmax cut.\n"
        "- **Baseline:** a trivial predictor scored on the same split; **order-blind** baselines ignore residue positions, an "
        "**order-aware** one reads them. **Nearest centroid:** assign the class whose average feature vector is closest.\n"
        "- **Adapter / safetensors / reload parity:** the saved trained tensors / a code-free tensor file format / the check that base + "
        "adapter reproduces the in-memory model's predictions.\n"
        "- **Homology leakage:** related sequences in both training and test splits, which inflates test scores on real proteins.\n\n"
        "## Conclusion (your notes)\n\n"
        "Fill in from the numbers this run printed; keep each claim to what the evidence shows.\n\n"
        "- **Task:** which sequences, which classes, which split sizes?\n"
        "- **Principal result:** the fine-tuned test accuracy, macro-F1 and AUROC, with `n`.\n"
        "- **Baselines / reference:** the four baselines on the same split, and `delta_vs_baselines`. Which baseline equals the model?\n"
        "- **Uncertainty or failure mode:** how many sequences is a difference? What did your Section 12 run change?\n"
        "- **Limitations:** what does this run *not* show (see Interpretation and limits)?\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/esm2-protein-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/esm2-protein-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/esm\n"
        "- Lin, Z. et al. (2022). Language models of protein sequences at the scale of evolution enable accurate structure prediction. bioRxiv 2022.07.20.500902. https://doi.org/10.1101/2022.07.20.500902\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
