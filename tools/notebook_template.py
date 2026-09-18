"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E protein sequence-classification adaptation: ESM-2 150M is
verified from its pinned snapshot, a synthetic order-sensitive dataset is validated and split,
trivial baselines are measured, a bounded AdamW fine-tuning of the classification head plus the
last encoder layers runs in the kernel, and the adapter is exported as safetensors and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "esm2-protein-pipeline"

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

TEMPLATE = {
    "package": "esm2_protein_pipeline",
    "repo_name": REPO,
    "stem": "esm2_protein",
    "notebook_name": "esm2_protein_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned ESM-2 150M snapshot (6 files, ~595 MB), generates the deterministic 96-sequence tutorial dataset in code (no download), "
        "validates it against the sequence-classification contract, splits it into stratified train/validation/test sets, "
        "computes mean-pooled sequence embeddings, measures a majority-class and a composition baseline on the test split, "
        "runs a bounded AdamW fine-tuning of the classification head and the last two encoder layers, evaluates accuracy, "
        "macro-F1 and AUROC on the held-out test split, classifies six freshly generated sequences, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled protein dataset as a CSV (`id,sequence,label` header), JSON array or JSONL file. It passes through the same "
        "validation, stratified split, baselines, adaptation, held-out evaluation, inference, artifact export and reload-parity "
        "cells as the synthetic sample. The expected schema, the alphabet and the ceilings are stated in the Prerequisites and in "
        "Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
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
        "gradient fine-tuning adapts that head together with the last encoder layers). The tutorial dataset is synthetic and "
        "deliberately order-sensitive: every sequence carries the same number of strongly hydrophobic residues, but in class `segment` "
        "they form one contiguous 18–22-residue stretch and in class `scattered` they are spread out. A residue-counting baseline "
        "cannot separate the classes; a model that reads the sequence can — so the comparison shows what fine-tuning adds."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify the "
        "immutable ESM-2 snapshot; validate a labelled protein dataset against explicit ceilings and split it without leakage; "
        "extract and inspect mean-pooled sequence embeddings; measure majority-class and composition baselines; run a bounded "
        "fine-tuning with explicit hyperparameters and a recorded trainable-parameter set; evaluate accuracy, macro-F1 and AUROC "
        "on an independent test split; classify new sequences with argmax scores; and export a safetensors adapter that reloads "
        "against the pinned base with verified prediction parity."
    ),
    "exclusions": (
        "residue-level (token) classification, contact or structure prediction, masked-token scoring, full-parameter "
        "fine-tuning of all 30 layers, or arbitrary unvalidated dataset formats. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU in about a minute of model time on a workstation; CUDA is used automatically when present and shortens adaptation.",
        "- **Knowledge:** amino-acid one-letter codes, what a train/validation/test split protects against, and how accuracy, macro-F1 and AUROC differ.",
        "- **Data contract:** records are `{id, sequence, label}`; sequences are uppercase one-letter strings over `ACDEFGHIKLMNPQRSTVWY` plus the ambiguity codes `BUZOX`, at most 1,022 residues (`MAX_RESIDUES`), unique ids and unique sequences; at least 12 records and 3 per class, 2..20 classes. BYOD accepts CSV (`id,sequence,label`), JSON array or JSONL. Do not upload confidential or restricted data to a hosted runtime unless authorized.",
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
                "`SEED` and cuts 20 % validation / 20 % test, so all three splits keep the class balance; random splitting assumes the records are "
                "independent, which holds for generated data and must be checked for real proteins (homologous sequences across splits leak).\n\n"
                "Look for: 96 records, classes `['scattered', 'segment']`, splits 56/20/20, and a written `outputs/{stem}_sample_dataset.csv` — the "
                "exact file shape BYOD expects. To use your own data, set `USE_BYOD = True` and re-run from this cell."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'VAL_FRACTION = 0.2  # @param {{type:"number"}}\n'
                'TEST_FRACTION = 0.2  # @param {{type:"number"}}\n'
                'SEED = 42  # @param {{type:"integer"}}\n\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    records = generate_sample_dataset()\n"
                "    data_source = f'synthetic hydrophobic-segment dataset (seed {{SAMPLE_SEED}}, {{SAMPLE_SIZE}} records)'\n\n"
                "dataset_manifest = validate_dataset(records)\n"
                "CLASSES = dataset_manifest['classes']\n"
                "splits = split_dataset(records, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(records, 'outputs/{stem}_sample_dataset.csv')\n\n"
                "print({{'data_source': data_source, 'n_records': dataset_manifest['n_records'], 'classes': CLASSES, 'class_counts': dataset_manifest['class_counts']}})\n"
                "print({{'residues': dataset_manifest['residues'], 'ceilings': dataset_manifest['ceilings'], 'digest': dataset_manifest['digest'][:16] + '...'}})\n"
                "print({{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}})\n"
                "example = train_records[0]\n"
                "print({{'example_id': example['id'], 'label': example['label'], 'residues': len(example['sequence']), 'longest_hydrophobic_run': longest_hydrophobic_run(example['sequence']), 'hydrophobic_fraction': round(hydrophobic_fraction(example['sequence']), 3)}})\n"
                "print(example['sequence'])"
            ),
        },
        {
            "md": (
                "## 5. Sequence embeddings (representation, not prediction)\n\n"
                "`pipe.embed` runs the verified base encoder and returns one 640-dimensional vector per sequence: the mean of the last hidden state over "
                "residue tokens only (`<cls>`, `<eos>` and padding are excluded). Embeddings are representations — they carry no label and no metric of "
                "their own; a downstream labelled task is what gives them meaning (EVAL9). The cell embeds eight validation sequences, writes them with "
                "their ids to `outputs/{stem}_embeddings.csv` (OUT4), and prints the mean cosine similarity within and between classes as an inspection, "
                "not an evaluation: the pretrained model has never seen this synthetic rule, so do not expect a clean separation before adaptation."
            ),
            "code": (
                "import csv\n"
                "import math\n\n"
                "embed_records = val_records[:8]\n"
                "embedding_result = pipe.embed([r['sequence'] for r in embed_records], names=[r['id'] for r in embed_records])\n"
                "vectors = embedding_result['embeddings']\n"
                "print({{'n_sequences': embedding_result['n_sequences'], 'dimension': embedding_result['dimension'], 'pooling': embedding_result['pooling']}})\n\n"
                "def cosine(a, b):\n"
                "    dot = sum(x * y for x, y in zip(a, b))\n"
                "    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))\n\n"
                "within, between = [], []\n"
                "for i in range(len(embed_records)):\n"
                "    for j in range(i + 1, len(embed_records)):\n"
                "        sim = cosine(vectors[i], vectors[j])\n"
                "        (within if embed_records[i]['label'] == embed_records[j]['label'] else between).append(sim)\n"
                "print({{'mean_cosine_within_class': round(sum(within) / len(within), 4) if within else None, 'mean_cosine_between_classes': round(sum(between) / len(between), 4) if between else None, 'note': 'inspection only; embeddings are unlabelled representations'}})\n\n"
                "with open('outputs/{stem}_embeddings.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['id', 'label'] + [f'dim_{{k}}' for k in range(embedding_result['dimension'])])\n"
                "    for r, vec in zip(embed_records, vectors):\n"
                "        writer.writerow([r['id'], r['label']] + [f'{{x:.6f}}' for x in vec])\n"
                "print('wrote outputs/{stem}_embeddings.csv')"
            ),
        },
        {
            "md": (
                "## 6. Baselines on the test split\n\n"
                "Two trivial predictors set the floor before any training (EVAL10/EVAL11). `majority_baseline` predicts the most frequent training class "
                "for every test record — 0.5 accuracy on a balanced split. `composition_baseline` fits one threshold on the hydrophobic residue **fraction** "
                "using the training split only (SPL8) and applies it to the test split; because the two synthetic classes share their composition by "
                "construction, it should also land near chance. A fine-tuned model that clears both has learned something about residue *order*. "
                "Both baselines are reported with the same `classification_metrics` fields as the model, so the numbers are directly comparable."
            ),
            "code": (
                "baseline_majority = majority_baseline(train_records, test_records, CLASSES)\n"
                "print({{k: baseline_majority[k] for k in ('baseline', 'predicted_label', 'accuracy', 'macro_f1')}})\n"
                "if len(CLASSES) == 2:\n"
                "    baseline_composition = composition_baseline(train_records, test_records, CLASSES)\n"
                "    print({{k: baseline_composition[k] for k in ('baseline', 'rule', 'train_accuracy', 'accuracy', 'macro_f1', 'auroc')}})\n"
                "else:\n"
                "    baseline_composition = None\n"
                "    print('composition baseline is defined for binary tasks only; skipped for', len(CLASSES), 'classes')"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning\n\n"
                "`pipe.adapt` builds `EsmForSequenceClassification` from the verified base weights (the head is newly initialised — the log line says so), "
                "freezes every parameter except the classification head, the final layer norm and the last `TRAINABLE_LAYERS` encoder layers, and runs "
                "AdamW with the hyperparameters below (FT4/FT6): these are tutorial values chosen for a one-minute CPU run, not production settings. "
                "Validation metrics are computed after each epoch for **monitoring only**; the final epoch's weights are kept, so no selection happens "
                "on the validation split (EVAL14). Training loss going down is optimisation evidence, not task-quality evidence (FT7) — Section 8 is where "
                "quality is measured. Look for the trainable/total parameter counts (about 10.3 M of 148 M with two layers) and validation accuracy "
                "rising above the 0.5 baseline within the first epochs."
            ),
            "code": (
                "import time\n\n"
                'EPOCHS = 4  # @param {{type:"integer"}}\n'
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 8  # @param {{type:"integer"}}\n'
                'TRAINABLE_LAYERS = 2  # @param {{type:"integer"}}\n\n'
                "started = time.perf_counter()\n"
                "adapt_result = pipe.adapt(\n"
                "    train_records,\n"
                "    val_records,\n"
                "    classes=CLASSES,\n"
                "    epochs=EPOCHS,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    trainable_layers=TRAINABLE_LAYERS,\n"
                "    seed=SEED,\n"
                ")\n"
                "adapt_seconds = round(time.perf_counter() - started, 1)\n"
                "print({{'method': adapt_result['method'], 'trainable_parameters': adapt_result['trainable_parameters'], 'total_parameters': adapt_result['total_parameters'], 'precision': adapt_result['precision'], 'device': pipe.device, 'seconds': adapt_seconds}})\n"
                "for step in adapt_result['history']:\n"
                "    print({{k: step[k] for k in step}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "`pipe.evaluate` classifies every record of a split and reports `accuracy` (discrete correctness under the argmax rule), `macro_f1` "
                "(the unweighted mean of per-class F1, which exposes a model that ignores a minority class), per-class precision/recall/F1 with support, "
                "and `auroc` (ranking quality of the positive-class score, independent of the argmax threshold; for more than two classes it is the "
                "macro one-vs-rest average). The **test split** was never used for training or monitoring, so its numbers are the independent evidence "
                "(SPL6/SPL7); the validation split is shown alongside for comparison. These are tutorial metrics on a synthetic 20-record split "
                "(EVAL6): a single holdout, no dispersion estimate. The report, with both baselines and the deltas against them, is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "val_metrics = pipe.evaluate(val_records)\n"
                "test_metrics = pipe.evaluate(test_records)\n"
                "print({{'split': 'validation', **{{k: val_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "print({{'split': 'test', **{{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "for cls_name, row in test_metrics['per_class'].items():\n"
                "    print({{'class': cls_name, **row}})\n\n"
                "evaluation_report = {{\n"
                "    'task': 'protein sequence classification (bounded fine-tuning of ESM-2 150M)',\n"
                "    'evidence': 'tutorial sample-sanity metrics on one stratified holdout; not a benchmark',\n"
                "    'estimation': 'single train/validation/test split, seed ' + str(SEED) + ', no dispersion estimate',\n"
                "    'data_source': data_source,\n"
                "    'dataset_digest': dataset_manifest['digest'],\n"
                "    'classes': CLASSES,\n"
                "    'splits': {{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}},\n"
                "    'baselines': {{'majority': baseline_majority, 'composition': baseline_composition}},\n"
                "    'validation_metrics': val_metrics,\n"
                "    'test_metrics': test_metrics,\n"
                "    'delta_vs_majority': {{k: round(test_metrics[k] - baseline_majority[k], 4) for k in ('accuracy', 'macro_f1')}},\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k != 'trainable_parameter_names'}},\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "print({{'delta_vs_majority': evaluation_report['delta_vs_majority'], 'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Inference on new sequences\n\n"
                "`pipe.classify` returns, per sequence, the argmax `label`, its `score` and the full `scores` dictionary in class order. The scores are "
                "softmax outputs of a head trained on a few dozen sequences — **not calibrated probabilities** (UNC2); the only decision rule is argmax "
                "(UNC3), and a deployment that must trade false positives against false negatives owns its own threshold. On the default path the "
                "new sequences are generated with a different seed, so their true labels are known and shown as a check; on the BYOD path the first "
                "six test-split records stand in as new data (INF2). Predictions are written to `outputs/{stem}_predictions.csv` with ids and per-class scores."
            ),
            "code": (
                "if USE_BYOD:\n"
                "    new_records = test_records[:6]\n"
                "    new_source = 'first six BYOD test-split records'\n"
                "else:\n"
                "    new_records = generate_sample_dataset(seed=7, size=6)\n"
                "    new_source = 'freshly generated sequences (seed 7)'\n"
                "input_manifest = validate_inputs([r['sequence'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "print({{'new_source': new_source, 'verdict': input_manifest['verdict'], 'n_sequences': input_manifest['n_sequences'], 'max_residues_observed': input_manifest['max_residues_observed']}})\n"
                "inference_result = pipe.classify([r['sequence'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "predictions = inference_result['predictions']\n"
                "print({{'decision_rule': inference_result['decision_rule']}})\n"
                "n_match = 0\n"
                "for p, r in zip(predictions, new_records):\n"
                "    n_match += p['label'] == r['label']\n"
                "    print({{'id': p['id'], 'predicted': p['label'], 'score': round(p['score'], 4), 'true_label': r['label']}})\n"
                "print({{'matches': n_match, 'of': len(new_records), 'note': 'sanity check on generated labels, not an evaluation'}})\n\n"
                "with open('outputs/{stem}_predictions.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['id', 'residues', 'predicted_label', 'score'] + [f'score_{{c}}' for c in CLASSES])\n"
                "    for p in predictions:\n"
                "        writer.writerow([p['id'], p['residues'], p['label'], f\"{{p['score']:.6f}}\"] + [f\"{{p['scores'][c]:.6f}}\" for c in CLASSES])\n"
                "print('wrote outputs/{stem}_predictions.csv')"
            ),
        },
        {
            "md": (
                "## 10. Export the adapter and verify a fresh reload\n\n"
                "`pipe.save_artifact` writes only the trained tensors (head, final layer norm and the unfrozen encoder layers, about 41 MB for two "
                "layers) as `adapter.safetensors` plus a `manifest.json` that records the artifact format, the exact base model id and revision the "
                "tensors belong to (ART4), the class order, the tensor names, the file size and SHA-256, and the adaptation configuration (OUT8). "
                "`ESM2Pipeline.from_artifact` then re-verifies the base snapshot, checks the artifact manifest and digests **before** deserialising, "
                "rebuilds the classifier and overlays the tensors — a fresh object from files, not the in-memory model (VER2). The cell compares its "
                "predictions on the same new sequences with the pre-export ones: labels must match exactly and scores within `1e-5` (VER4)."
            ),
            "code": (
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "pipe.save_artifact(artifact_dir, metadata={{'data_source': data_source, 'dataset_digest': dataset_manifest['digest'], 'test_metrics': {{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "with open(artifact_dir / ARTIFACT_MANIFEST_NAME, encoding='utf-8') as f:\n"
                "    artifact_manifest = json.load(f)\n"
                "print({{'format': artifact_manifest['format'], 'base_model': artifact_manifest['base_model'], 'classes': artifact_manifest['classes'], 'n_tensors': len(artifact_manifest['tensors']), 'files': artifact_manifest['files']}})\n\n"
                "reloaded_pipe = ESM2Pipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR)\n"
                "reloaded_result = reloaded_pipe.classify([r['sequence'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "max_score_diff = 0.0\n"
                "for before, after in zip(predictions, reloaded_result['predictions']):\n"
                "    assert before['id'] == after['id'] and before['label'] == after['label'], f'reload parity failure on {{before[\"id\"]}}'\n"
                "    max_score_diff = max(max_score_diff, abs(before['score'] - after['score']))\n"
                "assert max_score_diff < 1e-5, f'reload score drift {{max_score_diff}}'\n"
                "print({{'reload_parity': 'PASS', 'labels_equal': True, 'max_abs_score_diff': max_score_diff, 'loaded_from': reloaded_pipe.adaptation.get('loaded_from_artifact')}})"
            ),
        },
        {
            "md": (
                "## 11. Result export and provenance\n\n"
                "The last output, `outputs/{stem}_result.json`, gathers everything a reader needs to interpret the files above: the notebook source "
                "revision, the model id, immutable revision and licence, the dataset source and digest, the adaptation configuration, baseline and "
                "held-out metrics, the new-sequence predictions, the artifact manifest, the reload-parity result, and the runtime versions and device "
                "(OUT6/OUT7). No credential is involved anywhere in this notebook, so none can leak into it (OUT10)."
            ),
            "code": (
                "import platform\n\n"
                "result_payload = {{\n"
                "    'task': 'protein sequence classification adaptation (ESM-2 150M)',\n"
                "    'pipeline_class': 'ESM2Pipeline',\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'data_source': data_source,\n"
                "    'dataset_manifest': dataset_manifest,\n"
                "    'embedding_summary': {{'n_sequences': embedding_result['n_sequences'], 'dimension': embedding_result['dimension'], 'pooling': embedding_result['pooling']}},\n"
                "    'evaluation_report': evaluation_report,\n"
                "    'inference': {{'new_source': new_source, 'decision_rule': inference_result['decision_rule'], 'predictions': predictions}},\n"
                "    'artifact_format': ARTIFACT_FORMAT,\n"
                "    'artifact_format_version': ARTIFACT_FORMAT_VERSION,\n"
                "    'artifact_manifest': artifact_manifest,\n"
                "    'reload_parity': {{'labels_equal': True, 'max_abs_score_diff': max_score_diff}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'safetensors': safetensors.__version__,\n"
                "        'device': pipe.device,\n"
                "        'precision': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The fine-tuned head separates `segment` from `scattered` sequences that share their residue composition, which the composition "
        "baseline cannot do: the model is using residue order, exactly what a protein language model is pretrained to represent. That is the "
        "whole claim of this notebook. The test split has 20 synthetic records, the metrics come from one seeded holdout with no dispersion "
        "estimate, and the classes are defined by a generator rule rather than by biology — so a perfect score here says the adaptation "
        "contract works, not that ESM-2 predicts membrane segments, localisation, function or anything else about real proteins. On real data "
        "the same workflow needs homology-aware splits (random splitting of related sequences leaks), labels from experiments or curated "
        "databases, and enough records per class for the numbers to mean something. The softmax scores are uncalibrated; embeddings are "
        "representations that need a labelled downstream task before any quality can be stated.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone notebook, can acquire "
        "and digest-verify the pinned model, validate the demonstrated dataset contract, execute bounded fine-tuning, evaluate against trivial "
        "baselines on an independent split, and emit the shown machine-readable artifacts — without the repository being reachable. "
        "It does **not** establish benchmark superiority, production fitness, or biological validity of the classes.\n\n"
        "**Optional experiments (do not affect the default path):** set `TRAINABLE_LAYERS = 0` to train the head alone and compare the test "
        "metrics with the two-layer run; lower `EPOCHS` to 1 to see an under-trained head whose AUROC may still be high while accuracy sits near "
        "0.5 (ranking before thresholding); or bring a real labelled dataset through BYOD and read the composition baseline first — if it already "
        "scores well, your labels may be predictable from composition alone.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/esm2-protein-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/esm2-protein-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/esm\n"
        "- Lin, Z. et al. (2022). Language models of protein sequences at the scale of evolution enable accurate structure prediction. bioRxiv 2022.07.20.500902. https://doi.org/10.1101/2022.07.20.500902"
    ),
}
