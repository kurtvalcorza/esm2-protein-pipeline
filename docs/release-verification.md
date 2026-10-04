# Release verification

`tutorials/esm2_protein_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; exactly two kernel cells — the isolated
  install (pinned `uv` wheel checked by size and SHA-256, managed CPython, `--require-hashes --only-binary :all:`
  over the carried lock) and the router to the isolated worker — and seven cells titled `Infrastructure` and
  collapsed; `NOTEBOOK_SOURCE` recorded in exports; the guided-layer markers and the absence of the removed
  claims (review ESM-M1, ESM-M2, ESM-M4);
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ESM2Pipeline.from_pretrained(weights_dir=...)`, `validate_dataset`, `split_dataset`, `write_dataset_csv`,
  `pipe.embed`, `majority_baseline`, `composition_baseline`, `residue_composition_baseline`, `longest_run_baseline`,
  `validate_splits`, `pipe.reset_adaptation`, `pipe.adapt` with its explicit hyperparameters,
  `pipe.evaluate` on both the validation and the test split, `validate_inputs`, `pipe.classify`,
  `pipe.save_artifact`, `ESM2Pipeline.from_artifact` and the reload-parity assertion), the six expected `outputs/`
  paths, the learner-facing statements (scores are not calibrated probabilities, validation is monitoring only,
  embeddings are representations, the excluded capabilities, homology-aware splitting) and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, direct `transformers` / `huggingface_hub` / `safetensors` use **outside the
  carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`,
  `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected backends and temporary manifests, no
weights). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or GPU runtime (Linux x86_64; CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh Linux x86_64 CPU or GPU container; the committed notebook executed verbatim with **no repository checkout** (the notebook is standalone), its first cell building the isolated environment | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` (the isolated environment is skipped), a `google.colab` shim and pre-installed pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/esm2-t30-150m-ur50d/` (the standalone path writes the manifest itself and stages every listed
   file, so the directory may not be seeded);
3. choose **Run all once** without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `BYOD_PATH = ''`, `VAL_FRACTION = 0.2`, `TEST_FRACTION = 0.2`, `SEED = 42`, `EPOCHS = 4`,
   `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 8`, `TRAINABLE_LAYERS = 2`); the run must complete every code cell with no
   error output and **no restart** — record `restarted: false`; a run that needed a restart is not promotion
   evidence (RUN1, RUN10);
4. verify that Section 1 builds the isolated environment (it prints the isolated Python 3.12.12 and the number of
   locked packages), that the runtime cell reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision
   recorded in `metadata.dimer.generated_from`, and that the imported core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`,
   `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`;
5. verify every default-path stage completes:
   - the isolated environment built from the carried hash-locked lock with no GitHub access, and every later cell
     routed to it;
   - the three carried module cells execute (defining `ESM2Pipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `validate_dataset`, `split_dataset`, `generate_sample_dataset`, `write_dataset_csv`,
     `classification_metrics`, `majority_baseline`, `composition_baseline`) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the 6 entries fetched from `facebook/esm2_t30_150M_UR50D` at the immutable revision, and
     `verify_snapshot` reporting 6 verified files before the model loads;
   - the dataset manifest printed with 96 records, classes `['scattered', 'segment']`, 48/48 class counts, the
     ceilings and the digest, and the splits 56 / 20 / 20;
   - `pipe.embed` reporting 640-dimensional vectors and writing `outputs/esm2_protein_embeddings.csv`;
   - four baselines reported on the test split (majority accuracy 0.5; hydrophobic fraction near chance; the
     20-residue composition centroid above chance; the longest-hydrophobic-run rule at or near 1.0);
   - `pipe.adapt` reporting the trainable/total parameter counts and a four-epoch history with per-epoch validation
     metrics;
   - `pipe.evaluate` reporting validation and test accuracy, macro-F1, AUROC and per-class rows, and writing
     `outputs/esm2_protein_evaluation_report.json` with both baselines and the delta against the majority baseline;
   - `validate_inputs` accepting the six new sequences and `pipe.classify` writing
     `outputs/esm2_protein_predictions.csv` with per-class scores;
   - `pipe.save_artifact` writing `outputs/esm2_protein_adapter/{adapter.safetensors,manifest.json}`, and
     `ESM2Pipeline.from_artifact` reloading it with identical labels and a maximum absolute score difference below
     `1e-5` (the cell asserts both);
   - `outputs/esm2_protein_result.json` written with `NOTEBOOK_SOURCE`, the model identity, revision and licence,
     the dataset manifest, the evaluation report, the predictions, the artifact manifest and the runtime versions;
6. verify the exports exist and the interpretation section matches the observed path;
   then, in the same runtime, the **BYOD gate (REL12)**: set `USE_BYOD = True` with a representative labelled file
   (for example 30 records in 3 classes) and **Run after** from Section 4 — it must pass through adaptation,
   evaluation, export and reload with the BYOD data source in the adapter metadata — and record one refused input
   (for example a UTF-16 file or an empty upload) with its one-line message;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache and the weights directory were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `esm2_protein_colab.ipynb` | `22df052` / `638ec73e83d6` | 2026-09-18 | Kaggle batch kernel `dimer-nb2-esm2-protein` v1 (Python 3.12.13, Tesla T4, empty Hugging Face cache, no repository checkout) | **Passed only after a manual restart** — not a one-pass Run all, not promotion evidence. 13/13 code cells after a fresh-process restart following the in-kernel dependency install (review ESM-M1) |
| `esm2_protein_colab.ipynb` | `ae499e1` / `638ec73e83d6` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/esm2_protein_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/esm2_protein_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include the model download where it occurred; they are measurements for the stated runtime, not
general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | `22df052` / `638ec73e83d6` | Kaggle batch kernel `dimer-nb2-esm2-protein` v1 (Python 3.12.13, Tesla T4, clean cache) | Default sample path (download and verify 6 model files → validate → split → embed → baselines → adapt → evaluate → classify → export → reload) | 243.7 s | **Passed only after a manual restart** — not a one-pass Run all, not promotion evidence. 13/13 code cells after one fresh-process restart following the in-kernel install; test accuracy/macro-F1/AUROC 1.0 (n=20) against majority accuracy 0.5; reload parity 0.0. |
| 2026-09-18 | `ae499e1` / `638ec73e83d6` | Local pre-flight harness (Windows, CPython 3.12, CPU float32) | Default sample path (validate → split → embed → baselines → adapt → evaluate → classify → export → reload) | 92.9 s | **PASSED** — pre-flight; hosted clean-runtime run still required |
| 2026-10-04 | review-fix branch (revised notebook, before commit) | Local pre-flight harness (Windows, CPython 3.12, CPU float32, torch 2.14.0+cpu, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, snapshot pre-staged, `CUDA_VISIBLE_DEVICES=-1`) | Default path (16/16 code cells; learner cells 55.4 s); Section 12 activity (`TRAINABLE_LAYERS = 0`, Run after from Section 7: 6/6 cells, test 0.90 / AUROC 1.0, every export `trainable_layers: 0`); BYOD via `BYOD_PATH`: 30-record 3-class JSONL (splits 18/6/6) and 12-record binary CSV (8/2/2), both through export and reload (parity 0.0); refusals: cancelled upload and UTF-16 CSV with one-line messages, and after a refusal evaluation and export refuse (no head) | 334.9 s (all journeys) | **PASSED** — pre-flight only, **not** promotion evidence; the isolated-environment install path (Linux) was not exercised |
| 2026-10-04 | `37f7112` / `c8b7a65eab9b` | Colab CLI 0.7.4 sequential execution, fresh Colab VM, Tesla T4 (kernel Python 3.13.15; isolated Python 3.12.12, torch 2.14.0+cu130, transformers 4.57.6, safetensors 0.8.0, `cuda: True`; empty Hugging Face cache, no repository checkout) — not a browser Run all | Default settings only (download and verify 6 model files → validate → split → embed → baselines → adapt → evaluate → classify → export → reload) | 98.2 s (session wall time, including the 49 s environment build) | **PASSED — one pass, no restart, 0 errors.** 16/16 code cells in order (see the record below) |

### 2026-10-04 Colab CLI T4 run of `37f7112` (blob `c8b7a65eab9b`)

- **Executor:** Google Colab CLI 0.7.4 (`colab exec -f`) on a fresh Colab VM with a Tesla T4. The CLI executes every
  code cell in order in one kernel; it is **not** a browser Run all, it records no execution counts (order is taken
  from the `Executing cell k/16` lines in the log), and it renders no forms.
- **Source identity:** the notebook was fetched from `raw.githubusercontent.com` at the full commit
  `37f7112b75fa0e2b5dd0fc470ea0d100e21b008b`; its Git blob `c8b7a65eab9b91438e91f48054f1a77bbceb55ed` was checked
  before the VM was allocated, and the executed notebook's 16 code-cell sources equal the committed ones.
- **Outcome:** 16/16 code cells, one pass, **no restart** (`restarted: false`), 0 error outputs, no output asks for a
  restart. Cells 4–6 (the carried `pipeline`, `samples` and `metrics` modules) print nothing by design.
- **Runtime (cells 1–3):** isolated environment `/content/dimer_isolated_env`, Python 3.12.12, 47 locked packages,
  built in 49 s; every later cell routed to it; `NOTEBOOK_SOURCE.repository_revision` `23be8f5` equals
  `metadata.dimer.generated_from`.
- **Model (cell 7):** `facebook/esm2_t30_150M_UR50D` at `a695f6045e2e32885fa60af20c13cb35398ce30c` (MIT), 6 files
  (595,260,503 bytes) fetched and 6 verified; device `cuda:0`.
- **Observed metrics** (observations, not a benchmark; all equal to the worked answers quoted from the local CPU
  check): 96 records, 48/48, splits 56/20/20; embedding cosine within 0.961 / between 0.9434; test baselines majority
  0.5 (macro-F1 0.3333), composition centroid 0.75 (AUROC 0.8), hydrophobic fraction 0.45 (AUROC 0.47), longest
  hydrophobic run 1.0; 10,259,842 of 148,140,123 parameters trained (4.8 s), validation accuracy 0.85 → 1.0 → 1.0 →
  1.0; test accuracy, macro-F1 and AUROC 1.0 (n = 20); 6/6 new sequences correct (scores 0.572–0.8462); adapter
  reload parity PASS, maximum score difference 0.0; seven files under `outputs/`.
- **Evidence files** (`docs/execution-evidence/2026-10-04/`, byte-exact copies):
  - `esm2_protein_colab_37f7112_colab-cli-t4.ipynb` — SHA-256 `7001b067cde61263a74c928e709d1cfe13f7dcfe360122f3f7004e982f12b49b`
  - `esm2_protein_colab_37f7112_colab-cli-t4_exec.log` — SHA-256 `7c8c36046e5375826e16c52bce950bad4e456b99de48b5dc3e9eb18996015813`
  - `esm2_protein_colab_37f7112_colab-cli-t4_run_summary.json` — SHA-256 `4ef640738f75e760e8d38cb0078f6ca9992943a3a94b18e5e25998b13c1604ac`
- **Not exercised:** a browser Run all, the BYOD gate (REL12, step 6) and its refusals, and the Section 12 activity
  (`TRAINABLE_LAYERS = 0`; its worked answer of test 0.90 / AUROC 1.0 comes from the local CPU check only).

## Current status

**Candidate.** The 2026-09-18 promotion rested on a Kaggle run of blob `638ec73e83d6` that completed only after a
manual restart following the in-kernel install; under NOTEBOOK_SPEC 2.2 (RUN1, RUN10) that is not a one-pass Run all,
so the Notebook Review Framework v1 review (2026-10-03, ESM-M1) returned the repository to Candidate. The revised
notebook builds an isolated, hash-locked environment instead (Linux x86_64 only), and also fixes the review's
baseline (ESM-M2), BYOD (ESM-M3, ESM-m2), guided-layer (ESM-M4) and activity (ESM-m1) findings. Its local CPU
pre-flight (above) is not promotion evidence. On 2026-10-04 the current blob `c8b7a65eab9b` (commit `37f7112`)
completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 under the Colab CLI (16/16 code cells,
recorded above). That run is sequential CLI execution, not a browser Run all. Status stays **Candidate**: promotion
still needs the BYOD gate of step 6 (REL12) and the maintainer's approval.
