# ESM-2 150M Protein Sequence-Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/esm2-protein-pipeline`  
**Notebook:** `tutorials/esm2_protein_colab.ipynb`  
**Reviewed commit:** `d9d0f1a5f24108b9a65038e179628fc2363f77a7` (`main`, confirmed with `gh api repos/kurtvalcorza/esm2-protein-pipeline/commits/main`)  
**Notebook Git blob:** `638ec73e83d6708dc7259e058c8656c2bb6fce56`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-18 (commit `22df052`); the notebook last changed in `ae499e1`.  
**Finding prefix:** `ESM`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `b1cfe13`. The notebook declares 2.0.

## Executive assessment

The default path is solid engineering. The notebook carries `pipeline.py`, `samples.py` and `metrics.py` verbatim (generator parity enforced), asserts the inline manifest against the module identity, stages and re-hashes the pinned `facebook/esm2_t30_150M_UR50D` snapshot, validates a seeded 96-sequence dataset, splits it 56/20/20 stratified, runs majority-class and hydrophobic-fraction baselines, fine-tunes the head plus the last two encoder layers, evaluates on an independent test split, and exports a safetensors adapter that reloads from files with exact score parity. Its prose on uncalibrated softmax scores, validation-as-monitoring and homology leakage is accurate.

A direct CPU run of every code cell at the defaults reproduced the hosted record:

| Measure | This review (CPU, pins, defaults) | Kaggle T4 record (blob `638ec73e`) |
|---|---|---|
| Code cells completed | 13/13 (65.0 s; install skipped) | 13/13 after one restart |
| Dataset / splits | 96 records, 48/48, splits 56/20/20 | identical |
| Majority / composition baseline (test accuracy) | 0.50 / 0.45 | 0.5 / "near chance" |
| Trainable / total parameters | 10,259,842 / 148,140,123 | — |
| Validation accuracy per epoch | 0.85, 1.0, 1.0, 1.0 | — |
| Test accuracy / macro-F1 / AUROC (n=20) | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| Reload parity (max abs score diff) | 0.0 | 0.0 |

Four problems stand between this notebook and `Ready for intended use`:

1. **No one-pass `Run all` (ESM-M1).** The recorded hosted run needed a restart after the in-kernel `pip install`; the release record calls it PASS and the repository is marked Release-grade.
2. **The central conclusion is not supported by the experiment (ESM-M2).** The notebook says the classes "share their residue composition" and that "a residue-counting baseline cannot separate the classes", so the perfect score shows the model "is using residue order, exactly what a protein language model is pretrained to represent". On the same split, a one-line order rule the notebook itself defines (`longest_hydrophobic_run >= 18`) scores 1.0, and two order-blind composition baselines score 0.75 and 0.80, because the two classes draw their non-hydrophobic residues from different distributions.
3. **BYOD fails for datasets the stated contract accepts (ESM-M3).** The contract says ≥12 records and ≥3 per class. A binary dataset needs at least 56 records to get past `pipe.adapt`, because the validation and test splits are each re-validated against the 12-record minimum. A 30-record, 3-class dataset failed at cell 19; continuing past the error exported the earlier sample-trained adapter stamped with the BYOD file name.
4. **Guided layer largely absent (ESM-M4).** Declared `GUIDED`, but there is no stated audience, how-to-use, roadmap, glossary, prediction prompt, checkpoint, troubleshooting section or conclusion scaffold, and 49,139 characters of carried module code are not labelled as infrastructure.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: "amino-acid one-letter codes, what a train/validation/test split protects against, and how accuracy, macro-F1 and AUROC differ" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default, CUDA used when present |
| Promised outcomes | Pinned install; carried modules; digest-verified snapshot; validated dataset with ceilings and digest; leakage-free stratified split; inspected mean-pooled embeddings; majority and composition baselines; bounded fine-tune with recorded trainable set; accuracy / macro-F1 / AUROC on an independent test split; argmax inference on new sequences; safetensors adapter with verified reload parity; six `outputs/` files plus the dataset template; BYOD through "the same validation, stratified split, baselines, adaptation, held-out evaluation, inference, artifact export and reload-parity cells"; optional experiments |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `2bbcf0b` |
| Release status | `Release-grade` (`README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`) |

### Evidence actually obtained

- **Source inspection.** All 29 cells (13 code; cells 5, 7 and 9 are the carried modules). Also read: the three package modules, the generator and template, `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/`.
- **Static checks (this review).** `tools/build_notebook.py --check` exit 0 (notebook up to date); `tools/validate_release_assets.py` exit 0; `pytest -q tests` with `PYTHONPATH=src` 41 passed, exit 0. All code cells compile; no `TODO`/`FIXME` markers; no persisted outputs. Static checks are not execution evidence (REL8).
- **Documented execution evidence.** `docs/release-verification.md`: Kaggle batch kernel, Tesla T4, Python 3.12.13, empty Hugging Face cache, no repository checkout, **the reviewed blob**, 2026-09-18: "13/13 code cells after the expected fresh-process restart following dependency installation", 243.7 s, test accuracy/macro-F1/AUROC 1.0, reload parity 0.0. No Colab run, no BYOD run and no experiment run is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), Python 3.12, torch 2.14.0+cpu, transformers 4.57.6, huggingface-hub 0.36.2, safetensors 0.8.0, numpy 2.5.3 — the notebook's pins (CPU wheel), taken read-only from another pipeline repository's `.venv`. Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook. The restart behaviour therefore rests on the documented run only.
  - **Snapshot pre-seeded:** the six manifest files were copied from a local verified snapshot into the working-directory `weights/`; `stage_missing_files` fetched nothing and `verify_snapshot` re-hashed all six. Hub staging is covered by the Kaggle record only.
  - **Probes:** P0 static; P1 every code cell at defaults; P2 stronger trivial baselines on the same split; P3 the documented optional experiments (`TRAINABLE_LAYERS = 0` with a partial and a full rerun, `EPOCHS = 1`); P4 BYOD through a `google.colab` upload shim: one 30-record 3-class JSONL dataset and ten incompatible or cancelled inputs.
- **Not verified:** a Colab run; the real Colab upload dialog; GPU timing; Hub download in this environment.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** strong on the default path (P1 13/13, matching the hosted record; snapshot digest-checked; `adapt` rebuilds from the verified base on every call, so experiments do not stack; reload parity asserted). Defects: the install pattern forces a restart (ESM-M1); the BYOD split/validation contract is internally inconsistent (ESM-M3).
- **Scientific validity:** the split, baselines-on-train-only and no-selection-on-validation are correct. The comparison is not: the only order-blind baseline uses the one feature the generator equalised, and the only order-aware alternative is never measured, so the notebook draws a conclusion about what ESM-2 contributes that the experiment cannot support (ESM-M2).
- **Promise fulfilment:** default promises met. BYOD does not reach adaptation for small compatible datasets (ESM-M3); the optional experiments lack rerun instructions (ESM-m1).
- **Learner experience:** accurate, candid prose and useful "Look for" notes after each stage, but the guided layer is missing (ESM-M4).
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6, REL2 (ESM-M1); DAT12, DAT14, DAT19 (ESM-M3); REL12, since no BYOD positive/negative evidence is recorded (ESM-m3). SHOULD deviations: EVAL10 meaningful baseline (ESM-M2); GDL1–GDL7, GDL9–GDL14, UX8 (ESM-M4); GDL10, UX5 (ESM-m1); EXE2, UX10 (ESM-m2).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle: restart after install, then 13/13 | Section 1 says the cell "stops with a restart instruction" | **Not met** (ESM-M1) |
| Digest-verified pinned snapshot | cell 11 | 6/6 files verified, `fetched: []` (pre-seeded here) | clear | Met |
| Validated dataset, ceilings and digest | cell 13 | 96 records, 48/48, ceilings printed, digest `e0f9803e…` | clear | Met |
| Leakage-aware stratified split | cell 13 | 56/20/20; independence assumption and homology caveat stated | clear | Met |
| Embeddings as representations | cell 15 | 640-d, within 0.961 vs between 0.943 cosine | "inspection only" | Met |
| Baselines that set the floor | cell 17 | majority 0.50; hydrophobic fraction 0.45 | "a fine-tuned model that clears both has learned something about residue *order*" | Met as code; **interpretation not supported** (ESM-M2) |
| Bounded fine-tune, recorded trainable set | cell 19 | 10.26 M / 148.1 M; 4 epochs; val 0.85 → 1.0 | loss framed as optimisation evidence | Met |
| Independent test evaluation | cell 21 | test 1.0 / 1.0 / 1.0, n=20, report JSON written | "tutorial metrics… single holdout" | Met |
| "The model is using residue order, exactly what a protein language model is pretrained to represent" | closing cell | P2: `longest_hydrophobic_run >= 18` test 1.0; 20-residue composition centroid 0.75; C+H+Y fraction 0.80 | none | **Not met** (ESM-M2) |
| Argmax inference on new sequences | cell 23 | 6/6 match (seed 7), scores 0.57–0.85 | "not calibrated probabilities" | Met |
| Adapter export and fresh reload | cell 25 | parity 0.0, labels equal | loading vs reproducing distinguished | Met |
| BYOD through the same stages | cell 13 → 27 | P4: 30-record 3-class file validated and split 18/6/6, then cell 19 `ValueError: dataset has 6 records; at least 12 are required` | contract says ≥12 records, ≥3/class | **Not met** (ESM-M3) |
| Optional experiments | closing cell | P3: head-only test 0.90 / AUROC 1.0; `EPOCHS = 1` test 0.80 / AUROC 1.0 | no rerun list, no prediction or explanation step | Partly met (ESM-m1) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install, inspect the carried modules, stage and verify the snapshot | run cells | printed identity and verified-file count |
| Validate a labelled dataset and split it without leakage | run cell 13 | printed manifest and split sizes |
| Extract and inspect embeddings | run cell 15 | cosine summary |
| Measure baselines; run a bounded fine-tune; evaluate on an independent split | run cells 17–21 | printed metrics; no prompt asks the learner to interpret them |
| Classify new sequences; export and verify the adapter | run cells 23–25 | parity line |

All objectives are procedural ("install", "inspect", "run", "export"). No objective asks the learner to predict, compare, explain or diagnose, and no activity checks that an objective was exercised beyond running a cell.

## 4. Journeys

| Journey | Evidence basis | Outcome |
|---|---|---|
| First-time learner | Source inspection | Clear stage-by-stage prose with "Look for" notes; no audience statement, roadmap, glossary (AUROC, macro-F1, mean pooling, encoder layer, safetensors are assumed), prediction prompt, checkpoint or troubleshooting; 49 k characters of carried code at the top with no infrastructure label (ESM-M4). The closing interpretation teaches a conclusion the experiment does not support (ESM-M2). |
| Clean default | Documented (Kaggle T4, reviewed blob) + direct CPU | Kaggle: one restart after install (ESM-M1), then 13/13. CPU (install skipped, snapshot pre-seeded): 13/13 in 65.0 s, every reported metric identical to the record. |
| Active learning | Direct CPU | `TRAINABLE_LAYERS = 0`: a full rerun of cells 19–27 trains 411,522 parameters, test 0.90 / AUROC 1.0, reload parity 0.0 — a valid comparison. Rerunning only cells 19 and 21 updates `evaluation_report.json` (`trainable_layers: 0`) but leaves `result.json` and the adapter manifest at `trainable_layers: 2`, and overwrites the two-layer metrics in the namespace (ESM-m1). `EPOCHS = 1`: test accuracy 0.80, AUROC 1.0 on CPU — the "accuracy near 0.5" hint was not observed. |
| Reuse and recovery | Direct CPU via upload shim | Positive: a compatible 30-record 3-class JSONL file passed `validate_dataset` and split 18/6/6, then failed at cell 19 (ESM-M3). Continuing past the error, cell 21 raised a raw `KeyError: 'charged'` and cells 23–27 classified BYOD records with the previous binary adapter and exported it with `data_source: BYOD (my_proteins.jsonl)`. Negatives: lowercase, gap character, missing `label` column, one class, thin class, duplicate sequence, 1,100-residue sequence and `.fasta` were refused with messages naming the rule; a UTF-16 CSV raised a raw `UnicodeDecodeError`; a cancelled upload raised a bare `StopIteration` (ESM-m2). The real Colab dialog was not exercised. |

## 5. Findings

### Major

#### ESM-M1 — `Run all` needs a manual restart after the in-kernel install; recorded as PASS

- **Cell/section:** cell 3 (Section 1, "Install the pinned runtime"); `docs/release-verification.md`; `README.md`, `STATUS.md`, `tutorials/README.md` release status.
- **Observed issue:** cell 3 `pip install`s exact pins (`torch==2.14.0`, `numpy==2.5.3`, …) into the running kernel and raises `RuntimeError('… Restart the runtime, then rerun from the top.')` when a loaded distribution changed. The recorded hosted run of the reviewed blob passed only "after the expected fresh-process restart following dependency installation", and the record calls that PASS. The opening cell promises the default path needs no intervention.
- **Consequence:** a learner's first `Run all` stops in Section 1; the repository is marked Release-grade against a run that did not meet the one-pass contract.
- **Evidence:** documented execution evidence (Kaggle T4 record of blob `638ec73e`, "One expected fresh-process restart followed the install cell"); source inspection of cell 3 (`tools/build_notebook.py` lines 55–70 emit it). Direct execution skipped the install, so it neither confirms nor refutes the restart.
- **Recommended correction:** replace the in-kernel install with the fleet's uv isolated-environment pattern: an infrastructure cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that environment so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (origin/main). Change it in `tools/build_notebook.py`, regenerate, and re-record a hosted run; until then set the status to `Candidate` and record the restart as a failure of RUN10.
- **Acceptance check:** a fresh Colab (or Kaggle) runtime, one `Run all`, completes all code cells with no error output and no restart; `docs/release-verification.md` records that run for the new blob and no longer lists a restart under a PASS.
- **Spec:** RUN1, RUN10, ENV6, REL2 (MUST).

#### ESM-M2 — The composition-vs-order comparison does not support the notebook's conclusion about ESM-2

- **Cell/section:** opening cell ("A residue-counting baseline cannot separate the classes; a model that reads the sequence can — so the comparison shows what fine-tuning adds"); Section 6 ("the two synthetic classes share their composition by construction"); closing cell ("sequences that share their residue composition … the model is using residue order, exactly what a protein language model is pretrained to represent").
- **Observed issue:** the generator equalises only the hydrophobic count. `segment` records draw their background from UniProt-weighted frequencies (`samples._draw_background`) and replace hydrophobics with uniform polar residues; `scattered` records draw every polar residue uniformly. So the classes differ in composition, and the shipped composition baseline measures the single feature that was equalised. The classes are also defined by a hard rule: every `segment` has a hydrophobic run of 18–22 and every `scattered` a run of at most 5 (`_MAX_SCATTERED_RUN`), so a one-line order rule separates them perfectly.
- **Consequence:** the perfect fine-tuned score is equalled by a trivial threshold the notebook already prints, and order-blind features recover much of the signal. The learner is taught that the result demonstrates what a pretrained protein language model contributes; the experiment cannot show that.
- **Evidence:** direct execution (P2, same 56/20 train/test split): `longest_hydrophobic_run >= 18` fitted on train → train 1.0, test **1.0**; 20-residue composition nearest centroid → test **0.75**; C+H+Y fraction threshold → train 0.80, test **0.80** (train class means 0.125 segment vs 0.178 scattered); longest run over all 96 records: segment 18–22, scattered 1–5. The shipped hydrophobic-fraction baseline: test 0.45.
- **Recommended correction:** in `src/esm2_protein_pipeline/samples.py` draw the `scattered` polar residues from the same weighted background as `segment`, so the composition claim becomes true, and add a 20-residue composition baseline to show it. Add the `longest_hydrophobic_run` threshold as an order-aware trivial baseline in `metrics.py` and report it in Section 6. Then reword the opening and closing cells (`tools/notebook_template.py` lines 85–86, 191, 384–395): the experiment shows the adaptation contract can learn a residue-order rule; whether pretraining helps would need a comparison with an untrained or from-scratch encoder, which this notebook does not run.
- **Acceptance check:** on the default split, an order-blind composition baseline over all 20 residues scores within ±0.10 of 0.5 test accuracy; the order-aware heuristic is reported next to the model; no learner-facing sentence attributes the result to pretraining unless a pretrained-vs-untrained comparison is executed.
- **Spec:** EVAL10 (SHOULD); framework dimension 3 (unsupported conclusion); GDL14.

#### ESM-M3 — BYOD fails at adaptation for datasets the stated contract accepts

- **Cell/section:** Prerequisites and Section 4 ("at least 12 records and 3 per class, 2..20 classes"); cells 13, 19, 21; `samples.split_dataset`, `pipeline.adapt`, `pipeline.evaluate`.
- **Observed issue:** `adapt` and `evaluate` call `validate_dataset` on the validation and test splits with the default `min_records=12`. With the default 20 % / 20 % fractions, a binary dataset needs at least 56 records (28 per class) before either split reaches 12, and every class needs ≥3 records in each split. The notebook advertises a minimum of 12 records and 3 per class.
- **Consequence:** a learner whose file passes validation in cell 13 hits `ValueError: dataset has 6 records; at least 12 are required` in cell 19, an error that names neither the split nor the size actually needed. A learner who continues runs cells 21–27 with the previous sample adapter: cell 21 raises a raw `KeyError`, cell 23 labels BYOD records with the old classes, and cell 25 exports that adapter with `metadata.data_source = "BYOD (my_proteins.jsonl)"` and the BYOD digest.
- **Evidence:** direct execution (P4, 30 records, 3 classes × 10, JSONL): split 18/6/6, cell 19 `ValueError`, cell 21 `KeyError: 'charged'`, exported adapter classes `['scattered', 'segment']`, `train_records: 56`, metadata `data_source: BYOD (my_proteins.jsonl)`. Static probe: `generate_sample_dataset(size=n)` reaches split re-validation first at n = 56 (n = 12 and 54 fail).
- **Recommended correction:** make one contract: either state and check the effective minimum before splitting (per split ≥ the evaluation minimum, per class per split ≥ 1 or 3), raising a message that names the split and the number of records needed, or validate split membership with split-appropriate minimums. Clear `pipe`'s adapted state (or stop) when cell 13 switches source, so later cells cannot export a stale adapter under a new data label.
- **Acceptance check:** a 3-class, 30-record BYOD file either completes cells 13–27 through adaptation, evaluation, export and reload, or is refused in cell 13 with a message naming the required size; after any BYOD failure no exported file carries the BYOD data source with a sample-trained adapter.
- **Spec:** DAT12, DAT14, DAT19 (MUST); UX10.

#### ESM-M4 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** whole notebook; opening cell; cells 4–9.
- **Observed issue:** no intended learner or how-to-use instructions (GDL1–GDL2), no roadmap (GDL3), no explicit Input → Model → Output contract (GDL4; implied in prose), procedural objectives only (GDL5), no glossary for AUROC, macro-F1, mean pooling, encoder layer, adapter or safetensors (GDL6), no prediction before the baseline or evaluation results (GDL7), no checkpoints or sample answers (GDL9), no Predict → Change → Run → Observe → Explain activity (GDL10; see ESM-m1), carried module cells of 28,400 + 13,569 + 7,170 characters with no **Infrastructure** label or collapse (GDL11–GDL12), no troubleshooting section (GDL13), no conclusion scaffold (GDL14), no section synthesis (UX8). "Look for" notes after each stage partly meet GDL8.
- **Consequence:** a self-paced learner new to protein classification meets 49 k characters of library code before any model runs, is never asked to reason about a result, and gets no help when the install, the 595 MB download or BYOD fails.
- **Evidence:** source inspection; P0 term scan (no "how to use", "roadmap", "glossary", "check your", "troubleshoot", "infrastructure"; no `cellView: form` cells).
- **Recommended correction:** add the guided layer in `tools/notebook_template.py` / `tools/build_notebook.py`, following the reference notebook named in NOTEBOOK_SPEC §25.13 (`prithvi-flood-segmentation-pipeline/tutorials/DIMER_Philippines_Flood_Mapping_Capstone.ipynb`): audience and how-to-use, roadmap, glossary, a prediction before Sections 6 and 8, collapsible "Check your reasoning" answers, a structured experiment, `# @title Infrastructure: …` with `cellView: form` on the carried cells, troubleshooting, and a conclusion template.
- **Acceptance check:** each of GDL1–GDL14 can be pointed to a cell; the carried cells are titled `Infrastructure` and collapsed; at least two principal results are preceded by a prediction prompt and followed by a sample answer.
- **Spec:** GDL1–GDL7, GDL9–GDL14, UX8 (SHOULD).

### Minor

#### ESM-m1 — Optional experiments have no rerun list or prediction/explanation step

- **Cell/section:** closing cell, "Optional experiments".
- **Observed issue:** the text says "set `TRAINABLE_LAYERS = 0` … and compare the test metrics with the two-layer run" but names no cells to rerun and keeps no copy of the two-layer metrics; rerunning cell 19 and 21 overwrites `test_metrics`. A partial rerun leaves mixed exports. The `EPOCHS = 1` hint ("AUROC may still be high while accuracy sits near 0.5") did not match the CPU run (accuracy 0.80).
- **Consequence:** the comparison depends on the learner copying numbers by hand, and exports can describe two different models.
- **Evidence:** direct execution (P3a: after cells 19 and 21 only, `evaluation_report.json` `trainable_layers: 0`, `result.json` and adapter manifest `trainable_layers: 2`; P3b full rerun of 19–27: head-only test 0.90 / AUROC 1.0; P3c `EPOCHS = 1`: test 0.80 / AUROC 1.0).
- **Recommended correction:** turn one experiment into a structured activity: store the default metrics before the change, list "rerun cells 19–27", ask for a prediction first, print a side-by-side table, and ask for an explanation. Phrase the `EPOCHS = 1` expectation without a fixed outcome.
- **Acceptance check:** following the written instructions exactly produces a two-row comparison table and leaves all exported files describing the same model.
- **Spec:** GDL10, UX5 (SHOULD).

#### ESM-m2 — BYOD input path is Colab-only and two failure modes are raw exceptions

- **Cell/section:** cell 13.
- **Observed issue:** BYOD always imports `google.colab` and opens the upload dialog; there is no location field, although the Prerequisites list Jupyter as supported. A cancelled or empty upload raises a bare `StopIteration`; a UTF-16 CSV (a common spreadsheet export) raises a raw `UnicodeDecodeError`. BYOD records are written to `outputs/esm2_protein_sample_dataset.csv`.
- **Consequence:** Jupyter users and executors cannot reach BYOD; two plausible mistakes give no corrective action; the "sample" file name misdescribes user data.
- **Evidence:** source inspection; direct execution (P4 negatives: 8 of 10 incompatible inputs gave messages naming the rule; UTF-16 and cancelled upload did not).
- **Recommended correction:** add `BYOD_PATH = ''  # @param {type:"string"}` that, when set, reads the file without importing `google.colab`; catch an empty upload and a decode error with messages that name the fix ("no file selected", "save as UTF-8 CSV"); write BYOD data under a BYOD-specific name.
- **Acceptance check:** with `BYOD_PATH` set, cell 13 loads the file outside Colab; an empty upload and a UTF-16 file each produce a one-line actionable error.
- **Spec:** EXE2 (SHOULD), DAT19 (MUST, for the decode case), UX10.

#### ESM-m3 — Release record has no BYOD evidence

- **Cell/section:** `docs/release-verification.md`.
- **Observed issue:** the Release-grade promotion records only the default path. No representative BYOD input, no rejected input and no downstream BYOD stages are recorded.
- **Consequence:** REL12 is unmet, and ESM-M3 went undetected at promotion.
- **Evidence:** source inspection of the release record.
- **Recommended correction:** after ESM-M3 is fixed, record one positive BYOD run through reload and one clear rejection, with commit, blob and runtime.
- **Acceptance check:** the release record has a BYOD row for the release blob showing positive and negative outcomes.
- **Spec:** REL12 (MUST).

### Suggestions

- **ESM-S1 —** Regenerate against NOTEBOOK_SPEC 2.2 (the notebook, `tutorials/README.md` and `docs/release-verification.md` declare 2.0).
- **ESM-S2 —** Add a pretrained-vs-randomly-initialised encoder run as an optional experiment; it is the comparison that would let the notebook say what pretraining contributes.
- **ESM-S3 —** In the closing transfer prompt, name a concrete homology-aware split tool (sequence-identity clustering before splitting) so the stated limitation becomes an action.

## 6. Readiness

**Needs revision.** Open Majors: ESM-M1 (RUN1/RUN10 MUST), ESM-M2, ESM-M3 (DAT12/DAT14/DAT19 MUST), ESM-M4. REL12 is also unmet (ESM-m3). Remaining gates after fixes: a one-pass hosted `Run all` of the new blob, and recorded BYOD positive and negative runs.

## 7. Verified versus inferred

- **Verified by direct CPU execution:** the default path (13/13, metrics identical to the hosted record), the stronger baselines, both optional experiments, the BYOD size failure and stale export, ten BYOD rejections.
- **Verified from documented evidence:** the hosted restart and 13/13 T4 run of the reviewed blob.
- **Inferred, not verified:** that a Colab `Run all` also needs a restart (the guard fires whenever a pinned distribution already loaded in the kernel changes; Colab preloads several); the real upload dialog behaviour.
- **Most likely to be wrong:** ESM-M2's severity. The notebook's caveats ("a perfect score here says the adaptation contract works") limit the damage, so it could be argued Minor; it is graded Major because the opening and closing cells state as fact a composition equality that is false and attribute the result to pretraining.

*Review only; no fixes are included. Probe ZIP: `esm2_protein_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).*
