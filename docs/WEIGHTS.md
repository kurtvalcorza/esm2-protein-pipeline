# Weight provenance and DIMER hosting

This repository pins **one** snapshot with its own `dimer-base-manifest.json`.

## ESM-2 150M weights

- Upstream: `facebook/esm2_t30_150M_UR50D`
- Immutable revision: `a695f6045e2e32885fa60af20c13cb35398ce30c`
- Weight format: SafeTensors (`model.safetensors`, 595,257,706 bytes)
- Upstream weight license: MIT (`license: mit` in the pinned upstream README front matter)
- Local layout: `weights/esm2-t30-150m-ur50d/` holds the 6 manifest entries (`config.json`, `model.safetensors`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`, upstream `README.md`; 595,260,503 bytes total) with byte size and SHA-256 for each. `verify_snapshot()` in `src/esm2_protein_pipeline/pipeline.py` checks all of them before any load; `stage_missing_files(allow_download=True)` fetches only absent entries at the pinned revision. `.safetensors` files are git-ignored; the Git repository does not vendor the checkpoint.
- Cross-check: the manifest's `model.safetensors` digest `c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566` equals the `oid sha256` of the Hub LFS pointer at the pinned revision (`https://huggingface.co/facebook/esm2_t30_150M_UR50D/raw/a695f6045e2e32885fa60af20c13cb35398ce30c/model.safetensors`).

## Files deliberately not staged

The upstream repository also carries `pytorch_model.bin` (595,364,077 bytes) and `tf_model.h5` (593,355,136 bytes). Neither is listed in the manifest and neither is ever fetched or loaded: SafeTensors carries the same parameters without executing a pickle, and the loader passes `use_safetensors=True`. A DIMER profile upload for this model should use `model.safetensors`.

## DIMER hosting

- MIT permits use, modification, redistribution and commercial use subject to preservation of the licence and copyright notice. DIMER may mirror the pinned snapshot in its model store under those terms; the weights are redistributed unmodified.
- Loader trust boundary: `transformers` 4.57.6 builds the native `EsmModel` / `EsmForSequenceClassification` and `EsmTokenizer` classes for this checkpoint. The pipeline passes `trust_remote_code=False` and `local_files_only=True`, so no upstream Python is executed and no file is resolved from the Hub or an HF cache on the snapshot path.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
