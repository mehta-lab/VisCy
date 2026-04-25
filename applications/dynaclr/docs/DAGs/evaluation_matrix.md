# Evaluation Matrix DAG

**Status:** proposal (2026-04-24)
**Companion:** `evaluation.md` (per-run pipeline DAG)
**Goal:** run the same evaluation pipeline across **4 models × 4 datasets** with minimal config duplication, then join results via `compare_evals.py` for the NMI paper (Fig 2 smoothness, Fig 3 displacement, Table 1 classification). Infectomics splits into `infectomics-annotated` (trains linear classifiers) and `infectomics-unannotated` (consumes them) so trained LCs transfer to unlabeled datasets.

---

## 1. Matrix

|                        | **infectomics-annotated** (trains LC) | **infectomics-unannotated** (applies LC) | **alfi** (applies LC)   | **microglia** (applies LC) |
| ---------------------- | -------------------------------------- | ----------------------------------------- | ----------------------- | -------------------------- |
| **DynaCLR-2D-MIP-BagOfChannels** | ✅ exists (`v1.yaml`)                    | ⬜ create                                  | ✅ exists (`alfi-eval.yaml`) | ✅ exists (`microglia-eval.yaml`) |
| **DynaCLR-classical**  | ⬜ create                                | ⬜ create                                  | ⬜ create                | ⬜ create                  |
| **DINOv3-temporal-MLP-2D-BagOfChannels-v1**         | ✅ exists (`v1.yaml`)                    | ⬜ create                                  | ⬜ create                | ⬜ create                  |
| **DINOv3-frozen**      | ⬜ create (needs orchestrator change)   | ⬜ create                                  | ⬜ create                | ⬜ create                  |

**16 leaf configs total.** LC training happens only in the `infectomics-annotated` column; all other columns apply those pipelines.

---

## 2. Directory layout (target)

**Trained LCs live centrally**, not inside per-eval output dirs. One registry per model:

```
/hpc/projects/organelle_phenotyping/models/linear_classifiers/
├── DynaCLR-2D-MIP-BagOfChannels/
│   ├── manifest.json                        # {task, marker_filter, pipeline_path, trained_on, trained_at}
│   ├── infection_state_G3BP1.joblib
│   ├── infection_state_SEC61B.joblib
│   ├── infection_state_Phase3D.joblib
│   ├── organelle_state_G3BP1.joblib
│   └── ...
├── DynaCLR-classical/   { same layout }
├── DINOv3-temporal-MLP-2D-BagOfChannels-v1/          { same layout }
└── DINOv3-frozen/       { same layout }
```

This lets any eval run (new dataset, different timepoint split, etc.) fetch a specific (task, marker) classifier without re-running infectomics-annotated. Rebuilds are explicit.

```
applications/dynaclr/configs/evaluation/
├── recipes/
│   ├── predict.yml                          # existing — default predict settings
│   ├── predict_dinov3_frozen.yml            # NEW — HF-loaded, no ckpt_path
│   ├── reduce.yml                           # existing
│   ├── mmd_defaults.yml                     # existing
│   ├── plot_infectomics.yml                 # existing
│   ├── linear_classifiers_infectomics.yml   # existing → fold into infectomics-annotated.yml
│   ├── infectomics-annotated.yml            # NEW — trains LC, publishes to central registry
│   ├── infectomics-unannotated.yml          # NEW — fetches LC from central registry
│   ├── alfi.yml                             # NEW — fetches LC from central registry
│   └── microglia.yml                        # NEW — fetches LC from central registry
│
├── DynaCLR-2D-MIP-BagOfChannels/
│   ├── infectomics-annotated.yaml           # trains LC → /models/linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/
│   ├── infectomics-unannotated.yaml         # fetches from same central dir
│   ├── alfi.yaml
│   └── microglia.yaml
├── DynaCLR-classical/     { same 4 leaves }
├── DINOv3-temporal-MLP-2D-BagOfChannels-v1/            { same 4 leaves }
├── DINOv3-frozen/         { same 4 leaves }
│
├── eval_registry.yaml                       # 16 eval_dirs for compare_evals.py
└── run_all_evals.sh                         # submits 16 Nextflow runs (2 waves for LC dependency)
```

By convention each Wave-1 leaf writes to (and each Wave-2 leaf reads from):
```
/hpc/projects/organelle_phenotyping/models/linear_classifiers/{model_name}/
```

No `output_dir` lookup, no cross-path stitching — every leaf simply points at the per-model registry directory.

---

## 3. What lives where

### Shared recipe — `recipes/{dataset}.yml`
Written once, reused by all 4 models on that dataset. Captures anything dataset-specific that the model doesn't care about:
- `cell_index_path` — parquet path
- `steps` — pipeline step list (microglia omits `linear_classifiers` / `append_*`)
- `linear_classifiers.annotations` — annotation CSV paths
- `linear_classifiers.tasks` — tasks + marker_filters
- `reduce_dimensionality`, `reduce_combined`, `smoothness`, `plot` defaults

### Per-run leaf — `{Model}/{dataset}.yaml`
Only model-specific fields:
- `base:` — list of recipes to merge
- `training_config` — model arch YAML
- `ckpt_path` — weights (omitted for DINOv3-frozen)
- `output_dir` — per-run output location
- Rare overrides (e.g., `predict.precision`)

**Rule of thumb:** if a field would be identical across all 4 models for the same dataset, it belongs in `recipes/{dataset}.yml`. If it varies by model (even one model), it stays in the leaf.

---

## 4. DAG (matrix layer with central LC registry)

For each model, the 4 columns run in two waves:
- **Wave 1** — `infectomics-annotated` trains LC pipelines and **publishes** them to `/hpc/projects/organelle_phenotyping/models/linear_classifiers/{model}/`.
- **Wave 2** — `infectomics-unannotated`, `alfi`, `microglia` run in parallel. Each **fetches** pipelines from the same central registry via `append_predictions.pipelines_dir` and produces predictions without retraining.

```
LC_REGISTRY = /hpc/projects/organelle_phenotyping/models/linear_classifiers/

 ┌──────────────┐
 │ DynaCLR-     │──► WAVE 1: infectomics-annotated
 │ 2D-MIP-BoC   │        └─► publish to LC_REGISTRY/DynaCLR-2D-MIP-BagOfChannels/  ◄──┐
 └──────────────┘                                                            │
                         WAVE 2 (parallel, all three fetch same registry):   │
                            ├─► infectomics-unannotated                      │
                            ├─► alfi                        ────── append_predictions
                            └─► microglia                                    │
                                                                             │
 ┌──────────────┐                                                            │
 │ DynaCLR-     │──► WAVE 1 ─► LC_REGISTRY/DynaCLR-classical/ ──► WAVE 2 ──┘
 │ classical    │
 └──────────────┘
 (DINOv3-temporal-MLP-2D-BagOfChannels-v1, DINOv3-frozen: same structure, own registry folder)

                        all 16 eval_dirs
                                │
                                ▼
                          eval_registry.yaml ──► compare_evals.py
                                                      │
                                                      ▼
                                              comparison/{overlays,summary}
```

**Key invariants:**
- Wave-2 always applies the *same-model* classifiers — cross-model LC application would mix feature spaces and is never valid.
- The registry is canonical: no copies live inside per-eval output dirs. This means any future eval on a new dataset just points at `LC_REGISTRY/{model}/` — no need to re-run infectomics-annotated. Rebuilds are explicit (delete + re-run Wave 1).

**How cross-dataset LC application works in code:**
- **Wave 1** runs `steps: [..., linear_classifiers, append_annotations, append_predictions, ...]`. The `linear_classifiers` step writes pipelines to `linear_classifiers.publish_dir` if set, else to the legacy `output_dir/linear_classifiers/pipelines/`.
- **Wave 2** runs `steps: [..., append_predictions, ...]` (no `linear_classifiers` step). `append_predictions.pipelines_dir` points at `LC_REGISTRY/{model}/`. `append_predictions.py` loops over the manifest and applies each pipeline to cells whose marker matches — it does not retrain. Markers absent from the registry manifest (e.g. microglia's Brightfield, Retardance) get no prediction; this is expected and logged.

Orchestrator changes needed (§5.5): `run_linear_classifiers` must accept a `publish_dir` output target, and `_generate_append_predictions_yaml` must accept an explicit `pipelines_dir` input instead of hardcoding `output_dir/linear_classifiers/pipelines`.

---

## 5. Infrastructure work required

### 5.1 Orchestrator — support DINOv3-frozen (no `ckpt_path`)

`dynaclr prepare-eval-configs` currently requires `ckpt_path`. DINOv3-frozen loads weights from HuggingFace at model init time — there's no local checkpoint.

**Change needed** (scope: ~30 lines in `prepare_eval_configs.py`):
- Allow `ckpt_path: null` when the generated predict recipe omits `ckpt_path` in its `trainer`/top-level section.
- Add a `model_override` field to the eval config schema: if present, swap the model class path in the generated `predict.yml`. (The DynaCLR predict recipe assumes `ContrastiveModule` with a ConvNeXt encoder; DINOv3-frozen uses `FoundationModule` + `DINOv3Model`.)
- Alternative: accept a full `predict_config` path that the orchestrator copies verbatim into `configs/predict.yml`, skipping its own generation. Cleaner — recommended.

### 5.2 Recipe — `recipes/predict_dinov3_frozen.yml`

Based on `configs/prediction/dinov3_predict.yml`. Strips `data_path`/`tracks_path`/`z_range` (orchestrator fills those from `cell_index_path`) and adds eval-specific callback settings.

### 5.3 Matrix runner — `run_all_evals.sh` (two waves)

```bash
#!/bin/bash
set -euo pipefail

EVAL_DIR=applications/dynaclr/configs/evaluation
MODELS=(DynaCLR-2D-MIP-BagOfChannels DynaCLR-classical DINOv3-temporal-MLP-2D-BagOfChannels-v1 DINOv3-frozen)

run_nf () {
  local cfg=$1
  nextflow run applications/dynaclr/nextflow/main.nf \
    --eval_config "${EVAL_DIR}/${cfg}" \
    --workspace_dir "$PWD" \
    -work-dir "work/$(dirname $cfg)/$(basename $cfg .yaml)" \
    -resume
}

# Wave 1: train LCs on infectomics-annotated (parallel across models)
for m in "${MODELS[@]}"; do
  run_nf "$m/infectomics-annotated.yaml" &
done
wait

# Wave 2: apply LCs to the other 3 datasets (parallel across 4 models × 3 datasets = 12 jobs)
for m in "${MODELS[@]}"; do
  for d in infectomics-unannotated alfi microglia; do
    run_nf "$m/${d}.yaml" &
  done
done
wait
```

Wave 1 must finish per model before Wave 2 for that model runs (pipelines must exist). The `wait` between waves is the simplest barrier. Each leaf uses its own `-work-dir` to avoid collisions.

### 5.4 `eval_registry.yaml` — for `compare_evals.py`

```yaml
models:
  - name: DynaCLR-2D-MIP-BagOfChannels-infectomics-annotated
    eval_dir: /hpc/.../DynaCLR-2D-MIP-BagOfChannels/evaluations/infectomics-annotated/
  - name: DynaCLR-2D-MIP-BagOfChannels-infectomics-unannotated
    eval_dir: /hpc/.../DynaCLR-2D-MIP-BagOfChannels/evaluations/infectomics-unannotated/
  - name: DynaCLR-2D-MIP-BagOfChannels-alfi
    eval_dir: /hpc/.../DynaCLR-2D-MIP-BagOfChannels/evaluations/alfi/
  - name: DynaCLR-2D-MIP-BagOfChannels-microglia
    eval_dir: /hpc/.../DynaCLR-2D-MIP-BagOfChannels/evaluations/microglia/
  # ... 16 entries total
output_dir: /hpc/.../comparisons/nmi_figures/
fdr_threshold: 0.05
```

`compare_evals.py` already auto-discovers CSVs per `eval_dir` — no changes needed if the output layout is consistent.

### 5.5 Orchestrator — cross-run `pipelines_dir` for `append_predictions`

**Status:** landed in commits `5a629837` (writer/reader/schema) and `56b3e696` (decoupled `append_annotations`). The text below documents the resulting design.

**(a) `run_linear_classifiers` — publish to external dir.**

`run_linear_classifiers.py` writes pipelines to `output_dir/linear_classifiers/pipelines/` by default. When `linear_classifiers.publish_dir` is set, the writer additionally promotes the trained bundle into the central registry via atomic rename + symlink swap (see §5.6):

```yaml
linear_classifiers:
  publish_dir: /hpc/projects/organelle_phenotyping/models/linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/
  # ... annotations, tasks, etc.
```

When set, pipelines + `manifest.json` are also published to that registry dir (in addition to the in-run staging copy). The manifest is intentionally minimal — lineage lives in the directory structure (model name = parent dir, version = `vN/`):

```json
{
  "trained_at": "2026-04-25T06:19:08+00:00",
  "pipelines": [
    {"task": "infection_state", "marker_filter": "G3BP1", "path": "infection_state_G3BP1.joblib"},
    ...
  ]
}
```

Older list-format manifests (just an array of pipeline dicts) are no longer supported — landed as a clean break.

**(b) `_generate_append_predictions_yaml` — fetch from explicit dir.**

`append_predictions.pipelines_dir` is now honored when set; otherwise the generator falls back to the legacy in-run path:
```python
pipelines_dir = eval_cfg.append_predictions.pipelines_dir \
    or (output_dir / "linear_classifiers" / "pipelines")
```

The guard requiring `linear_classifiers` in `steps` is relaxed: `append_predictions` is allowed standalone when `pipelines_dir` is set externally.

**(c) `append_annotations` schema decoupled from LC config.**

Wave-2 datasets like alfi carry annotation CSVs but do not train LCs, so they cannot put annotations under `linear_classifiers.annotations`. A new `AppendAnnotationsStepConfig` was added with its own `annotations: list[AnnotationSource]`. The `append_annotations` step now sources annotations from either schema (preferring the new one when both are present), and `append_annotations.py` auto-discovers task columns from the CSV when `tasks: []` is empty (Wave-2 datasets don't enumerate tasks explicitly).

Wave-2 leaves point at the model's registry root + version selector (default `latest`):
```yaml
append_predictions:
  pipelines_dir: /hpc/projects/organelle_phenotyping/models/linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/latest
```

Wave-1 leaves write into the same registry root (versioning is handled automatically — see §5.6):
```yaml
linear_classifiers:
  publish_dir: /hpc/projects/organelle_phenotyping/models/linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/
```

Because every leaf references the same central path per model, this line is a natural candidate to move into `recipes/{model}.yml` — but since we opted for dataset recipes, not model recipes, the registry path lives in the leaf. Alternative: single global `recipes/lc_registry.yml` mapping model name → registry path, merged in via the leaf's `base:`.

### 5.6 LC registry versioning

Each model's registry holds a series of versioned bundles plus a `latest` symlink that points at the most recent one:

```
linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/
├── latest -> v3                # symlink (relative target)
├── v1/
│   ├── manifest.json
│   ├── infection_state_G3BP1.joblib
│   └── ...
├── v2/
└── v3/
```

#### Publishing (`run_linear_classifiers` writer)

The training step writes atomically: stage everything in a temp dir, rename into `vN`, then swap the symlink. Sketch:

```python
def publish_pipelines(publish_dir: Path, trained: list[tuple[str, str, Pipeline]],
                      manifest: dict) -> Path:
    """Atomically publish a new versioned LC bundle and update the latest symlink."""
    publish_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pick next version number
    existing = sorted(int(p.name[1:]) for p in publish_dir.glob("v*")
                      if p.is_dir() and p.name[1:].isdigit())
    next_v = (max(existing) + 1) if existing else 1
    new_dir = publish_dir / f"v{next_v}"

    # 2. Write to staging dir (never directly to vN — partial writes never observable)
    staging = Path(tempfile.mkdtemp(prefix=f"v{next_v}.", dir=publish_dir))
    for task, marker, pipeline in trained:
        joblib.dump(pipeline, staging / f"{task}_{marker}.joblib")
    manifest["version"] = next_v
    manifest["trained_at"] = datetime.now(UTC).isoformat()
    with open(staging / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 3. Atomic rename: staging -> vN (POSIX guarantees atomicity on same FS)
    os.rename(staging, new_dir)

    # 4. Atomic symlink swap: write latest.new, then rename over latest
    latest = publish_dir / "latest"
    latest_new = publish_dir / "latest.new"
    if latest_new.is_symlink() or latest_new.exists():
        latest_new.unlink()
    os.symlink(new_dir.name, latest_new)   # relative target ("v3"), not absolute
    os.replace(latest_new, latest)         # atomic over existing symlink

    return new_dir
```

Three guarantees:
- **Staging dir** — if the job crashes mid-write, `vN/` never appears in a half-written state.
- **`os.rename(staging, vN)`** — POSIX guarantees atomicity on the same filesystem.
- **`os.replace(latest.new, latest)`** — atomic symlink swap. A reader doing `readlink("latest")` at any instant sees either the old target or the new one, never a dangling link.

Concurrent training of the same model is guarded with `fcntl.flock` on `publish_dir/.lock` around steps 1–4. Unlikely in practice (Wave 1 per model runs once per campaign), but cheap.

#### Reading (`append_predictions` reader)

Resolve the symlink **once at startup** so the entire run is consistent — even if a new version is published mid-run, this run sticks with the version it saw:

```python
def load_pipelines(pipelines_dir: Path) -> tuple[Path, dict, list[Pipeline]]:
    resolved = pipelines_dir.resolve()           # follow latest -> vN
    version_tag = resolved.name                   # "v3"
    manifest = json.loads((resolved / "manifest.json").read_text())
    manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]

    pipelines = []
    for entry in manifest["pipelines"]:
        pipelines.append(joblib.load(resolved / entry["pipeline_path"]))

    click.echo(f"LC registry: {pipelines_dir} -> {resolved} ({version_tag}, manifest_sha={manifest_sha})")
    return resolved, manifest, pipelines
```

#### Manifest schema

```json
{
  "version": 3,
  "trained_at": "2026-04-24T15:33:21+00:00",
  "feature_space": "DynaCLR-2D-MIP-BagOfChannels",
  "embedding_ckpt_path": "/hpc/.../DynaCLR-2D-MIP-BagOfChannels/.../last.ckpt",
  "embedding_ckpt_sha256": "ab12cd...",
  "training_config_git_sha": "742b426",
  "annotation_csv_shas": {
    "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV": "ef34..."
  },
  "pipelines": [
    {"task": "infection_state", "marker_filter": "G3BP1", "pipeline_path": "infection_state_G3BP1.joblib"},
    ...
  ]
}
```

Wave-2 then asserts `manifest["feature_space"] == eval_cfg.model.name` and fails loudly if not — guard against accidentally pointing DINOv3-temporal-MLP-2D-BagOfChannels-v1 at the DynaCLR-2D-MIP-BagOfChannels registry.

#### Lineage in the output zarr

Each `predicted_{task}__{model}` column gets a sibling `.uns` entry so any figure regenerated from the zarr has a direct back-reference to the exact bundle used:

```python
adata.uns[f"predicted_{task}__{model}_lc_version"] = version_tag      # "v3"
adata.uns[f"predicted_{task}__{model}_lc_manifest_sha"] = manifest_sha
adata.uns[f"predicted_{task}__{model}_lc_path"] = str(resolved)
```

#### Pinning vs. `latest`

Active development uses `latest`:
```yaml
pipelines_dir: /hpc/.../linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/latest
```

Paper rerun scripts pin an explicit version at submission time:
```yaml
pipelines_dir: /hpc/.../linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/v2   # NMI submission
```

#### Inspection CLI

Small helper so we don't have to `cat` JSON:

```bash
$ dynaclr list-lc-versions DynaCLR-2D-MIP-BagOfChannels
v1  2026-03-12  feature_space=DynaCLR-2D-MIP-BagOfChannels  tasks=4  pipelines=12
v2  2026-04-01  feature_space=DynaCLR-2D-MIP-BagOfChannels  tasks=4  pipelines=12
v3  2026-04-24  feature_space=DynaCLR-2D-MIP-BagOfChannels  tasks=4  pipelines=12   <- latest
```

#### Garbage collection

Keep all versions until disk pressure warrants pruning (each bundle is <100MB). Add `dynaclr gc-lc {model} --keep-last 5` later if needed. Future: a `--tag` flag on training (`--tag nmi-submission`) marks a version as protected from GC.

#### NFS caveat

Our `/hpc/projects/...` is NFSv4-backed; symlinks and atomic `os.replace` work there. If a future filesystem doesn't support symlinks, fall back to writing a `LATEST` text file containing the version name (`"v3"`) and resolve via that.

---

## 6. Implementation order

| # | Task | Status | Commit |
|---|------|--------|--------|
| 1 | Extract dataset recipes (`recipes/{infectomics-annotated,alfi,microglia}.yml`) | ✅ done | `00d709d3` |
| 2 | Move existing leaves into per-model folders + update `base:` lists | ✅ done | `45c3d93b` |
| 3 | Orchestrator: `publish_dir` writer + external `pipelines_dir` reader (§5.5 a, b) | ✅ done | `5a629837` |
| 3b | Decouple `append_annotations` schema from LC config (§5.5 c) | ✅ done | `56b3e696` |
| 4 | Wave-1 SLURM submission script | ✅ done | `cae868cc` |
| 5 | Run Wave-1 to validate writer end-to-end (publish `v1/` + symlink) | ✅ done — see §9 |
| 6 | Tune nextflow.config retry strategy + lower PHATE subsample (REDUCE_COMBINED OOM mitigation) | ✅ done — see §9 |
| 7 | Run Wave-2 (alfi) to validate reader against `latest` | ⬜ pending |
| 8 | Create `infectomics-unannotated` leaves (need new cell_index parquet) | ⬜ pending |
| 9 | Create DynaCLR-classical row (4 leaves + checkpoint path) | ⬜ pending |
| 10 | Add DINOv3-temporal-MLP-2D-BagOfChannels-v1 alfi + microglia + infectomics-unannotated leaves | ⬜ pending |
| 11 | Orchestrator: frozen-inference passthrough (§5.1). Blocker for DINOv3-frozen. | ⬜ pending |
| 12 | DINOv3-frozen row (4 leaves + `recipes/predict_dinov3_frozen.yml`) | ⬜ pending |
| 13 | `run_all_evals.sh` two-wave runner + `eval_registry.yaml` | ⬜ pending |
| 14 | Output column namespacing (`predicted_{task}__{model}`) — follow-up | ⬜ deferred |
| 15 | Lineage-aware PHATE subsampling (whole-track sampling via `(fov_name, track_id)`) | ✅ done |

---

## 7. Resolved decisions

- **Model naming convention** — training-config stem (e.g. `DynaCLR-2D-MIP-BagOfChannels`, `DynaCLR-2D-BagOfChannels-v3`, `DINOv3-temporal-MLP-2D-BagOfChannels-v1`). Same name used for the registry directory under `linear_classifiers/`, the leaf-config folder under `configs/evaluation/`, and the implicit `feature_space` (= registry parent dir name). Avoids the `-v3`/`vN` collision because LC versions are always `vN` integers under each model dir.
- **Classifier granularity** — one pipeline per `(task, marker_filter)`. Filename `{task}_{marker}.joblib`. Wave-2 looks up cells by `marker` and applies the matching pipeline.
- **Output column namespacing** (deferred) — `append_predictions` will eventually write `predicted_{task}__{model}` so 4 models can write predictions to the same per-experiment zarr without overwriting each other. Currently writes `predicted_{task}` (single-model only); landing this is task #14.
- **LC registry versioning** — versioned directories with `latest` symlink (§5.6). Atomic publish via staging-dir + `os.rename` + `os.replace`. Reader resolves `latest` once at startup. Minimal manifest (`{trained_at, pipelines: [...]}`) — lineage encoded in directory structure (parent = model, dir = `vN`). Pin explicit `vN` for reruns; use `latest` for active development.
- **Manifest format** — clean break from the old list-of-dicts format. Reader hard-fails on legacy manifests.
- **Markers absent from manifest** — log a coverage report (`predicted N/M markers, missing: [Brightfield, Retardance]`) and continue. Cells of unmatched markers get no prediction.
- **SLURM retry strategy** — time-only escalation (`time = base * task.attempt`, max 2 retries) on exit codes 140 (SIGUSR2 from `--signal B:USR2@30`) and 137 (SIGKILL after time limit). Memory stays flat across retries because our jobs are bounded by per-experiment cell quotas, not by RAM ceilings. Generic Python crashes (exit 1) are NOT retried.
- **PHATE subsample size** — lowered from 50,000 to 20,000 in `recipes/reduce.yml` to keep REDUCE_COMBINED under the 4h `cpu_heavy` time limit. Lineage-aware subsampling (task #15) is the proper long-term fix.
- **MMD pruned from matrix runs** — the 4×4 matrix recipes don't include MMD steps. The existing `mmd_defaults.yml` recipe is left in place for one-off non-matrix runs (e.g. `DynaCLR-3D-BagOfChannels-v2.yaml`).

## 8. Remaining open questions

- **`infectomics-unannotated` cell_index** — which experiments go here? Candidates: 07_22 ZIKV OOD experiments lacking annotation CSVs, plus any other infectomics dataset we want predicted labels on. Needs a new collection YAML + parquet build before its leaf configs can resolve.
- **DynaCLR-classical checkpoint** — which run? Need path to `training_config` + `last.ckpt`.
- **DINOv3-frozen orchestrator approach** — `predict_config:` passthrough (copy a full predict YAML verbatim into `configs/predict.yml`) vs. `model_override:` field (swap the model class in the generated predict). Passthrough is simpler; override is more consistent with the existing recipe-merge pattern. Recommendation: passthrough.
- **`output_dir` convention** — standardize to `{model_root}/evaluations/{dataset_column}/` so `eval_registry.yaml` entries are predictable across the matrix. Existing runs use inconsistent names (`evaluations/alfi/` vs. `evaluation_lc_v1/`); migrate when convenient.

---

## 9. Validation log

### 2026-04-25 — first end-to-end Wave-1 run (DynaCLR-2D-MIP-BagOfChannels × infectomics-annotated)

Launched via `sbatch applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels/run_infectomics_annotated.sh` (job 31416428). Pipeline ran end-to-end through APPEND_PREDICTIONS, then failed at REDUCE_COMBINED on the 4h SLURM time limit (exit 140) during PHATE on a 50k subsample of 350k cells.

**What landed successfully:**

- ✅ PREDICT, SPLIT, REDUCE×19, SMOOTHNESS×19, APPEND_ANNOTATIONS, **LINEAR_CLASSIFIERS**, **APPEND_PREDICTIONS** all completed
- ✅ Central registry populated as designed:
  ```
  /hpc/projects/organelle_phenotyping/models/linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/
  ├── latest -> v1
  └── v1/
      ├── manifest.json    (new dict format, trained_at + pipelines list)
      └── 13× *.joblib     (4 tasks × {G3BP1, SEC61B, Phase3D, viral_sensor} or {G3BP1, SEC61B})
  ```
- ✅ Atomic publish + symlink swap verified on disk
- ✅ Manifest is the new dict format (`{"trained_at": ..., "pipelines": [...]}`)
- ✅ APPEND_PREDICTIONS ran successfully against the in-run pipelines dir (Wave 1 self-applies; Wave 2 will apply against `latest` symlink)

**What broke (and was fixed for resume):**

- ❌ REDUCE_COMBINED hit 4h `cpu_heavy` time limit during PHATE
- 🔧 Mitigation 1: `nextflow.config` — added `time = { base * task.attempt }` + retry on exit 140/137 (max 2). Memory stays flat.
- 🔧 Mitigation 2: `recipes/reduce.yml` — `phate.subsample: 20_000` (was 50,000). PHATE complexity is roughly N², so ~6× faster fit.

**Next step:** resubmit the same SLURM script with `-resume`. Nextflow will skip everything that already succeeded; only REDUCE_COMBINED + PLOT_COMBINED + per-experiment PLOT need to run. Then submit Wave-2 alfi to validate the reader against `latest`.
