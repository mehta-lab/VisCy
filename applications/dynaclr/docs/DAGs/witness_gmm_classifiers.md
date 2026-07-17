# Witness-GMM annotations → linear classifiers DAG

**Written:** 2026-07-16
**Status:** Active. Replaces the earlier witness-score classifier path
(the sign+dead-zone gate + circular annotation grading, now removed).
**Rolls up to:** `.ed_planning/dynaclr/batch_correction/per_microscope_classifier/PLAN.md`

A witness→GMM label **is an annotation**: its meaning is named by the modality it
is computed from — a witness over `viral_sensor` produces `infection_state`
(infected/uninfected); over an organelle marker, `organelle_remodeling_state`
(remodel/noremodel). So Stage A writes a **named biological-state column** with the
real class vocabulary, a file indistinguishable from a hand annotation, and the
**existing annotation training path** (`run-linear-classifiers`,
`label_source: annotations`) consumes it unchanged.

## End-to-end DAG (two stages, one existing training path)

```mermaid
flowchart TD
    Z["embeddings zarr<br/>obs: experiment, marker, fov_name, id, track_id, t,<br/>perturbation, hours_post_perturbation<br/><i>one microscope / marker per config — no LOT, no pooling</i>"]

    subgraph A["STAGE A · dynaclr witness-gmm-labels (NEW)"]
        direction TB
        A1["references from wells/filters:<br/>X = control cells, Y = perturbed cells"]
        A2["bandwidth = median_heuristic(X, Y)<br/><i>viscy_utils.evaluation.mmd</i>"]
        A3["score every cell: w(z) = witness_function(z, X, Y, bw)<br/><i>mmd</i>"]
        A4["per perturbed CONDITION: 2-component GMM on w[cond]<br/><i>witness_gmm.fit_gmm_labels</i><br/>remod = argmin(means); posterior ≥ gmm_pos_threshold → positive<br/>negatives = ALL control-well cells; ambiguous → dropped<br/>unimodal GMM (separated=False) → marker skipped"]
        A5["map GMM ±1 → class_map vocabulary<br/>(e.g. infected / uninfected)"]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    LBL["<b>ANNOTATION FILE</b> <label_column>.csv|parquet<br/>key: fov_name + id (or fov_name + t + track_id) + experiment<br/>named state column, e.g. infection_state ∈ {infected, uninfected}<br/><i>hand-annotation format — producer-agnostic</i>"]

    subgraph B["STAGE B · dynaclr run-linear-classifiers (EXISTING, label_source: annotations)"]
        direction TB
        B1["_annotation_run_specs → load_annotation_anndata (join by key)"]
        B2["train_linear_classifier → save joblib →<br/>metrics_summary.csv → publish → PDF"]
        B1 --> B2
    end

    OUT["output_dir/<br/>metrics_summary.csv · {task}_summary.pdf<br/>pipelines/{task}_{marker}.joblib<br/>[publish_dir/vN + latest] → append-predictions"]

    Z --> A --> LBL --> B --> OUT
    LBL -. "teacher/student: SEC61 labels,<br/>train on a different modality's zarr" .-> B
```





## What's parallel vs sequential

```mermaid
flowchart TD
    E["experiments<br/>(Stage A pools them per marker)"]
    W["witness → per-condition GMM<br/>(one pass per marker)"]
    L["labels.parquet<br/>(single annotation file)"]
    R["run-linear-classifiers<br/>(one LC per (task, marker), existing loop)"]
    O["metrics_summary.csv + pipelines/"]
    E --> W --> L --> R --> O
```



## Recipe / config

**Stage A** — `labels_config.yml`:

```yaml
witness_gmm_labels:
  experiments:
    - experiment: "2026_04_28_A549_SEC61B_DENV"
      embeddings_zarr: ".../2-phenotyping/predictions/embeddings"
      control_filter: {perturbation: uninfected}
      perturbed_filter: {perturbation: [DENV], hours_post_perturbation: {ge: 18, lt: 24}}
  marker_filters: [viral_sensor]          # from viral_sensor → infection_state
  label_column: infection_state
  class_map: {positive: infected, negative: uninfected}
  gmm_pos_threshold: 0.8
  bandwidth: null                         # median heuristic
  max_reference_cells: 5000
  condition_column: perturbation
  output_path: ".../infection_state_witness.csv"
```

For an organelle marker: `marker_filters: [SEC61B]`,
`label_column: organelle_remodeling_state`,
`class_map: {positive: remodel, negative: noremodel}`.

**Stage B** — `train_config.yml` (the existing annotation path):

```yaml
linear_classifiers:
  label_source: annotations
  embeddings_path: ".../embeddings"       # same modality, or a phase zarr for SEC61→phase
  annotations:
    - experiment: "2026_04_28_A549_SEC61B_DENV"
      path: ".../infection_state_witness.csv"
  tasks: [{task: infection_state}]
  use_scaling: true
  split_train_data: 0.8
  split_groups_by: [experiment, fov_name, track_id]
```

Invoke:

```sh
dynaclr witness-gmm-labels -c labels_config.yml
dynaclr run-linear-classifiers -c train_config.yml
```

## Related

- Annotation training path: [evaluation.md](evaluation.md)
