"""Attempt-25 export: representative frames, FOV folds, phase-matched pools and pinned-format dumps."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/zikv_set_transformer/tomm20_cycle_vs_infection_pca80.yml"
spec = importlib.util.spec_from_file_location("tomm20_set_classifier", ROOT / "tools/tomm20_set_classifier.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CFG = m.load_config(CONFIG)
TRAIN = m.pinned_train(CFG)


def synthetic_frames(seed=0):
    """Two datasets, each with 5 control-well and 10 ZIKV-well FOVs of 12 tracks x 60 half-hour frames."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds in ["07_01", "04_10"]:
        for well, well_type, n_fov in [("A/1", "control", 5), ("B/1", "zikv", 10)]:
            for f in range(n_fov):
                fov = f"{well}/{f:03d}"
                for track in range(12):
                    if well_type == "control":
                        group, status = "CONTROL", "uninfected"
                    elif track % 2:
                        group, status = "BYSTANDER", "not_infected"
                    else:
                        group, status = "onset", "infected"
                    onset_t = int(rng.integers(20, 40))
                    for t in range(60):
                        rows.append(
                            dict(
                                dataset=ds,
                                fov_name=fov,
                                track_id=track,
                                t=t,
                                lineage_uid=f"{ds}::{fov}::{track}",
                                infection_status=status,
                                hpi=3.0 + 0.5 * t,
                                tau_h=0.5 * (t - onset_t) if group == "onset" else np.nan,
                                cc_state=str(rng.choice(m.ALL_PHASES, p=[0.35, 0.3, 0.3, 0.05])),
                                group=group,
                                well=well,
                                well_type=well_type,
                            )
                        )
    d = pd.DataFrame(rows)
    d["fov_key"] = d.dataset + "::" + d.fov_name
    d["x"] = rng.uniform(200, 800, len(d))
    d["y"] = rng.uniform(200, 800, len(d))
    emb = rng.normal(size=(len(d), CFG["embedding_dims"])).astype(np.float32)
    return d, emb


@pytest.fixture(scope="module")
def exported():
    """Representatives, folds and pools of the synthetic table with the attempt config."""
    d, emb = synthetic_frames()
    reps, _ = m.representatives(d, emb, CFG)
    fold_of = m.assign_folds(reps, CFG["n_folds"], CFG["seed"])
    folds = {**m.cv_splits(fold_of, CFG["n_folds"]), **m.lodo_splits(reps, CFG["lodo_val_fraction"], CFG["seed"])}
    pools, counts, decision = m.build_pools(reps, folds, CFG)
    return d, emb, reps, fold_of, folds, pools, counts, decision


def test_one_frame_per_track_and_bin_nearest_the_centre():
    d = pd.DataFrame(
        dict(
            dataset="07_01",
            fov_name="B/1/000",
            track_id=1,
            t=[0, 1, 2, 3, 4],
            tau_h=[-8.0, -7.5, -6.5, -6.0, -5.5],
        )
    )
    d["tau_bin"] = np.floor((d.tau_h + 8) / 2) * 2 - 8
    reps = m.nearest_to_center(d, "tau_bin", "tau_h", 2)
    assert reps.t.tolist() == [1, 4]


def test_representatives_are_unique_per_track_bin(exported):
    _, _, reps, *_ = exported
    onset = reps[reps.group.eq("onset")]
    refs = reps[reps.group.isin(m.REFERENCES)]
    assert not onset.duplicated([*m.TRACK, "tau_bin"]).any()
    assert not refs.duplicated([*m.TRACK, "hpi_bin"]).any()
    assert onset.tau_h.ge(CFG["tau_start"]).all() and onset.tau_h.lt(CFG["tau_stop"]).all()
    assert set(onset.class_i) <= set(m.model_i_classes(CFG))
    assert reps.row_id.tolist() == list(range(len(reps)))


def test_fovs_are_disjoint_across_splits(exported):
    _, _, reps, fold_of, folds, *_ = exported
    m.check_disjoint(folds)
    for f in folds.values():
        assert set().union(*(set(f[s]) for s in m.SPLITS)) == set(reps.fov_key)
    tests = [k for name, f in folds.items() if name.startswith("fold") for k in f["test"]]
    assert sorted(tests) == sorted(fold_of)
    for name, f in folds.items():
        if name.startswith("lodo"):
            train_ds = {k.split("::")[0] for k in f["train"] + f["val"]}
            test_ds = {k.split("::")[0] for k in f["test"]}
            assert len(train_ds) == 1 and len(test_ds) == 1 and train_ds != test_ds
    per_fold = reps.drop_duplicates("fov_key").assign(fold=lambda x: x.fov_key.map(fold_of))
    controls = per_fold[per_fold.well_type.eq("control")].groupby("fold").size()
    assert controls.max() - controls.min() <= 1
    with pytest.raises(ValueError, match="two splits"):
        m.check_disjoint(dict(bad=dict(train=["a"], val=["a"], test=["b"])))


def test_model_i_pools_are_phase_matched(exported):
    _, _, reps, _, _, pools, counts, _ = exported
    for (model, fold), p in pools.items():
        if model != "model_i":
            continue
        for split in m.SPLITS:
            for c in p["classes"]:
                phases = reps.cc_state.iloc[p[split][c]].value_counts()
                assert set(phases.index) == set(CFG["phases"]), (fold, split, c)
                assert (phases == phases.iloc[0]).all(), (fold, split, c, phases.to_dict())
                assert reps.class_i.iloc[p[split][c]].eq(c).all()
    kept = counts[counts.model.eq("model_i") & counts.kept]
    assert kept.after.eq(3 * kept.per_stratum_after).all()


def test_model_c_pools_hold_only_balanced_references(exported):
    _, _, reps, _, _, pools, _, decision = exported
    for (model, fold), p in pools.items():
        if model != "model_c":
            continue
        assert p["classes"][:3] == CFG["phases"]
        for split in m.SPLITS:
            for c in p["classes"]:
                cells = reps.iloc[p[split][c]]
                assert cells.group.isin(m.REFERENCES).all()
                assert cells.cc_state.eq(c).all()
                sources = cells.group.value_counts()
                assert set(sources.index) == set(m.REFERENCES) and (sources == sources.iloc[0]).all()
    assert decision["include_m"] == ("M" in pools[("model_c", "fold0")]["classes"])


def test_dropped_class_is_absent_from_every_split(exported):
    _, _, reps, _, folds, *_ = exported
    strict = dict(CFG, min_train_cells=10**6)
    pools, counts, _ = m.build_pools(reps, folds, strict)
    for p in pools.values():
        assert p["classes"] == []
        assert all(p[s] == {} for s in m.SPLITS)
        assert all(r.startswith("train") for r in p["dropped"].values())
    assert not counts.kept.any()


def test_dumps_round_trip_through_pinned_load_dataset(exported, tmp_path):
    _, _, reps, _, _, pools, _, _ = exported
    emb = reps[m.pc_columns(CFG)].to_numpy(np.float32)
    p = pools[("model_i", "fold0")]
    for split in m.SPLITS:
        directory = tmp_path / split
        m.write_dump(directory, p["classes"], p[split], reps, emb, CFG["dump_n_cells"])
        loaded = m.verify_dump(directory, p["classes"], p[split], emb, TRAIN.load_dataset)
        assert loaded == {c: len(p[split][c]) for c in p["classes"]}
    ds = TRAIN.load_dataset(str(tmp_path / "test"), n_cells=10)
    for c, g in ds.gene_to_idx.items():
        meta = pd.DataFrame(ds._gene_dump_meta[g]).add_prefix("dump_")
        joined = meta.merge(reps, left_on="dump_index", right_on="row_id", how="left", validate="one_to_one")
        assert joined.row_id.notna().all()
        assert joined.dump_experiment.eq(joined.dataset).all() and joined.dump_well.eq(joined.fov_name).all()
        assert (joined.dump_segmentation_id == joined.track_id.astype(str) + ":" + joined.t.astype(str)).all()
        assert joined.dump_cc_state.eq(joined.cc_state).all() and joined.dump_group.eq(joined.group).all()
        assert joined.class_i.eq(c).all()
        np.testing.assert_array_equal(ds._gene_pools[g].numpy(), emb[joined.row_id.to_numpy()])
        embs, ch_ids, mask, label = ds[ds.perturbation_list.index(c)]
        assert embs.shape == (10, CFG["embedding_dims"]) and label == g and ch_ids.eq(0).all()
    with pytest.raises(FileExistsError):
        m.write_dump(tmp_path / "test", p["classes"], p["test"], reps, emb, CFG["dump_n_cells"])


def test_member_checks_reject_a_mixed_pool(exported):
    _, _, reps, _, _, pools, _, _ = exported
    rows = []
    for (model, fold), p in pools.items():
        for split in m.SPLITS:
            for c in p["classes"]:
                rows.append(pd.DataFrame(dict(model=model, fold=fold, split=split, cls=c, row_id=p[split][c])))
    members = pd.concat(rows, ignore_index=True)
    m.check_members(members, reps, CFG)
    bad = members.merge(reps[["row_id", "group", "fov_key"]], on="row_id")
    target = bad.index[bad.model.eq("model_c") & bad.cls.eq("G1") & bad.group.eq("BYSTANDER")][0]
    same_fov = reps.fov_key.eq(bad.fov_key[target]) & reps.group.eq("onset") & reps.cc_state.eq("G1")
    bad.loc[target, "row_id"] = int(reps.row_id[same_fov].iloc[0])
    bad = bad[members.columns]
    with pytest.raises(ValueError, match="Model C"):
        m.check_members(bad, reps, CFG)
