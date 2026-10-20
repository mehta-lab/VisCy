"""Set-classifier export: lineage label rule, representative frames, FOV folds, matched pools and pinned dumps."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/zikv_set_transformer/tomm20_cycle_vs_infection_pca80_sensor_verified.yml"
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
    assert set(onset.infection_class) <= set(m.infection_classes(CFG))
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


def test_infection_classifier_pools_are_phase_matched(exported):
    _, _, reps, _, _, pools, counts, _ = exported
    for (model, fold), p in pools.items():
        if model != "infection_classifier":
            continue
        for split in m.SPLITS:
            for c in p["classes"]:
                phases = reps.cc_state.iloc[p[split][c]].value_counts()
                assert set(phases.index) == set(CFG["phases"]), (fold, split, c)
                assert (phases == phases.iloc[0]).all(), (fold, split, c, phases.to_dict())
                assert reps.infection_class.iloc[p[split][c]].eq(c).all()
    kept = counts[counts.model.eq("infection_classifier") & counts.kept]
    assert kept.after.eq(3 * kept.per_stratum_after).all()


def test_cell_cycle_classifier_pools_hold_only_balanced_references(exported):
    _, _, reps, _, _, pools, _, decision = exported
    for (model, fold), p in pools.items():
        if model != "cell_cycle_classifier":
            continue
        assert p["classes"][:3] == CFG["phases"]
        for split in m.SPLITS:
            for c in p["classes"]:
                cells = reps.iloc[p[split][c]]
                assert cells.group.isin(m.REFERENCES).all()
                assert cells.cc_state.eq(c).all()
                sources = cells.group.value_counts()
                assert set(sources.index) == set(m.REFERENCES) and (sources == sources.iloc[0]).all()
    assert decision["include_m"] == ("M" in pools[("cell_cycle_classifier", "fold0")]["classes"])


def test_dropped_infection_classs_absent_from_every_split(exported):
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
    p = pools[("infection_classifier", "fold0")]
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
        assert joined.infection_class.eq(c).all()
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
    target = bad.index[bad.model.eq("cell_cycle_classifier") & bad.cls.eq("G1") & bad.group.eq("BYSTANDER")][0]
    same_fov = reps.fov_key.eq(bad.fov_key[target]) & reps.group.eq("onset") & reps.cc_state.eq("G1")
    bad.loc[target, "row_id"] = int(reps.row_id[same_fov].iloc[0])
    bad = bad[members.columns]
    with pytest.raises(ValueError, match="Cell-cycle classifier"):
        m.check_members(bad, reps, CFG)


def lineage_table():
    """TCN- and sensor-anchored frames of six lineages in one 07_01 control FOV and one ZIKV FOV."""
    lineages = [
        ("A/3/000", 1, ["uninfected"] * 3, ["uninfected"] * 3),
        ("B/3/000", 2, ["pre_onset", "infected", "infected"], ["pre_onset", "pre_onset", "infected"]),
        ("B/3/000", 3, ["not_infected"] * 3, ["pre_onset", "infected", "infected"]),
        ("B/3/000", 4, ["not_infected"] * 3, ["not_infected"] * 3),
        ("B/3/000", 5, ["ambiguous"] * 3, ["ambiguous"] * 3),
        ("B/3/000", 6, ["pre_onset", "infected", "infected"], ["not_infected"] * 3),
    ]
    tcn, sensor = [], []
    for fov, track, tcn_status, sensor_status in lineages:
        for t in range(3):
            key = dict(dataset="07_01", fov_name=fov, track_id=track, t=t, lineage_uid=f"07_01::{fov}::{track}")
            onset = tcn_status[t] in m.ONSET_STATUSES
            tcn.append(
                dict(
                    key,
                    infection_status=tcn_status[t],
                    hpi=10.0 + 0.5 * t,
                    tau_h=0.5 * t - 0.5 if onset else np.nan,
                    cc_state="G1",
                    mantis_death=track == 4 and t == 2,
                )
            )
            sensor.append(dict(key, infection_status=sensor_status[t]))
    return pd.DataFrame(tcn), pd.DataFrame(sensor)


def test_lineage_rule_keeps_only_classes_both_anchors_agree_on():
    tcn, sensor = lineage_table()
    table = m.select_lineages(tcn, sensor, require_agreement=True)
    uid = "07_01::{}::{}".format
    assert table.loc[uid("B/3/000", 3), ["tcn", "sensor"]].tolist() == ["not_infected", "onset"]
    kept = table.tcn[table.kept].to_dict()
    assert kept == {uid("A/3/000", 1): "uninfected", uid("B/3/000", 2): "onset", uid("B/3/000", 4): "not_infected"}
    loose = m.select_lineages(tcn, None, require_agreement=False)
    assert set(loose.index[loose.kept]) == {uid("A/3/000", 1), *(uid("B/3/000", k) for k in (2, 3, 4, 6))}
    mixed = tcn.assign(
        infection_status=lambda x: np.where(x.track_id.eq(4) & x.t.eq(0), "ambiguous", x.infection_status)
    )
    with pytest.raises(ValueError, match="more than one infection_status"):
        m.lineage_classes(mixed)


def test_load_labels_excludes_a_tcn_bystander_that_the_sensor_calls_onset(tmp_path):
    tcn, sensor = lineage_table()
    tcn.to_parquet(tmp_path / "tcn.parquet", index=False)
    sensor.to_parquet(tmp_path / "sensor.parquet", index=False)
    cfg = dict(
        CFG,
        labels=str(tmp_path / "tcn.parquet"),
        labels_sha256=m.sha(tmp_path / "tcn.parquet"),
        expected_frames=len(tcn),
        sensor_labels=dict(path=str(tmp_path / "sensor.parquet"), sha256=m.sha(tmp_path / "sensor.parquet")),
        require_sensor_agreement=True,
    )
    d, record, table = m.load_labels(cfg)
    groups = d.groupby("track_id").group.unique().map(list).to_dict()
    assert groups == {1: ["CONTROL"], 2: ["onset"], 4: ["BYSTANDER"]}
    assert len(d[d.track_id.eq(4)]) == 2 and record["mantis_death_excluded"] == 1
    assert record["live_frames_excluded_by_rule"] == 9
    assert d.tau_h[d.group.eq("onset")].notna().all() and d.tau_h[d.group.ne("onset")].isna().all()
    sensor.loc[0, "lineage_uid"] = "07_01::A/3/000::9"
    sensor.to_parquet(tmp_path / "sensor.parquet", index=False)
    cfg["sensor_labels"]["sha256"] = m.sha(tmp_path / "sensor.parquet")
    with pytest.raises(ValueError, match="different lineages"):
        m.load_labels(cfg)
