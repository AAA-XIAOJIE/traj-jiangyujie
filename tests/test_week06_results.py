"""Published evidence checks; no optimistic throughput claims after violations."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get("WEEK06_RESULTS",ROOT/"week06/results"))


def test_all_declared_runs_and_same_width_conditions_present():
    data=pd.read_csv(OUT/"metrics.csv")
    assert len(data)==39
    assert not data.duplicated(["scene","width_m","model","variant"]).any()
    assert set(data.model)=={"sfm","idm","orca"}
    assert set(data.variant)=={"base","guided","steering"}
    assert len(data[data.scene=="narrow"])==24
    assert len(data[data.scene=="circle"])==6 and len(data[data.scene=="circle_mixed"])==9


def test_arrival_and_safety_are_distinct():
    data=pd.read_csv(OUT/"metrics.csv")
    expected=(data.arrived==data.agents)&(data.body_overlap_pair_steps==0)&(data.wall_violation_agent_steps==0)
    np.testing.assert_array_equal(data.geometry_valid_complete,expected)
    np.testing.assert_array_equal(data.physically_feasible_complete,expected&(data.dynamic_bound_violation_agent_steps==0)&(data.steering_bound_violation_agent_steps==0))
    assert ((data.arrived>=0)&(data.arrived<=data.agents)).all()
    assert data.loc[data.arrived<data.agents,"all_arrival_time_s"].isna().all()
    assert data.loc[data.scene!="circle","ade_after_calibration_m"].isna().all()


def test_trace_shapes_initial_states_and_counterfactual_reference():
    metrics=pd.read_csv(OUT/"metrics.csv");z=np.load(OUT/"trajectories.npz")
    initials={}
    for row in metrics.itertuples():
        width=None if np.isnan(row.width_m) else row.width_m
        key=f"{row.scene}_{width}_{row.model}_{row.variant}"
        p=z[key+"__positions"];v=z[key+"__velocities"];t=z[key+"__times"]
        assert p.shape==v.shape==(len(t),row.agents,2)
        assert np.isfinite(p).all() and np.isfinite(v).all()
        np.testing.assert_allclose(np.diff(t),row.dt_s,atol=1e-10)
        if row.scene in initials:np.testing.assert_array_equal(p[0],initials[row.scene])
        else:initials[row.scene]=p[0]
        arrived=z[key+"__arrival_times"]
        assert np.isfinite(arrived).sum()==row.arrived


def test_source_input_matches_previous_week_and_hash_record():
    e=json.loads((OUT/"experiment.json").read_text(encoding="utf8"))
    assert hashlib.sha256((ROOT/e["source_file"]).read_bytes()).hexdigest()==e["source_sha256"]
    assert e["geometric_minimum_road_width_m"]==pytest.approx(2.8)
    for name,h in e["source_code_sha256"].items():
        assert hashlib.sha256((ROOT/"week06/model"/name).read_bytes()).hexdigest()==h
    for name,h in e["dependency_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h


def test_output_manifest_and_initial_condition_coverage():
    receipt=json.loads((OUT/"reproducibility.json").read_text(encoding="utf8"))
    assert hashlib.sha256((ROOT/"week06/config.json").read_bytes()).hexdigest()==receipt["config_sha256"]
    for name,digest in receipt["output_sha256"].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    initial=pd.read_csv(OUT/"initial_conditions.csv")
    assert initial.groupby("scene").size().to_dict()=={"circle":64,"circle_mixed":64,"narrow":7}
    assert not initial.duplicated(["scene","agent_id"]).any()


@pytest.mark.skipif("WEEK06_RESULTS" not in os.environ,reason="Set independent output directory to compare runs")
def test_independent_numerical_reproduction():
    baseline=pd.read_csv(ROOT/"week06/results/metrics.csv")
    independent=pd.read_csv(OUT/"metrics.csv")
    pd.testing.assert_frame_equal(baseline,independent,check_dtype=False,rtol=1e-8,atol=1e-8)
