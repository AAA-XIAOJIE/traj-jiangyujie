"""Audit Week 05 scientific evidence, units, experiment separation and failures."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from week05.model.calibration import objective
from week05.model.simulation import from_observations, simulate

ROOT = Path(__file__).resolve().parents[1]
WEEK = ROOT / "week05"
COMMITTED = WEEK / "results"
RESULTS = Path(os.environ.get("WEEK05_RESULTS", str(COMMITTED))).resolve()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def test_input_geometry_and_calibration_validation_are_disjoint():
    data = pd.read_csv(WEEK / "data/following_pair.csv")
    config = load_json(WEEK / "config.json")
    experiment = load_json(RESULTS / "experiment.json")
    calibration = load_json(RESULTS / "calibration.json")
    manifest = load_json(WEEK / "data/manifest.json")
    assert not data.isna().any().any()
    assert not data.duplicated(["time_s"]).any()
    assert data.follower_id.nunique() == data.leader_id.nunique() == 1
    assert data.lane_id.nunique() == 1
    np.testing.assert_allclose(np.diff(data.time_s), .1, atol=1e-10)
    np.testing.assert_allclose(data.net_gap_m,
        data.leader_x_m-data.follower_x_m-data.leader_length_m, atol=1e-8)
    assert (data.net_gap_m > 0).all()
    train = data[data.relative_time_s < config["calibration_end_s"]]
    validation = data[data.relative_time_s >= config["calibration_end_s"]]
    assert len(train) == calibration["training_samples"] == 300
    assert len(validation) == calibration["validation_samples"] == 301
    assert set(train.frame_id).isdisjoint(validation.frame_id)
    np.testing.assert_allclose(experiment["calibration_time_s"],
                               [train.time_s.min(), train.time_s.max()])
    np.testing.assert_allclose(experiment["validation_time_s"],
                               [validation.time_s.min(), validation.time_s.max()])
    assert experiment["leader_length_m"] == pytest.approx(3.017526)
    assert manifest["selection_is_independent_of_models"] is True
    assert hashlib.sha256((WEEK / "data/following_pair.csv").read_bytes()).hexdigest() == manifest["subset_sha256"]


def test_every_model_scenario_and_parameter_set_has_the_same_initial_state():
    trajectories = pd.read_csv(RESULTS / "trajectories.csv")
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    experiment = load_json(RESULTS / "experiment.json")
    assert len(metrics) == 18
    assert set(metrics.model) == {"OVM", "FVDM", "IDM"}
    assert set(metrics.parameter_set) == {"baseline", "calibrated"}
    assert set(metrics.scenario) == {"constant", "braking", "observed"}
    assert not metrics.duplicated(["model", "parameter_set", "scenario"]).any()
    groups = trajectories.groupby(["model", "parameter_set", "scenario"])
    assert groups.ngroups == 18
    for _, frame in groups:
        np.testing.assert_allclose(frame.time_s, np.arange(301)*.1, atol=1e-9)
        initial = frame.iloc[0]
        assert initial.x_m == pytest.approx(experiment["shared_initial_position_m"])
        assert initial.v_m_s == pytest.approx(experiment["shared_initial_speed_m_s"])
        assert initial.leader_x_m == pytest.approx(experiment["initial_leader_position_m"])
        assert initial.gap_m == pytest.approx(experiment["initial_net_gap_m"])


def test_counterfactual_accuracy_is_missing_and_observed_errors_recompute():
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    trajectories = pd.read_csv(RESULTS / "trajectories.csv")
    config = load_json(WEEK / "config.json")
    data = pd.read_csv(WEEK / "data/following_pair.csv")
    truth = data[data.relative_time_s >= config["calibration_end_s"]]
    error_columns = ["position_rmse_m", "speed_rmse_m_s", "gap_rmse_m"]
    assert metrics.loc[metrics.scenario != "observed", error_columns].isna().all().all()
    for row in metrics[metrics.scenario == "observed"].itertuples():
        predicted = trajectories[(trajectories.model == row.model) &
            (trajectories.parameter_set == row.parameter_set) &
            (trajectories.scenario == "observed")]
        if row.completed:
            for estimate, target, column in [
                (predicted.x_m, truth.follower_x_m, "position_rmse_m"),
                (predicted.v_m_s, truth.follower_v_raw_m_s, "speed_rmse_m_s"),
                (predicted.gap_m, truth.net_gap_m, "gap_rmse_m"),
            ]:
                expected = np.sqrt(np.mean((estimate.to_numpy()-target.to_numpy())**2))
                assert getattr(row, column) == pytest.approx(expected, rel=1e-8, abs=1e-8)
        else:
            assert all(np.isnan(getattr(row, column)) for column in error_columns)


def test_failures_and_stopping_interventions_remain_visible():
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    trajectories = pd.read_csv(RESULTS / "trajectories.csv")
    for row in metrics.itertuples():
        frame = trajectories[(trajectories.model == row.model) &
            (trajectories.parameter_set == row.parameter_set) &
            (trajectories.scenario == row.scenario)]
        assert row.valid_output_samples == frame.x_m.notna().sum()
        assert row.raw_negative_speed_requests >= row.stopping_events >= 0
        if row.status == "collision":
            assert not row.completed
            assert np.isfinite(row.collision_time_s)
            assert row.minimum_gap_m == pytest.approx(0., abs=1e-10)
            assert row.minimum_proposed_gap_m <= 0
            failed = frame.time_s >= row.collision_time_s
            assert frame.loc[failed, ["x_m", "v_m_s", "gap_m"]].isna().all().all()
            assert frame.loc[~failed, "x_m"].notna().all()
        elif row.completed:
            assert frame[["x_m", "v_m_s", "gap_m"]].notna().all().all()
            assert row.minimum_gap_m > 0
            assert (frame.v_m_s >= 0).all()
            assert np.isnan(row.collision_time_s)
            assert row.minimum_gap_m <= frame.gap_m.min()+1e-8
        if row.status == "requires_nonnegative_speed_boundary":
            assert row.raw_negative_speed_requests > 0
            assert np.isfinite(row.first_negative_speed_request_s)


def test_recorded_calibration_objective_uses_only_the_training_prefix():
    config = load_json(WEEK / "config.json")
    data = pd.read_csv(WEEK / "data/following_pair.csv")
    train = data[data.relative_time_s < config["calibration_end_s"]]
    calibration = load_json(RESULTS / "calibration.json")
    for model, fitted in calibration["fit"].items():
        names = fitted["calibrated_parameter_names"]
        for set_name, score_key in [("baseline", "baseline_objective"),
                                    ("calibrated", "objective")]:
            p = calibration["parameters"][set_name][model]
            score = objective(model, [[p[name] for name in names]], names, train, config)[0]
            assert score == pytest.approx(fitted[score_key], rel=1e-9, abs=1e-10)
        for name, (lower, upper) in fitted["bounds"].items():
            assert lower <= calibration["parameters"]["calibrated"][model][name] <= upper


@pytest.mark.parametrize("model", ["OVM", "FVDM", "IDM"])
def test_calibration_and_main_simulator_agree_at_the_same_numerical_step(model):
    config = load_json(WEEK / "config.json")
    data = pd.read_csv(WEEK / "data/following_pair.csv")
    train = data[data.relative_time_s < config["calibration_end_s"]]
    p = load_json(RESULTS / "calibration.json")["parameters"]["calibrated"][model]
    names = list(config["parameter_bounds"][model])
    score = objective(model, [[p[name] for name in names]], names, train, config)[0]
    trajectory, meta = simulate(model, p, train.relative_time_s.to_numpy(),
        from_observations(train), float(train.follower_x_m.iloc[0]),
        float(train.follower_v_raw_m_s.iloc[0]), dt=config["calibration_step_s"])
    assert meta["completed"] and meta["raw_negative_speed_requests"] == 0
    independent_score = np.mean(
        ((trajectory.x_m.to_numpy()-train.follower_x_m.to_numpy())/config["position_loss_scale_m"])**2 +
        ((trajectory.v_m_s.to_numpy()-train.follower_v_raw_m_s.to_numpy())/config["speed_loss_scale_m_s"])**2)
    assert score == pytest.approx(independent_score, rel=1e-10, abs=1e-10)


def test_timestep_study_preserves_qualitative_outcomes_and_reports_fine_reference():
    convergence = pd.read_csv(RESULTS / "convergence.csv")
    config = load_json(WEEK / "config.json")
    assert len(convergence) == 27
    errors = ["max_position_difference_vs_fine_m", "max_speed_difference_vs_fine_m_s"]
    for _, group in convergence.groupby(["model", "scenario"]):
        np.testing.assert_allclose(sorted(group.step_s), sorted(config["convergence_steps_s"]))
        assert group.status.nunique() == 1
        assert np.isfinite(group[errors]).all().all()
        assert (group[errors] >= 0).all().all()
        fine = group[group.step_s == min(config["convergence_steps_s"])]
        np.testing.assert_allclose(fine[errors], 0., atol=1e-12)


def test_input_config_and_outputs_match_recorded_provenance():
    record_path = RESULTS / "reproducibility.json"
    if not record_path.exists():
        pytest.skip("Generate final provenance with week05.model.run before release")
    record = load_json(record_path)
    assert hashlib.sha256((WEEK / "data/following_pair.csv").read_bytes()).hexdigest() == record["input_sha256"]
    assert hashlib.sha256((WEEK / "config.json").read_bytes()).hexdigest() == record["config_sha256"]
    for name, expected in record["source_code_sha256"].items():
        assert hashlib.sha256((WEEK / name).read_bytes()).hexdigest() == expected, name
    for name, expected in record["output_sha256"].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == expected, name


@pytest.mark.skipif(RESULTS == COMMITTED,
                   reason="Set WEEK05_RESULTS to audit an independent output directory")
def test_independent_numerical_results_match_the_submitted_results():
    for name in ["metrics.csv", "trajectories.csv", "convergence.csv",
                 "braking_sensitivity.csv", "feedback_ablation.csv"]:
        pd.testing.assert_frame_equal(pd.read_csv(RESULTS / name), pd.read_csv(COMMITTED / name),
                                      check_dtype=False, rtol=1e-8, atol=1e-7)
