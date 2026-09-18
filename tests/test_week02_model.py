"""Numerical contracts for the course ORCA port and experiment parser."""
import numpy as np
import pytest

from week02.model.analysis import read_experiment, velocity, initial_conditions
from week02.model.interaction import time_to_collision
from week02.model.orca import lp2, step
from week02.model.simulation import simulate


def test_half_plane_projection_and_speed_disk():
    # x <= 0.5, y >= 0.2
    lines = [(np.array([0.5, 0.0]), np.array([0.0, 1.0])),
             (np.array([0.0, 0.2]), np.array([1.0, 0.0]))]
    failed, result = lp2(lines, 1.0, np.array([2.0, 0.0]))
    assert failed == 2
    np.testing.assert_allclose(result, [0.5, 0.2], atol=1e-8)


def test_two_agents_head_on_remain_separated():
    p = np.array([[-2., 0.], [2., 0.]])
    v = np.zeros_like(p)
    goals = -p
    minimum = 4.0
    for _ in range(150):
        pref = goals - p
        pref /= np.maximum(np.linalg.norm(pref, axis=1)[:, None], 1.0)
        v, _, violation = step(p, v, pref, np.ones(2), .25, 2., .04)
        p += .04 * v
        minimum = min(minimum, np.linalg.norm(p[0] - p[1]))
        assert violation < 1e-7
    assert minimum >= .5 - 1e-5


def test_agents_moving_apart_keep_preferred_velocity():
    p = np.array([[-2., 0.], [2., 0.]])
    v = np.array([[-1., 0.], [1., 0.]])
    actual, failed, _ = step(p, v, v, np.ones(2), .25, 2., .04)
    np.testing.assert_allclose(actual, v)
    assert failed == 0


def test_coincident_agents_are_finite_and_separate():
    p = np.zeros((2, 2))
    v, failed, _ = step(p, p, p, np.ones(2), .25, 2., .04)
    assert np.isfinite(v).all()
    assert np.linalg.norm(v[0] - v[1]) > 0
    assert failed > 0  # impossible one-step separation at the speed limit


def test_velocity_units_and_constant_motion():
    p = np.arange(20)[:, None, None] * .04 * np.array([[[2., -1.]]])
    np.testing.assert_allclose(velocity(p, .04), np.broadcast_to([2., -1.], p.shape), atol=1e-12)


def test_file_order_and_missing_frame_validation(tmp_path):
    p = tmp_path / "data.txt"
    np.savetxt(p, [[1, 10, -100, 0, 180], [2, 10, 100, 0, 180],
                   [1, 11, -90, 0, 180], [2, 11, 90, 0, 180]])
    e = read_experiment(p)
    np.testing.assert_allclose(e["positions"][0], [[-1, 0], [1, 0]])
    assert e["times"][1] == .04
    np.savetxt(p, [[1, 10, -100, 0, 180], [2, 10, 100, 0, 180], [1, 11, -90, 0, 180]])
    with pytest.raises(ValueError, match="Incomplete"):
        read_experiment(p)


def test_ttc_head_on_away_and_miss():
    relative = np.array([[2., 0.], [2., 0.], [2., 1.]])
    speed = np.array([[1., 0.], [-1., 0.], [1., 0.]])
    t = time_to_collision(relative, speed, .5)
    assert t[0] == pytest.approx(1.5)
    assert np.isinf(t[1:]).all()
    assert time_to_collision(np.array([.2, 0.]), np.zeros(2), .5) == 0


def test_initialization_does_not_use_future_trajectory():
    p = np.zeros((101, 2, 2))
    p[:, 0, 0] = np.arange(101) * .04
    p[:, 1, 1] = np.arange(101) * .08
    e = {"positions": p, "times": np.arange(101) * .04}
    config = {"initial_velocity_window_s": .2, "calibration_end_s": 2.0}
    reference = initial_conditions(e, config)
    p[51:] += 10000
    for actual, expected in zip(initial_conditions(e, config), reference):
        np.testing.assert_array_equal(actual, expected)


def test_non_degenerate_orca_is_rotation_equivariant():
    p = np.array([[-1., -.2], [1., .1], [0., 2.]])
    v = np.array([[.9, .1], [-.7, -.05], [.1, -.6]])
    theta = .43
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    expected, _, _ = step(p, v, v, np.ones(3), .25, 2., .04)
    actual, _, _ = step(p @ rotation.T, v @ rotation.T, v @ rotation.T, np.ones(3), .25, 2., .04)
    np.testing.assert_allclose(actual, expected @ rotation.T, atol=1e-8)


def test_supplied_source_is_byte_preserved():
    import hashlib
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "week02/model/data/circle-10m-64-1.txt"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "05d52540e3a4f427f52627cc4aaf9c7d1dde7228ca5513995284d0f52f14d790"
