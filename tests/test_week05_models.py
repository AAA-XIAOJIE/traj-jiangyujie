"""Independent formula, sign and dimensional checks for the three models.

Expected values use simple analytical states rather than snapshots of fitted
parameters.  Closing speed is v_follower-v_leader throughout the public API.
"""
from copy import deepcopy
import math

import numpy as np
import pytest

from week05.model.models import (acceleration, bando_velocity,
                                 idm_desired_gap, optimal_velocity)
from week05.model.simulation import Leader, ballistic_step, simulate


def ov_parameters(**changes):
    p = dict(alpha=.5, V1=10., V2=5., C1=.1, C2=2.)
    p.update(changes)
    return p


def idm_parameters(**changes):
    p = dict(v0=36., T=1., a=1., b=1., s0=2., s1=0., delta=4.)
    p.update(changes)
    return p


def test_bando_dimensionless_velocity_reference_points():
    # Bando's V(h)=tanh(h-2)+tanh(2), before SI adaptation.
    assert bando_velocity(0.) == pytest.approx(0., abs=1e-15)
    assert bando_velocity(2.) == pytest.approx(.9640275800758169)
    assert bando_velocity(100.) == pytest.approx(1.964027580075817)


def test_optimal_velocity_and_ovm_hand_calculation():
    p = ov_parameters()
    assert optimal_velocity(20., p) == pytest.approx(10.)
    assert optimal_velocity(30., p) == pytest.approx(13.807970779778824)
    # At a 20 m net gap, alpha=.5/s, V=10 m/s, v=8 m/s -> 1 m/s².
    assert acceleration("OVM", 20., 8., 6., p) == pytest.approx(1.)
    assert acceleration("OVM", 20., 10., 10., p) == pytest.approx(0.)
    # Preserve the published mathematical function; a nonnegative-speed rule
    # belongs in the integrator and must not silently alter V at small gaps.
    small_gap = ov_parameters(V1=1., V2=2., C1=1., C2=1.)
    assert optimal_velocity(.1, small_gap) == pytest.approx(-.4325957403980487)


def test_fvdm_closing_speed_sign_and_zero_coupling_limit():
    p = ov_parameters(**{"lambda": .4, "cutoff_m": 100.})
    # Closing by 2 m/s reduces the OVM acceleration by .8 m/s².
    assert acceleration("FVDM", 20., 8., 6., p) == pytest.approx(.2)
    assert acceleration("FVDM", 20., 8., 10., p) == pytest.approx(1.8)
    without = {**p, "lambda": 0.}
    assert acceleration("FVDM", 20., 8., 6., without) == pytest.approx(
        acceleration("OVM", 20., 8., 6., without))


def test_fvdm_interaction_cutoff_uses_front_to_front_headway():
    p = ov_parameters(**{"lambda": .4, "cutoff_m": 100.})
    # Net gap 90 m + a 10 m leader is exactly the original 100 m cutoff.
    base = acceleration("OVM", 90., 8., 6., p, leader_length=10.)
    at_boundary = acceleration("FVDM", 90., 8., 6., p, leader_length=10.)
    assert at_boundary == pytest.approx(base-.8)
    beyond = acceleration("FVDM", 90., 8., 6., p, leader_length=10.001)
    assert beyond == pytest.approx(base)


def test_idm_original_s1_term_and_exact_numeric_acceleration():
    p = idm_parameters(s1=4.)
    # sqrt(9/36)=.5, so s*=2+4*.5+9+9*(9-8)/2=17.5 m.
    assert idm_desired_gap(9., 8., p) == pytest.approx(17.5)
    # a=1, (v/v0)^4=1/256, (s*/20)^2=49/64.
    assert acceleration("IDM", 20., 9., 8., p) == pytest.approx(.23046875)


def test_idm_original_is_not_the_max_truncated_variant():
    p = idm_parameters(v0=30.)
    # An intentionally fast departing leader exposes the original squared
    # negative desired gap; max(0,...) would give a different positive a.
    assert idm_desired_gap(10., 40., p) == pytest.approx(-138.)
    assert acceleration("IDM", 20., 10., 40., p) == pytest.approx(
        -46.62234567901235)


def test_idm_equilibrium_rest_and_free_road_limits():
    p = idm_parameters()
    v = 9.
    equilibrium_gap = (p["s0"]+v*p["T"])/math.sqrt(1-(v/p["v0"])**4)
    assert acceleration("IDM", equilibrium_gap, v, v, p) == pytest.approx(
        0., abs=1e-12)
    assert acceleration("IDM", 2., 0., 0., p) == pytest.approx(0.)
    assert acceleration("IDM", 1., 0., 0., p) < 0
    assert acceleration("IDM", 3., 0., 0., p) > 0
    assert acceleration("IDM", math.inf, 0., 0., p) == pytest.approx(1.)
    assert acceleration("IDM", math.inf, 36., 36., p) == pytest.approx(0.)


@pytest.mark.parametrize("model", ["OVM", "FVDM", "IDM"])
def test_metre_to_centimetre_conversion_preserves_physical_response(model):
    # A true unit conversion changes all length-bearing quantities, including
    # the inverse-distance C1 and the FVDM cutoff, but not relaxation times.
    p = (idm_parameters(s1=4.) if model == "IDM" else
         ov_parameters(**{"lambda": .4, "cutoff_m": 100.}))
    converted = deepcopy(p)
    if model == "IDM":
        for key in ["v0", "a", "b", "s0", "s1"]:
            converted[key] *= 100.
    else:
        converted["V1"] *= 100.
        converted["V2"] *= 100.
        converted["C1"] /= 100.
        converted["cutoff_m"] *= 100.
    result_m = acceleration(model, 20., 9., 8., p, leader_length=5.)
    result_cm = acceleration(model, 2000., 900., 800., converted,
                             leader_length=500.)
    assert result_cm == pytest.approx(100*result_m, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("model", ["OVM", "FVDM", "IDM"])
def test_nonpositive_gap_is_reported_instead_of_hidden_epsilon_clipping(model):
    p = (idm_parameters() if model == "IDM" else
         ov_parameters(**{"lambda": .4, "cutoff_m": 100.}))
    for gap in [0., -1.]:
        with pytest.raises(ValueError):
            acceleration(model, gap, 8., 6., p)


@pytest.mark.parametrize("model", ["OVM", "FVDM", "IDM"])
def test_vectorized_formula_matches_individual_physical_states(model):
    p = (idm_parameters(s1=4.) if model == "IDM" else
         ov_parameters(**{"lambda": .4, "cutoff_m": 100.}))
    gaps = np.array([10., 20., 120.])
    speeds = np.array([3., 8., 10.])
    leaders = np.array([2., 6., 12.])
    actual = acceleration(model, gaps, speeds, leaders, p)
    expected = [acceleration(model, s, v, vl, p)
                for s, v, vl in zip(gaps, speeds, leaders)]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def constant_leader(x=100., speed=0., length=5.):
    return Leader(np.array([0., 10.]), np.array([x, x+speed*10]),
                  np.array([speed, speed]), length, "constant", speed)


def test_ballistic_stop_event_uses_stopping_distance_not_full_step_motion():
    # v=2 m/s and a=-4 m/s² stop after .5 s at a displacement of .5 m,
    # even though the simulation step extends to 1 s.
    x, v, stopped = ballistic_step(100., 2., -4., 1.)
    assert (x, v, stopped) == (100.5, 0., True)
    x, v, stopped = ballistic_step(100., 2., -4., .25)
    assert (x, v, stopped) == (100.375, 1., False)
    assert ballistic_step(100., 0., -4., 1.) == (100., 0., True)


def test_front_centre_gap_subtracts_the_measured_leader_length():
    leader = constant_leader(x=20., speed=2., length=10.)
    result, meta = simulate("OVM", ov_parameters(), [0., .1], leader,
                            initial_position=5., initial_speed=2., dt=.05)
    assert result.gap_m.iloc[0] == pytest.approx(5.)
    assert meta["completed"]


def test_open_loop_follower_state_is_carried_between_output_times():
    # Closed-form arithmetic for three ballistic steps; output sampling must
    # not reset the follower to any measured or initial state.
    p = ov_parameters(alpha=1., V1=4., V2=0.)
    result, meta = simulate("OVM", p, [0., .1, .2, .3],
                            constant_leader(x=1000.), 0., 2., dt=.1)
    np.testing.assert_allclose(result.v_m_s, [2., 2.2, 2.38, 2.542], atol=1e-12)
    np.testing.assert_allclose(result.x_m, [0., .21, .439, .6851], atol=1e-12)
    assert meta["completed"]


def test_collision_is_reported_and_future_trajectory_is_not_fabricated():
    p = ov_parameters(alpha=1., V1=20., V2=0.)
    result, meta = simulate("OVM", p, np.arange(0, .51, .1),
                            constant_leader(x=10.), 0., 20., dt=.07)
    assert meta["status"] == "collision"
    assert not meta["completed"]
    assert meta["collision_time_s"] == pytest.approx(.25, abs=1e-10)
    assert meta["minimum_gap_m"] == pytest.approx(0.)
    assert meta["minimum_proposed_gap_m"] < 0
    assert np.isfinite(result.loc[result.time_s < .25, "x_m"]).all()
    assert result.loc[result.time_s > .25, ["x_m", "v_m_s", "gap_m"]].isna().all().all()


def test_initial_collision_is_a_recorded_failure_not_an_uncaught_formula_error():
    result, meta = simulate("OVM", ov_parameters(), [0., .1],
                            constant_leader(x=10.), 5., 0., dt=.05)
    assert not meta["completed"] and meta["status"] == "collision"
    assert meta["collision_time_s"] == 0.
    assert result[["x_m", "v_m_s", "gap_m"]].isna().all().all()


def test_invalid_initial_reverse_speed_is_reported_before_integration():
    result, meta = simulate("IDM", idm_parameters(), [0., .1],
                            constant_leader(), 0., -1., dt=.05)
    assert not meta["completed"] and meta["status"] == "invalid_initial_state"
    assert result[["x_m", "v_m_s", "gap_m"]].isna().all().all()


def test_nonnegative_speed_boundary_is_explicitly_counted():
    # Original IDM requests negative acceleration at standstill if s<s0.
    # The numerical boundary holds the stopped car and reports interventions.
    result, meta = simulate("IDM", idm_parameters(), [0., .1, .2],
                            constant_leader(x=6.), 0., 0., dt=.05)
    np.testing.assert_allclose(result.x_m, 0.)
    np.testing.assert_allclose(result.v_m_s, 0.)
    assert meta["completed"]
    assert meta["raw_negative_speed_requests"] == 4
    assert meta["stopping_events"] == 0
    assert meta["status"] == "requires_nonnegative_speed_boundary"


def test_observed_leader_speed_is_consistent_with_interpolated_position():
    leader = Leader(np.array([0., .1, .2]), np.array([10., 11., 13.]),
                    np.array([999., 999., 999.]), 5.)
    assert leader.at(.05) == pytest.approx((10.5, 10.))
    assert leader.at(.15) == pytest.approx((12., 20.))
    assert leader.at(.2) == pytest.approx((13., 20.))
    # Numerical integration roundoff must not switch the derivative side.
    assert leader.at(.1-1e-12)[1] == pytest.approx(20.)
    assert leader.at(.1)[1] == pytest.approx(20.)
    assert leader.at(.1-1e-6)[1] == pytest.approx(10.)


def test_coordinate_translation_does_not_change_follower_response():
    p = ov_parameters(alpha=1., V1=4., V2=0.)
    times = [0., .1, .2, .3]
    base, _ = simulate("OVM", p, times, constant_leader(x=100.), 0., 2., dt=.05)
    moved, _ = simulate("OVM", p, times, constant_leader(x=1000100.),
                        1000000., 2., dt=.05)
    np.testing.assert_allclose(moved.x_m-1000000., base.x_m, atol=1e-9)
    np.testing.assert_allclose(moved.v_m_s, base.v_m_s, atol=1e-12)
    np.testing.assert_allclose(moved.gap_m, base.gap_m, atol=1e-9)
