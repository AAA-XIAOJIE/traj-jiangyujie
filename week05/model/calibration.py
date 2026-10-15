"""Deterministic bounded multistart search on calibration observations only.

Vectorization evaluates candidate parameters together; it does not assimilate
the observed follower state after initialization.
"""
from __future__ import annotations

import numpy as np

from .models import DEFAULTS, acceleration
from .simulation import from_observations


def objective(model, candidates, names, data, config):
    candidates = np.atleast_2d(candidates)
    n = len(candidates)
    params = dict(DEFAULTS[model])
    params.update({name:candidates[:, j] for j, name in enumerate(names)})
    leader = from_observations(data)
    t = data.relative_time_s.to_numpy(); t = t-t[0]
    observations_x = data.follower_x_m.to_numpy()
    observations_v = data.follower_v_raw_m_s.to_numpy()
    x = np.full(n, observations_x[0]); v = np.full(n, observations_v[0])
    score = np.zeros(n); valid = np.ones(n, dtype=bool)
    for i in range(len(t)):
        score += ((x-observations_x[i])/config["position_loss_scale_m"])**2
        score += ((v-observations_v[i])/config["speed_loss_scale_m_s"])**2
        if i == len(t)-1: break
        dt = t[i+1]-t[i]
        if not np.isclose(dt, config["calibration_step_s"]):
            raise ValueError("Calibration input must use the recorded 0.1 s grid")
        lx, lv = leader.at(t[i])
        gap = lx-x-leader.length
        a = acceleration(model, np.maximum(gap, 1e-6), v, lv, params, leader.length)
        xx = x+v*dt+.5*a*dt*dt; vv = v+a*dt
        # Invalid candidates are penalized, never repaired to obtain a good fit.
        good = np.isfinite(xx) & np.isfinite(vv) & (vv >= 0)
        next_lx, _ = leader.at(t[i+1])
        good &= (next_lx-xx-leader.length) > 0
        valid &= good
        x = np.where(valid, xx, observations_x[i+1])
        v = np.where(valid, vv, observations_v[i+1])
    return np.where(valid, score/len(t), 1e6)


def fit(model, data, config):
    bounds = config["parameter_bounds"][model]
    names = list(bounds)
    lower, upper = np.array(list(bounds.values())).T
    width = upper-lower
    rng = np.random.default_rng(config["calibration_seed"])
    bank = np.vstack([np.array([DEFAULTS[model][name] for name in names]),
                      lower+rng.random((256,len(names)))*width])
    scores = objective(model, bank, names, data, config)
    selected = np.argsort(scores, kind="stable")[:config["calibration_restarts"]]
    current = bank[selected]; losses = scores[selected]
    step = np.full(len(current), .15)
    evaluated = len(bank)
    history = []
    for iteration in range(config["calibration_rounds"]):
        bank = [current]
        for axis in range(len(names)):
            delta = np.zeros_like(current); delta[:,axis] = width[axis]*step
            bank.extend([np.clip(current+delta,lower,upper),np.clip(current-delta,lower,upper)])
        proposals = np.stack(bank, axis=1)
        proposal_losses = objective(model, proposals.reshape(-1,len(names)), names, data, config).reshape(proposals.shape[:2])
        evaluated += int(proposal_losses.size)
        winner = proposal_losses.argmin(axis=1)
        improved = proposal_losses[np.arange(len(current)),winner] < losses-1e-12
        current = proposals[np.arange(len(current)),winner]
        losses = proposal_losses[np.arange(len(current)),winner]
        step = np.where(improved, step, step*.65)
        history.append(dict(iteration=iteration, best_loss=float(losses.min())))
    best = int(losses.argmin())
    params = dict(DEFAULTS[model]);params.update(dict(zip(names,map(float,current[best]))))
    return params, dict(model=model, calibrated_parameter_names=names, bounds=bounds,
                        objective=float(losses[best]), baseline_objective=float(scores[0]),
                        evaluated_candidates=evaluated, history=history,
                        boundary_parameters=[name for name,value,lo,hi in zip(names,current[best],lower,upper)
                                             if min(abs(value-lo),abs(value-hi)) < 1e-4*(hi-lo)],
                        mask="relative_time_s < 30; no validation follower targets accessed")
