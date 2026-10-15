"""Prescribed-leader, free-rollout follower simulation with explicit failures."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .models import acceleration


@dataclass
class Leader:
    time: np.ndarray
    position: np.ndarray
    speed: np.ndarray
    length: float
    kind: str = "observed"
    initial_speed: float = 0.
    brake_start: float = 5.
    braking: float = 1.5

    def at(self, time):
        if self.kind == "observed":
            # Both integrations use the right-hand derivative at observation knots.
            # Snap only roundoff near a knot, not a genuine pre-knot instant.
            j = min(max(int(np.searchsorted(self.time, time+1e-9, side="right")-1), 0), len(self.time)-2)
            velocity = (self.position[j+1]-self.position[j])/(self.time[j+1]-self.time[j])
            return float(np.interp(time, self.time, self.position)), float(velocity)
        if self.kind == "constant":
            return float(self.position[0]+self.initial_speed*time), self.initial_speed
        if self.kind == "braking":
            before = min(time, self.brake_start)
            since = max(0., time-self.brake_start)
            moving = min(since, self.initial_speed/self.braking)
            x = self.position[0] + self.initial_speed*before + self.initial_speed*moving-.5*self.braking*moving**2
            return float(x), max(0., self.initial_speed-self.braking*since)
        raise ValueError(self.kind)


def from_observations(data, kind="observed", braking=1.5):
    t = data.relative_time_s.to_numpy()
    return Leader(t-t[0], data.leader_x_m.to_numpy(), data.leader_v_raw_m_s.to_numpy(),
                  float(data.leader_length_m.iloc[0]), kind,
                  float(data.leader_v_raw_m_s.iloc[0]), braking=braking)


def ballistic_step(position, speed, accel, dt):
    """Author-recommended stopping event: exact stop position, no reverse motion."""
    if speed+accel*dt < 0 and accel < 0:
        return position-speed*speed/(2*accel), 0., True
    return position+speed*dt+.5*accel*dt*dt, speed+accel*dt, False


def simulate(model, params, times, leader, initial_position, initial_speed, dt=.05):
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0) or dt <= 0:
        raise ValueError("Use increasing output times starting at 0 and a positive step")
    x, v, t = float(initial_position), float(initial_speed), 0.
    rows = []; stop_requests = 0; stop_events = 0
    first_request = None; collision = None; failure = None
    min_gap = math.inf; min_proposed_gap = math.inf
    minimum_accel = math.inf; maximum_accel = -math.inf
    initial_gap = leader.at(0.)[0]-x-leader.length
    if initial_gap <= 0:
        failure = "collision"; collision = 0.; min_gap = initial_gap
    if not all(math.isfinite(z) for z in (x, v, initial_gap)) or v < 0:
        failure = "invalid_initial_state"
    for target in times:
        while t < target-1e-10 and failure is None:
            step = min(dt, target-t)
            lx, lv = leader.at(t)
            gap = lx-x-leader.length
            min_gap = min(min_gap, gap)
            if gap <= 0:
                collision = t; failure = "collision"; break
            a = float(acceleration(model, gap, v, lv, params, leader.length))
            if not math.isfinite(a) or abs(a) > 1e6:
                failure = "nonfinite_or_explosive_acceleration"; break
            minimum_accel = min(minimum_accel, a); maximum_accel = max(maximum_accel, a)
            xx, vv, stopped = ballistic_step(x, v, a, step)
            if stopped:
                stop_requests += 1
                if v > 1e-8: stop_events += 1
                if first_request is None: first_request = t
            next_lx, _ = leader.at(t+step)
            next_gap = next_lx-xx-leader.length
            min_proposed_gap = min(min_proposed_gap, next_gap)
            if next_gap <= 0:
                fraction = gap/(gap-next_gap)
                collision = t+step*fraction
                failure = "collision"
                # Stop at the estimated first contact; never continue through the leader.
                t = collision; min_gap = min(min_gap, 0.)
                break
            min_gap = min(min_gap, next_gap)
            x, v, t = xx, vv, t+step
        lx, lv = leader.at(float(target))
        if failure is None:
            gap = lx-x-leader.length; min_gap = min(min_gap, gap)
            a = float(acceleration(model, gap, v, lv, params, leader.length))
            rows.append(dict(time_s=target, leader_x_m=lx, leader_v_m_s=lv,
                             x_m=x, v_m_s=v, gap_m=gap, a_m_s2=a))
        else:
            rows.append(dict(time_s=target, leader_x_m=lx, leader_v_m_s=lv,
                             x_m=np.nan, v_m_s=np.nan, gap_m=np.nan, a_m_s2=np.nan))
    metadata = dict(status=failure or ("requires_nonnegative_speed_boundary" if stop_requests else "ok"),
                    collision_time_s=collision, raw_negative_speed_requests=stop_requests,
                    stopping_events=stop_events, first_negative_speed_request_s=first_request,
                    minimum_gap_m=min_gap, minimum_proposed_gap_m=min_proposed_gap,
                    minimum_acceleration_m_s2=minimum_accel,
                    maximum_acceleration_m_s2=maximum_accel,
                    completed=bool(failure is None), max_step_s=dt)
    return pd.DataFrame(rows), metadata
