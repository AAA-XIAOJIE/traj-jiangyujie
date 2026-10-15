"""Original OVM/FVDM/IDM dynamics, with explicit SI conventions.

The comparison's OVM and FVDM share Jiang et al. (2001), Eq. (5)'s
dimensional optimal-velocity function. Bando's Eq. (24) is also exposed
separately in its original scaled coordinates, never labelled as SI.
No desired-speed or IDM desired-gap clipping is hidden in these formulas.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    "OVM": dict(alpha=.41, V1=6.75, V2=7.91, C1=.13, C2=1.57),
    "FVDM": dict(alpha=.41, V1=6.75, V2=7.91, C1=.13, C2=1.57,
                 **{"lambda": .5, "cutoff_m": 100.}),
    "IDM": dict(v0=120/3.6, T=1.6, a=.73, b=1.67, s0=2., s1=0., delta=4.),
}


def bando_velocity(headway_scaled):
    """Bando 1995 Eq. (24), in the original scaled position/speed units."""
    return np.tanh(np.asarray(headway_scaled)-2.) + np.tanh(2.)


def optimal_velocity(gap, p):
    """Jiang Eq. (5): h-l_lead=net gap, in m; output in m/s."""
    return p["V1"] + p["V2"]*np.tanh(p["C1"]*np.asarray(gap)-p["C2"])


def idm_desired_gap(speed, leader_speed, p):
    """Treiber 2000 Eq. (7), including s1 and WITHOUT a modern max() variant."""
    speed = np.asarray(speed)
    if np.any(speed < 0):
        raise ValueError("IDM requires a nonnegative speed state")
    closing = speed - np.asarray(leader_speed)
    return (p["s0"] + p.get("s1", 0.)*np.sqrt(speed/p["v0"]) +
            p["T"]*speed + speed*closing/(2*np.sqrt(p["a"]*p["b"])))


def acceleration(model, gap, speed, leader_speed, p, leader_length=5.):
    """dv/dt in m/s²; gap is net bumper gap; closing speed is v-v_lead."""
    gap, speed, leader_speed = np.broadcast_arrays(gap, speed, leader_speed)
    if np.any(gap <= 0):
        raise ValueError("Nonpositive net gap is a collision, not a valid input")
    if model == "OVM":
        return p["alpha"]*(optimal_velocity(gap, p)-speed)
    if model == "FVDM":
        gain = np.where(gap+leader_length <= p.get("cutoff_m", 100.), p["lambda"], 0.)
        return p["alpha"]*(optimal_velocity(gap, p)-speed) + gain*(leader_speed-speed)
    if model == "IDM":
        desired = idm_desired_gap(speed, leader_speed, p)
        return p["a"]*(1-(speed/p["v0"])**p["delta"]-(desired/gap)**2)
    raise ValueError(f"Unknown model: {model}")
