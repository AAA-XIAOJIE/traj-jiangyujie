"""Four traffic measurements with explicit spatial and temporal support.

Internal units: metres, seconds, vehicles. Public output: veh/km, veh/h, km/h.
Input coordinates are the NGSIM front-centre reference points, not vehicle bodies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trajectory_segments(data, lane=2, substeps=2):
    """Linear pieces, no interpolation across missing observations.

    At a lane-label change, membership changes halfway through the 0.1 s
    observation interval. Even substeps preserve that explicitly assumed instant.
    """
    if substeps < 2 or substeps % 2:
        raise ValueError("Use an even number of substeps to preserve lane changes")
    d = data.sort_values(["vehicle_id", "time_s"]).reset_index(drop=True)
    next_row = d.shift(-1)
    dt = next_row.time_s.to_numpy() - d.time_s.to_numpy()
    keep = (d.vehicle_id == next_row.vehicle_id).to_numpy() & np.isclose(dt, .1, atol=1e-7)
    keep &= ((d.lane_id == lane) | (next_row.lane_id == lane)).to_numpy()
    a, b = d.loc[keep], next_row.loc[keep]
    start, finish = a.time_s.to_numpy(), b.time_s.to_numpy()
    x, dx = a.local_y_m.to_numpy(), b.local_y_m.to_numpy() - a.local_y_m.to_numpy()
    velocity = dx / (finish - start)
    if (velocity < -1e-7).any():
        raise ValueError("Backward trajectories require a signed-flow policy")
    frames = []
    for j in range(substeps):
        f0, f1 = j / substeps, (j + 1) / substeps
        member = (a.lane_id if j < substeps // 2 else b.lane_id).to_numpy() == lane
        frames.append(pd.DataFrame({
            "vehicle_id": a.vehicle_id.to_numpy()[member].astype(int),
            "t0": (start + f0 * (finish - start))[member],
            "t1": (start + f1 * (finish - start))[member],
            "x0": (x + f0 * dx)[member], "x1": (x + f1 * dx)[member],
            "velocity": velocity[member],
            "reported_velocity": (a.speed_m_s.to_numpy() + (f0 + f1) / 2 *
                                   (b.speed_m_s.to_numpy() - a.speed_m_s.to_numpy()))[member],
        }))
    return pd.concat(frames, ignore_index=True)


def clip_space(segments, left, right):
    """Exact clipping of linear pieces to [left, right); retain stopped vehicles."""
    if right <= left:
        raise ValueError("Spatial length must be positive")
    x0, x1 = segments.x0.to_numpy(), segments.x1.to_numpy()
    dx = x1 - x0
    moving = dx > 1e-12
    denom = np.where(moving, dx, 1.)
    lo = np.where(moving, np.maximum(0., (left - x0) / denom), 0.)
    hi = np.where(moving, np.minimum(1., (right - x0) / denom), 1.)
    good = (hi > lo + 1e-12) & (moving | ((x0 >= left) & (x0 < right)))
    result = segments.loc[good].copy()
    t0 = segments.t0.to_numpy(); dt = segments.t1.to_numpy() - t0
    result["t0"] = (t0 + lo * dt)[good]
    result["t1"] = (t0 + hi * dt)[good]
    result["x0"] = (x0 + lo * dx)[good]
    result["x1"] = (x0 + hi * dx)[good]
    return result


def windows(duration, start, end, offset=0.):
    if duration <= 0 or end <= start or offset < 0:
        raise ValueError("Invalid window geometry")
    lower = np.arange(start + offset, end - 1e-9, duration)
    return [(float(a), float(min(a + duration, end))) for a in lower]


def record(method, a, b, left, right, k, q, n, coverage=1., **extra):
    return dict(method=method, start_s=a, end_s=b, duration_s=b-a,
                x_min_m=left, x_max_m=right, length_m=right-left,
                k_veh_km=k*1000, q_veh_h=q*3600,
                v_km_h=q/k*3.6 if np.isfinite(k) and k > 0 else np.nan,
                samples=n, coverage=coverage, **extra)


def edie(segments, left, right, bounds, method="edie"):
    pieces = clip_space(segments, left, right)
    s, e, v = (pieces[c].to_numpy() for c in ["t0", "t1", "velocity"])
    ids = pieces.vehicle_id.to_numpy()
    result = []
    for a, b in bounds:
        dt = np.maximum(0., np.minimum(e, b) - np.maximum(s, a))
        area = (right - left) * (b - a)
        time, distance = dt.sum(), np.dot(dt, v)
        result.append(record(method, a, b, left, right, time/area, distance/area,
                             int(np.unique(ids[dt > 1e-10]).size),
                             vehicle_seconds=time, vehicle_metres=distance))
    return pd.DataFrame(result)


def crossings(segments, point):
    # A passage is owned by its incoming piece; a stopped piece creates no event.
    valid = (segments.x0 < point) & (segments.x1 >= point) & (segments.velocity > 0)
    s = segments.loc[valid]
    return pd.DataFrame({"time_s": s.t0 + (point - s.x0) / s.velocity,
                         "velocity": s.velocity, "vehicle_id": s.vehicle_id})


def detector(segments, left, right, bounds, point=None):
    point = (left+right)/2 if point is None else point
    events = crossings(segments, point)
    result = []
    for a, b in bounds:
        current = events[(events.time_s >= a-1e-9) & (events.time_s < b-1e-9)]
        n = len(current); q = n/(b-a)
        # Harmonic passage speed; without passages a queue's density is unknown.
        k = (1/current.velocity).sum()/(b-a) if n else np.nan
        result.append(record("detector", a, b, left, right, k, q, n,
                             detector_x_m=point))
    return pd.DataFrame(result), events


def triangle_cdf(z):
    """CDF of the unit-mass triangular kernel supported on [-1, 1]."""
    z = np.clip(np.asarray(z, dtype=float), -1., 1.)
    return np.where(z <= 0, .5*(z+1)**2, 1-.5*(1-z)**2)


def voronoi_state(x, v, left, right):
    """Integrate full one-dimensional cells over the ROI; no boundary ghosts."""
    order = np.argsort(x); x, v = np.asarray(x)[order], np.asarray(v)[order]
    if len(x) < 3:
        return np.nan, np.nan, 0.
    mid = (x[1:] + x[:-1])/2
    lo, hi = mid[:-1], mid[1:]
    length = hi-lo
    overlap = np.maximum(0., np.minimum(hi, right)-np.maximum(lo, left))
    # Edge agents do not have two observed neighbours. The ROI must be bracketed.
    coverage = overlap.sum()/(right-left)
    if (length <= 1e-10).any() or coverage < 1-1e-8:
        return np.nan, np.nan, coverage
    fractions = overlap/length
    return fractions.sum()/(right-left), np.dot(fractions, v[1:-1])/(right-left), coverage


def kernel_state(x, v, left, right, halfwidth):
    if halfwidth <= 0:
        raise ValueError("Kernel bandwidth must be positive")
    mass = triangle_cdf((right-x)/halfwidth) - triangle_cdf((left-x)/halfwidth)
    return mass.sum()/(right-left), np.dot(mass, v)/(right-left)


def spatial_fields(segments, left, right, halfwidths, step=.05):
    """Midpoint time quadrature over linear, fixed-lane pieces."""
    slots = np.rint(segments.t0.to_numpy()/step).astype(int)
    order = np.argsort(slots, kind="stable"); slot = slots[order]
    x = ((segments.x0.to_numpy()+segments.x1.to_numpy())/2)[order]
    v = segments.velocity.to_numpy()[order]
    borders = np.r_[0, np.flatnonzero(np.diff(slot))+1, len(slot)]
    result = []
    for i, j in zip(borders[:-1], borders[1:]):
        xx, vv = x[i:j], v[i:j]
        k, q, coverage = voronoi_state(xx, vv, left, right)
        row = dict(t0=slot[i]*step, t1=(slot[i]+1)*step,
                   voronoi_k=k, voronoi_q=q, voronoi_coverage=coverage)
        for h in halfwidths:
            k, q = kernel_state(xx, vv, left, right, h)
            row[f"kernel_{h:g}_k"], row[f"kernel_{h:g}_q"] = k, q
        result.append(row)
    return pd.DataFrame(result)


def average_field(fields, prefix, method, left, right, bounds):
    s, e = fields.t0.to_numpy(), fields.t1.to_numpy()
    k, q = fields[prefix+"_k"].to_numpy(), fields[prefix+"_q"].to_numpy()
    valid = np.isfinite(k) & np.isfinite(q)
    result = []
    for a, b in bounds:
        dt = np.maximum(0., np.minimum(e, b)-np.maximum(s, a))
        coverage = dt[valid].sum()/(b-a)
        kk = np.dot(dt[valid], k[valid])/(b-a) if coverage > 1-1e-8 else np.nan
        qq = np.dot(dt[valid], q[valid])/(b-a) if coverage > 1-1e-8 else np.nan
        result.append(record(method, a, b, left, right, kk, qq,
                             int(((dt > 1e-10) & valid).sum()), coverage))
    return pd.DataFrame(result)


def local_trend(k, q, bandwidth=20., minimum=4):
    """Descriptive local linear fit on observed density support only, not capacity."""
    k, q = np.asarray(k), np.asarray(q)
    good = np.isfinite(k) & np.isfinite(q); k, q = k[good], q[good]
    grid = np.linspace(k.min(), k.max(), 100) if len(k) else np.array([])
    fitted = np.full(len(grid), np.nan)
    for i, x in enumerate(grid):
        d = (k-x)/bandwidth; use = abs(d) < 1
        if use.sum() < minimum:
            continue
        weight = (1-d[use]**2)**2
        design = np.column_stack([np.ones(use.sum()), k[use]-x])
        fitted[i] = np.linalg.lstsq(design*np.sqrt(weight[:, None]),
                                    q[use]*np.sqrt(weight), rcond=None)[0][0]
    return grid, fitted
