"""Read the supplied experiment and evaluate observed/simulated trajectories."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def read_experiment(path, fps=25.0, scale=0.01):
    raw = np.loadtxt(path, ndmin=2)
    if raw.shape[1] != 5 or not np.isfinite(raw).all():
        raise ValueError("Expected finite ID, frame, x, y, height columns")
    if fps <= 0 or scale <= 0 or not np.equal(raw[:, :2], np.floor(raw[:, :2])).all():
        raise ValueError("Invalid frame/ID or time/length conversion")
    ids, frames = np.unique(raw[:, 0]).astype(int), np.unique(raw[:, 1]).astype(int)
    if not np.all(np.diff(frames) == 1):
        raise ValueError("Missing common frames; do not silently bridge gaps")
    positions = np.empty((len(frames), len(ids), 2))
    for k, identity in enumerate(ids):
        rows = raw[raw[:, 0] == identity]
        rows = rows[np.argsort(rows[:, 1])]
        if not np.array_equal(rows[:, 1], frames):
            raise ValueError("Incomplete/duplicate individual frames")
        positions[:, k] = rows[:, 2:4] * scale
    initial = positions[0]
    fit = np.linalg.lstsq(np.column_stack([2 * initial, np.ones(len(ids))]),
                          np.sum(initial**2, axis=1), rcond=None)[0]
    center = fit[:2]
    fitted_radius = float(np.sqrt(fit[2] + center @ center))
    return {"positions": positions, "times": (frames - frames[0]) / fps,
            "ids": ids, "frames": frames, "center": center,
            "radius": fitted_radius, "goals": 2 * center - initial,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def velocity(positions, dt, half_window=5):
    """Centered 0.4 s displacement at 25 Hz; shorter one-sided endpoints."""
    k = np.arange(len(positions))
    left, right = np.maximum(0, k - half_window), np.minimum(len(k) - 1, k + half_window)
    return (positions[right] - positions[left]) / ((right - left) * dt)[:, None, None]


def initial_conditions(experiment, config):
    """Use initial positions and the stated first 2 s calibration only."""
    p, times = experiment["positions"], experiment["times"]
    dt = times[1] - times[0]
    k = int(round(config["initial_velocity_window_s"] / dt))
    initial_velocity = (p[k] - p[0]) / (k * dt)
    # Segment velocities whose TWO endpoints are inside [1 s, 2 s].
    start, stop = int(round(1.0 / dt)), int(round(config["calibration_end_s"] / dt))
    stride = 5
    speeds = np.linalg.norm((p[start + stride:stop + 1] - p[start:stop + 1 - stride]) / (stride * dt), axis=-1)
    desired = np.clip(np.quantile(speeds, 0.75, axis=0), 1.0, 3.2)
    return p[0].copy(), initial_velocity, desired


def evaluate(positions, times, experiment, body_radius=0.25, goal_tolerance=0.5):
    dt = times[1] - times[0]
    center, goals = experiment["center"], experiment["goals"]
    v = velocity(positions, dt)
    speed = np.linalg.norm(v, axis=-1)
    radial = np.linalg.norm(positions - center, axis=-1)
    distance_goal = np.linalg.norm(positions - goals, axis=-1)
    arrived = distance_goal <= goal_tolerance
    arrival_times = np.array([times[np.flatnonzero(arrived[:, i])[0]] if arrived[:, i].any() else np.nan
                              for i in range(positions.shape[1])])
    inward = goals - positions[0]
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    right_axis = np.column_stack([inward[:, 1], -inward[:, 0]])
    lateral = np.einsum("tij,ij->ti", positions - positions[0], right_axis)
    side = np.mean(lateral, axis=0)
    # Common 0.2 s spatial sampling reduces native tracking jitter.
    samples = np.unique(np.r_[np.arange(0, len(positions), 5), len(positions) - 1])
    length = np.linalg.norm(np.diff(positions[samples], axis=0), axis=-1).sum(axis=0)
    straight = np.linalg.norm(goals - positions[0], axis=-1)
    n = positions.shape[1]
    ai, bi = np.triu_indices(n, 1)
    relative = positions[:, ai] - positions[:, bi]
    distances = np.linalg.norm(relative, axis=-1)
    # Swept closest approach in each integration interval, catches tunneling.
    delta = np.diff(relative, axis=0)
    fraction = np.clip(-np.sum(relative[:-1] * delta, axis=-1) / np.maximum(np.sum(delta**2, axis=-1), 1e-20), 0, 1)
    swept = np.linalg.norm(relative[:-1] + fraction[..., None] * delta, axis=-1)
    overlaps = swept < 2 * body_radius - 1e-5
    closest = np.argmin(radial, axis=0)
    core_speed = np.array([speed[max(0, k - 10):min(len(times), k + 11), i].mean() for i, k in enumerate(closest)])
    approach_speed = speed[(times >= 1.6) & (times <= 2.6)].mean(axis=0)
    turn = np.arccos(np.clip(np.einsum("tij,ij->ti", v, inward) / np.maximum(speed, 1e-12), -1, 1))
    mask = (speed > 0.3) & (distance_goal > 1)
    trajectory_error = np.linalg.norm(positions - experiment["positions"], axis=-1)
    stats = {
        "people": n, "duration_s": float(times[-1]),
        "mean_path_m": float(length.mean()), "median_detour_ratio": float(np.median(length / straight)),
        "mean_speed_mps": float(speed.mean()),
        "min_mean_radius_m": float(radial.mean(axis=1).min()),
        "time_min_mean_radius_s": float(times[np.argmin(radial.mean(axis=1))]),
        "right_fraction": float(np.mean(side > 0.1)), "left_fraction": float(np.mean(side < -0.1)),
        "right_fraction_zero_threshold": float(np.mean(side > 0)),
        "neutral_fraction": float(np.mean(abs(side) <= 0.1)),
        "median_max_lateral_m": float(np.median(np.max(abs(lateral), axis=0))),
        "approach_speed_mps": float(approach_speed.mean()), "closest_approach_speed_mps": float(core_speed.mean()),
        "slower_at_closest_fraction": float(np.mean(core_speed < approach_speed)),
        "mean_heading_deviation_deg": float(np.rad2deg(turn[mask]).mean()),
        "arrived_by_end": int(np.isfinite(arrival_times).sum()),
        "median_arrival_s_completed_only": float(np.nanmedian(arrival_times)) if np.isfinite(arrival_times).any() else None,
        "min_swept_center_distance_m": float(swept.min()),
        "disk_overlap_pair_intervals": int(overlaps.sum()),
        "disk_overlap_unique_pairs": int(np.any(overlaps, axis=0).sum()),
        "ade_after_calibration_m": float(trajectory_error[times > 2.0].mean()),
        "fde_m": float(trajectory_error[-1].mean()),
    }
    series = pd.DataFrame({"time_s": times, "mean_speed_mps": speed.mean(axis=1),
                           "mean_radius_m": radial.mean(axis=1), "minimum_spacing_m": distances.min(axis=1),
                           "people_within_3m": np.sum(radial < 3, axis=1),
                           "arrived_fraction": np.mean(np.maximum.accumulate(arrived, axis=0), axis=1)})
    individual = pd.DataFrame({"id": experiment["ids"], "path_m": length, "detour_ratio": length / straight,
                               "signed_lateral_mean_m": side, "max_lateral_m": np.max(abs(lateral), axis=0),
                               "approach_speed_mps": approach_speed, "closest_speed_mps": core_speed,
                               "arrival_s": arrival_times, "ade_m": trajectory_error.mean(axis=0),
                               "fde_m": trajectory_error[-1]})
    return {"stats": stats, "series": series, "individual": individual, "velocity": v, "speed": speed,
            "lateral": lateral, "radial": radial, "positions": positions}
