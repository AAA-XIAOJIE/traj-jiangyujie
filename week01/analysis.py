"""A1 geometry only; no behavioural labels are inferred from overlaps."""
from __future__ import annotations

import numpy as np
import pandas as pd


def contiguous_groups(track: pd.DataFrame, spec: dict):
    """Adapted from the PEM-Flow figure script: sort first, never bridge a gap.

    ETH uses its native exact frame step. UCY has irregular control points;
    an explicitly configured long-interval guard is a display rule, not an
    annotation that a tracking failure occurred.
    """
    track = track.sort_values("frame_id", kind="stable")
    delta = np.diff(track.frame_id.to_numpy())
    expected = spec.get("expected_step_frames")
    allowed = (delta == expected) if expected is not None else ((delta > 0) & (delta / spec["fps"] <= spec["max_link_s"] + 1e-9))
    for index in np.split(np.arange(len(track)), np.flatnonzero(~allowed) + 1):
        if len(index):
            yield track.iloc[index]


def build_geometry(df: pd.DataFrame, spec: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return within-person edges, endpoint times, per-track audit and breaks."""
    segments, times, audit, breaks = [], [], [], []
    for track_id, track in df.groupby("track_id", sort=True):
        groups = list(contiguous_groups(track, spec))
        count = 0
        distance = 0.0
        for group in groups:
            xy = group[["x_m", "y_m"]].to_numpy()
            t = group.time_s.to_numpy()
            if len(group) >= 2:
                edge = np.stack([xy[:-1], xy[1:]], axis=1)
                segments.extend(edge)
                times.extend(np.column_stack([t[:-1], t[1:]]))
                count += len(edge)
                distance += float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        for left, right in zip(groups[:-1], groups[1:]):
            a, b = left.iloc[-1], right.iloc[0]
            breaks.append({"scene": spec["name"], "track_id": int(track_id), "left_frame": int(a.frame_id), "right_frame": int(b.frame_id), "interval_s": float(b.time_s - a.time_s), "distance_m": float(np.hypot(b.x_m - a.x_m, b.y_m - a.y_m)), "reason": "non_native_step" if spec["family"] == "ETH" else "control_interval_over_guard"})
        audit.append({"scene": spec["name"], "track_id": int(track_id), "annotation_points": len(track), "start_s": float(track.time_s.min()), "end_s": float(track.time_s.max()), "duration_s": float(track.time_s.max() - track.time_s.min()), "components": len(groups), "rendered_edges": count, "linked_path_length_m": distance})
    edges = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    endpoint_times = np.asarray(times, dtype=float).reshape(-1, 2)
    break_columns = ["scene", "track_id", "left_frame", "right_frame", "interval_s", "distance_m", "reason"]
    return edges, endpoint_times, pd.DataFrame(audit), pd.DataFrame(breaks, columns=break_columns)


def clip_segments(edges: np.ndarray, times: np.ndarray, start: float, end: float):
    """Clip accepted geometric links to a time window without extrapolation."""
    if end <= start:
        raise ValueError("Window end must exceed start")
    keep = (times[:, 1] > start) & (times[:, 0] < end)
    edge, t = edges[keep], times[keep]
    if not len(edge):
        return edge.copy(), np.empty(0)
    t0 = np.maximum(t[:, 0], start)
    t1 = np.minimum(t[:, 1], end)
    weight0 = (t0 - t[:, 0]) / (t[:, 1] - t[:, 0])
    weight1 = (t1 - t[:, 0]) / (t[:, 1] - t[:, 0])
    delta = edge[:, 1] - edge[:, 0]
    cropped = np.stack([edge[:, 0] + weight0[:, None] * delta, edge[:, 0] + weight1[:, None] * delta], axis=1)
    return cropped, (t0 + t1) / 2 - start


def select_window(df: pd.DataFrame, spec: dict, width: float) -> dict:
    """Choose the earliest 12 s window with most tracks having an accepted link.

    Candidates start at whole seconds within annotation coverage. This is a
    display selection criterion; the overview always retains the full data.
    """
    starts = np.arange(0, max(0, np.floor(df.time_s.max() - width)) + 1)
    counts = np.zeros(len(starts), dtype=int)
    for _, track in df.groupby("track_id", sort=True):
        present = np.zeros(len(starts), dtype=bool)
        for group in contiguous_groups(track, spec):
            if len(group) >= 2:
                present |= (group.time_s.iloc[-1] > starts) & (group.time_s.iloc[0] < starts + width)
        counts += present
    chosen = int(np.argmax(counts))
    return {"start_s": float(starts[chosen]), "end_s": float(starts[chosen] + width), "tracks_with_visible_link": int(counts[chosen]), "criterion": "maximum tracks with accepted links; integer-second grid; earliest tie"}


def summarize(df, spec, edges, tracks, breaks, window):
    singletons = sum(len(g) == 1 for _, tr in df.groupby("track_id") for g in contiguous_groups(tr, spec))
    return {
        "scene": spec["name"], "label": spec["label"], "family": spec["family"],
        "sampling": spec["sampling"], "annotation_points": len(df),
        "tracks": int(df.track_id.nunique()), "first_frame": int(df.frame_id.min()),
        "last_frame": int(df.frame_id.max()), "coverage_s": float(df.time_s.max()),
        "fps": spec["fps"], "rendered_edges": len(edges),
        "unlinked_long_intervals": len(breaks), "singleton_components": singletons,
        "x_min_m": float(df.x_m.min()), "x_max_m": float(df.x_m.max()),
        "y_min_m": float(df.y_m.min()), "y_max_m": float(df.y_m.max()),
        "window": window,
    }
