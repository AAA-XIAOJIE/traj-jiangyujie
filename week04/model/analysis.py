"""Reproducible fixed-scope experiment, including scale and quality diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .measures import (trajectory_segments, windows, edie, detector, spatial_fields,
                       average_field, clip_space)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["edie", "detector", "voronoi", "kernel"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")


def write_table(frame, path):
    """Stable LF line endings keep evidence hashes valid after a Git checkout."""
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


def compute(out):
    config = json.loads((ROOT/"config.json").read_text(encoding="utf8"))
    manifest = json.loads((ROOT/"data/manifest.json").read_text(encoding="utf8"))
    source = ROOT/"data"/manifest["subset_file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["subset_sha256"]:
        raise ValueError("Input subset checksum changed")
    data = pd.read_csv(source)
    if data.isna().any().any() or data.duplicated(["vehicle_id", "frame_id"]).any():
        raise ValueError("Missing or duplicate records must be resolved explicitly")
    left, right = config["x_min_m"], config["x_max_m"]
    start, end = config["time_start_s"], config["time_end_s"]
    segments = trajectory_segments(data, config["lane"], config["quadrature_substeps"])
    pieces = clip_space(segments, left, right)
    pieces = pieces[(pieces.t1 > start) & (pieces.t0 < end)].copy()
    pieces["t0"] = pieces.t0.clip(lower=start)
    pieces["t1"] = pieces.t1.clip(upper=end)
    step = config["sample_step_s"] / config["quadrature_substeps"]
    fields = spatial_fields(segments, left, right, config["kernel_halfwidths_m"], step=step)
    output = []
    for T in config["windows_s"]:
        bounds = windows(T, start, end)
        a = edie(segments, left, right, bounds)
        b, events = detector(segments, left, right, bounds, config["detector_x_m"])
        c = average_field(fields, "voronoi", "voronoi", left, right, bounds)
        d = average_field(fields, f"kernel_{config['kernel_halfwidth_m']:g}", "kernel", left, right, bounds)
        result = pd.concat([a, b, c, d], ignore_index=True)
        result["window_s"] = T
        output.append(result)
    points = pd.concat(output, ignore_index=True)
    write_table(points, out/"fundamental_points.csv")
    write_table(events, out/"detector_crossings.csv")
    records = []
    for (method, T), group in points.groupby(["method", "window_s"], sort=False):
        valid = group[np.isfinite(group.k_veh_km) & np.isfinite(group.q_veh_h)]
        weights = valid.duration_s.to_numpy()
        kmean = np.average(valid.k_veh_km, weights=weights)
        qmean = np.average(valid.q_veh_h, weights=weights)
        ref = points[(points.method == "edie") & (points.window_s == T)]
        paired = valid.merge(ref, on="start_s", suffixes=("", "_reference"))
        records.append(dict(method=method, window_s=T, windows=len(group), valid_windows=len(valid),
                            mean_k_veh_km=kmean, mean_q_veh_h=qmean, mean_v_km_h=qmean/kmean,
                            k_min=valid.k_veh_km.min(), k_max=valid.k_veh_km.max(),
                            q_min=valid.q_veh_h.min(), q_max=valid.q_veh_h.max(),
                            q_sd=valid.q_veh_h.std(ddof=1), k_sd=valid.k_veh_km.std(ddof=1),
                            q_mae_vs_edie=np.mean(abs(paired.q_veh_h-paired.q_veh_h_reference)),
                            k_mae_vs_edie=np.mean(abs(paired.k_veh_km-paired.k_veh_km_reference))))
    summary = pd.DataFrame(records)
    write_table(summary, out/"summary.csv")
    # Spatial subdivision is a parameter study of Edie, not a fifth measurement.
    spatial = []
    for length in config["subcell_lengths_m"]:
        for a in np.arange(left, right, length):
            r = edie(segments, a, min(a+length, right), windows(10, start, end))
            r["cell_length_m"] = length
            spatial.append(r)
    spatial = pd.concat(spatial, ignore_index=True)
    write_table(spatial, out/"spatial_sensitivity.csv")
    kernel = []
    for h in config["kernel_halfwidths_m"]:
        r = average_field(fields, f"kernel_{h:g}", "kernel", left, right, windows(30, start, end))
        r["halfwidth_m"] = h; kernel.append(r)
    write_table(pd.concat(kernel), out/"kernel_sensitivity.csv")
    shifts = []
    for T in config["windows_s"]:
        for offset in [0., T/2]:
            r = edie(segments, left, right, windows(T, start, end, offset))
            r["offset_s"], r["window_s"] = offset, T; shifts.append(r)
    write_table(pd.concat(shifts), out/"offset_sensitivity.csv")
    # Quadrature convergence: twice as many time samples, same interpolation and lane rule.
    finer = trajectory_segments(data, config["lane"], 2 * config["quadrature_substeps"])
    fine_fields = spatial_fields(finer, left, right, [config["kernel_halfwidth_m"]], step=step/2)
    convergence = []
    for method, prefix in [("voronoi", "voronoi"), ("kernel", f"kernel_{config['kernel_halfwidth_m']:g}")]:
        fine = average_field(fine_fields, prefix, method, left, right, windows(30, start, end))
        coarse = points[(points.method == method) & (points.window_s == 30)].reset_index(drop=True)
        convergence.append(dict(method=method,
                                max_k_difference=float(abs(fine.k_veh_km-coarse.k_veh_km).max()),
                                max_q_difference=float(abs(fine.q_veh_h-coarse.q_veh_h).max())))
    # Additivity verifies exact time/space clipping, including partial final windows.
    reference = points[(points.method == "edie") & (points.window_s == 10)].set_index("start_s")
    additivity = []
    for length in config["subcell_lengths_m"]:
        g = spatial[spatial.cell_length_m == length].groupby("start_s")
        for metric in ["vehicle_seconds", "vehicle_metres"]:
            error = abs(g[metric].sum()-reference[metric]).max()
            additivity.append(dict(cell_length_m=length, metric=metric, max_abs_error=float(error)))
    # The raw supplied speed column is checked but not mixed into the four measurements.
    speed_diagnostic = []
    for a, b in windows(30, start, end):
        dt = np.maximum(0., np.minimum(pieces.t1, b)-np.maximum(pieces.t0, a))
        q_geometry = float(np.dot(dt, pieces.velocity)/((right-left)*(b-a))*3600)
        q_column = float(np.dot(dt, pieces.reported_velocity)/((right-left)*(b-a))*3600)
        speed_diagnostic.append(dict(start_s=a, q_geometry=q_geometry, q_speed_column=q_column))
    write_table(pd.DataFrame(speed_diagnostic), out/"speed_definition_check.csv")
    gap = data.groupby("vehicle_id").time_s.diff()
    changes = data.lane_id.ne(data.groupby("vehicle_id").lane_id.shift()) & gap.notna()
    spikes = pieces[pieces.velocity > 40]
    raw_intervals = set(zip(spikes.vehicle_id, np.floor((spikes.t0+1e-7)*10).astype(int)))
    physical = (pieces.t1-pieces.t0).to_numpy()
    audit = dict(input=manifest, analysis_config=config,
                 subset_missing=int(data.isna().sum().sum()), subset_duplicate_keys=int(data.duplicated(["vehicle_id", "frame_id"]).sum()),
                 long_gaps=int((gap > .100001).sum()), lane_transitions_in_support=int(changes.sum()),
                 roi_unique_vehicles=int(pieces.vehicle_id.nunique()),
                 roi_vehicle_seconds=float(physical.sum()), roi_vehicle_metres=float(np.dot(physical,pieces.velocity)),
                 roi_speed_derivative_max_m_s=float(pieces.velocity.max()),
                 roi_intervals_above_40_m_s=len(raw_intervals),
                 roi_anomaly_vehicle_ids=sorted(int(x) for x in spikes.vehicle_id.unique()),
                 anomaly_policy="retained, no silent filtering or smoothing; counts and IDs in this audit, velocity comparison in quality.png",
                 geometric_vs_reported_speed_mae_m_s=float(np.average(abs(pieces.velocity-pieces.reported_velocity), weights=physical)),
                 voronoi_time_samples=len(fields), voronoi_invalid_time_samples=int(fields.voronoi_k.isna().sum()),
                 voronoi_invalid_in_analysis=int(fields[(fields.t0 >= start) & (fields.t1 <= end)].voronoi_k.isna().sum()),
                 quadrature_convergence=convergence, spatial_additivity=additivity,
                 time_mean_edie_k_range=float(summary[summary.method=="edie"].mean_k_veh_km.max()-summary[summary.method=="edie"].mean_k_veh_km.min()),
                 time_mean_edie_q_range=float(summary[summary.method=="edie"].mean_q_veh_h.max()-summary[summary.method=="edie"].mean_q_veh_h.min()),
                 full_observation_window_s=899.9,
                 common_support="0 <= t < 840 s; all four methods valid; complete 10/30/60 s windows; source 883.6-899.9 s lacks Voronoi guards",
                 unused_tail_s=59.9,
                 final_windows="Primary comparison has no partial windows; offset sensitivity uses its recorded actual durations")
    write_json(out/"audit.json", audit)
    print(summary.to_string(index=False), flush=True)
    print("Audit:", {k:audit[k] for k in ["roi_unique_vehicles", "roi_intervals_above_40_m_s", "voronoi_invalid_time_samples", "quadrature_convergence"]}, flush=True)
    return data, segments, points, summary, audit
