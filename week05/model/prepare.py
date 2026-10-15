"""Audit the classroom SI source and select one model-independent following pair.

The CSV has no preceding-vehicle identifier.  Immediate leaders are inferred at
every observed frame by sorting all lane-2 front-centre longitudinal positions.
No trajectory smoothing, fitted model, or prediction error enters selection.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = "NGSIM_I80_20050413_1600-1615_SI.csv"
SOURCE_SHA256 = "df86a98e490daba1bf6a2ff3a1ca2e346be1cf2bdaf66a037910b74f40695255"
RULE = {
    "lane_id": 2,
    "vehicle_class": 2,
    "minimum_duration_s": 60.0,
    "export_duration_s": 60.0,
    "minimum_net_gap_m": 1.0,
    "maximum_position_speed_m_s": 30.0,
    "maximum_raw_speed_m_s": 30.0,
    "maximum_abs_raw_speed_increment_m_s2": 8.0,
    "maximum_position_vs_raw_speed_rmse_m_s": 0.25,
    "sample_interval_s": 0.1,
    "calibration_duration_s": 30.0,
    "ranking": "eligible maximal stable-leader runs: duration descending, start time ascending, follower ID ascending; export first 60 seconds",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def infer_pairs(source: pd.DataFrame) -> pd.DataFrame:
    """Higher longitudinal coordinate is ahead on this eastbound road."""
    lane = source.loc[source.lane_id.eq(RULE["lane_id"])].copy()
    lane = lane.sort_values(["frame_id", "local_y_m", "vehicle_id"])
    group = lane.groupby("frame_id", sort=False)
    for original, name in [
        ("vehicle_id", "leader_id"),
        ("local_y_m", "leader_x_m"),
        ("vehicle_length_m", "leader_length_m"),
        ("speed_m_s", "leader_v_raw_m_s"),
        ("vehicle_class", "leader_class"),
    ]:
        lane[name] = group[original].shift(-1)
    lane["net_gap_m"] = lane.leader_x_m - lane.local_y_m - lane.leader_length_m
    lane = lane.sort_values(["vehicle_id", "frame_id"])
    previous = lane.shift()
    new_run = (
        lane.vehicle_id.ne(previous.vehicle_id)
        | lane.leader_id.fillna(-1).ne(previous.leader_id.fillna(-1))
        | lane.frame_id.sub(previous.frame_id).ne(1)
    )
    lane["run_id"] = new_run.cumsum()
    return lane


def candidate_metrics(run_id: int, pair: pd.DataFrame) -> dict:
    duration = float(pair.time_s.iloc[-1] - pair.time_s.iloc[0])
    row = {
        "run_id": int(run_id), "follower_id": int(pair.vehicle_id.iloc[0]),
        "leader_id": int(pair.leader_id.iloc[0]) if pair.leader_id.notna().all() else None,
        "start_s": float(pair.time_s.iloc[0]), "end_s": float(pair.time_s.iloc[-1]),
        "duration_s": duration, "samples": len(pair),
    }
    reasons = []
    if duration + 1e-8 < RULE["minimum_duration_s"]:
        reasons.append("duration_below_60s")
    if pair.leader_id.isna().any():
        reasons.append("no_observed_leader")
    if len(pair) >= 2 and pair.leader_id.notna().all():
        dt = np.diff(pair.time_s.to_numpy())
        position_v = np.r_[np.diff(pair.local_y_m) / dt, np.diff(pair.leader_x_m) / dt]
        raw_f = pair.speed_m_s.to_numpy()
        raw_l = pair.leader_v_raw_m_s.to_numpy()
        raw_mid = np.r_[(raw_f[1:] + raw_f[:-1]) / 2, (raw_l[1:] + raw_l[:-1]) / 2]
        raw_acc = np.r_[np.diff(raw_f) / dt, np.diff(raw_l) / dt]
        row.update(
            min_net_gap_m=float(pair.net_gap_m.min()),
            max_net_gap_m=float(pair.net_gap_m.max()),
            minimum_position_speed_m_s=float(position_v.min()),
            maximum_position_speed_m_s=float(position_v.max()),
            minimum_raw_speed_m_s=float(min(raw_f.min(), raw_l.min())),
            maximum_raw_speed_m_s=float(max(raw_f.max(), raw_l.max())),
            maximum_abs_raw_speed_increment_m_s2=float(abs(raw_acc).max()),
            position_vs_raw_speed_rmse_m_s=float(np.sqrt(np.mean((position_v - raw_mid) ** 2))),
        )
        if not (pair.vehicle_class.eq(2).all() and pair.leader_class.eq(2).all()):
            reasons.append("not_two_passenger_cars")
        if not np.allclose(dt, RULE["sample_interval_s"], rtol=0, atol=1e-7):
            reasons.append("temporal_gap")
        if row["min_net_gap_m"] < RULE["minimum_net_gap_m"]:
            reasons.append("net_gap_below_1m")
        if row["minimum_position_speed_m_s"] < -1e-7 or row["minimum_raw_speed_m_s"] < 0:
            reasons.append("reverse_motion_or_negative_speed")
        for key in ["maximum_position_speed_m_s", "maximum_raw_speed_m_s", "maximum_abs_raw_speed_increment_m_s2"]:
            if row[key] > RULE[key] + 1e-8:
                reasons.append(key + "_above_threshold")
        if row["position_vs_raw_speed_rmse_m_s"] > RULE["maximum_position_vs_raw_speed_rmse_m_s"] + 1e-8:
            reasons.append("position_vs_raw_speed_rmse_above_threshold")
        if pair.vehicle_length_m.nunique() != 1 or pair.leader_length_m.nunique() != 1:
            reasons.append("inconsistent_vehicle_lengths")
    row["eligible"] = not reasons
    row["exclusion_reasons"] = ";".join(reasons)
    return row


def prepare(source_zip: Path, output: Path, audit_output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    audit_output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as archive:
        raw = archive.read(SOURCE_CSV)
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError("Classroom SI CSV differs from the recorded Week04 input")
    source = pd.read_csv(io.BytesIO(raw))
    if len(source) != 1226650 or source.isna().any().any():
        raise ValueError("Unexpected source size or missing observations")
    if source.duplicated(["vehicle_id", "frame_id"]).any():
        raise ValueError("Duplicate vehicle/frame identifiers")
    if not np.allclose(source.time_s, (source.frame_id - source.frame_id.min()) * .1, rtol=0, atol=1e-7):
        raise ValueError("Frame identifiers and relative seconds disagree")
    lane = infer_pairs(source)
    candidates = pd.DataFrame([candidate_metrics(int(run_id), pair) for run_id, pair in lane.groupby("run_id", sort=True)])
    candidates = candidates.sort_values(["eligible", "duration_s", "start_s", "follower_id"], ascending=[False, False, True, True])
    candidates.to_csv(audit_output / "candidates.csv", index=False, float_format="%.9f", lineterminator="\n")
    candidates.to_csv(output / "candidates.csv", index=False, float_format="%.9f", lineterminator="\n")
    eligible = candidates.loc[candidates.eligible]
    if eligible.empty:
        raise ValueError("No pair passes the documented model-independent quality thresholds")
    selected = eligible.iloc[0]
    stable = lane.loc[lane.run_id.eq(selected.run_id)]
    start = float(selected.start_s)
    pair = stable.loc[stable.time_s.le(start + RULE["export_duration_s"] + 1e-7)].copy()
    pair = pair.rename(columns={"vehicle_id": "follower_id", "local_y_m": "follower_x_m", "speed_m_s": "follower_v_raw_m_s", "vehicle_length_m": "follower_length_m"})
    pair["relative_time_s"] = pair.time_s - start
    for who in ("follower", "leader"):
        # This diagnostic uses central differences internally and one-sided
        # differences at endpoints. Raw positions/speeds remain the data targets.
        pair[who + "_v_from_x_m_s"] = np.gradient(pair[who + "_x_m"].to_numpy(), pair.time_s.to_numpy(), edge_order=2)
    pair["leader_id"] = pair.leader_id.astype(int)
    pair["leader_identity_source"] = "same_lane_next_front_centre_in_each_frame"
    pair["phase"] = np.where(pair.relative_time_s < RULE["calibration_duration_s"] - 1e-7, "calibration", "validation")
    columns = ["time_s", "relative_time_s", "frame_id", "lane_id", "follower_id", "leader_id", "follower_x_m", "leader_x_m", "follower_v_raw_m_s", "leader_v_raw_m_s", "follower_v_from_x_m_s", "leader_v_from_x_m_s", "follower_length_m", "leader_length_m", "net_gap_m", "phase", "leader_identity_source"]
    pair = pair[columns].reset_index(drop=True)
    if len(pair) != 601 or pair.leader_id.nunique() != 1 or pair.follower_id.nunique() != 1:
        raise ValueError("Unexpected export length or pair identity change")
    pair.to_csv(output / "following_pair.csv", index=False, float_format="%.9f", lineterminator="\n")
    reason_counts = candidates.exclusion_reasons.str.split(";").explode().value_counts()
    audit = {
        "source_rows": int(len(source)), "source_vehicles": int(source.vehicle_id.nunique()),
        "missing_by_column": {key: int(value) for key, value in source.isna().sum().items()},
        "duplicate_vehicle_frame_keys": 0, "lane_2_rows": len(lane),
        "stable_leader_runs_total": len(candidates), "eligible_runs": len(eligible),
        "exclusion_reason_counts": {key: int(value) for key, value in reason_counts.items() if key},
        "selection_rule": RULE,
        "selected_maximal_run": {key: value.item() if isinstance(value, np.generic) else value for key, value in selected.to_dict().items()},
        "source_csv": SOURCE_CSV, "source_csv_sha256": actual_hash,
        "candidates_file": "candidates.csv",
        "vehicle_lengths_m": {"follower": float(pair.follower_length_m.iloc[0]), "leader": float(pair.leader_length_m.iloc[0]), "both_constant_in_selected_run": True},
        "export_speed_agreement": {
            "definition": "RMSE of supplied speed minus central-position-derivative speed on the exported 601 points; one-sided second-order endpoint differences",
            "follower_rmse_m_s": float(np.sqrt(np.mean((pair.follower_v_raw_m_s - pair.follower_v_from_x_m_s) ** 2))),
            "leader_rmse_m_s": float(np.sqrt(np.mean((pair.leader_v_raw_m_s - pair.leader_v_from_x_m_s) ** 2))),
        },
        "export": {
            "start_s": start, "end_s": start + 60,
            "calibration_interval_s": [start, start + 30],
            "validation_interval_s": [start + 30, start + 60],
            "calibration_fit_mask": "relative_time_s < 30; boundary at 30 s is reserved for validation initialization",
            "validation_samples": int(pair.phase.eq("validation").sum()),
            "calibration_samples": int(pair.phase.eq("calibration").sum()),
            "rows": len(pair), "minimum_net_gap_m": float(pair.net_gap_m.min()),
            "maximum_net_gap_m": float(pair.net_gap_m.max()),
        },
    }
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    (audit_output / "selection_audit.json").write_text(audit_text, encoding="utf8", newline="\n")
    (output / "selection_audit.json").write_text(audit_text, encoding="utf8", newline="\n")
    manifest = {
        "source_zip": source_zip.name, "source_zip_sha256": file_hash(source_zip),
        "source_csv": SOURCE_CSV, "source_csv_sha256": actual_hash,
        "subset_file": "following_pair.csv", "subset_sha256": file_hash(output / "following_pair.csv"),
        "lane_id": 2, "follower_id": int(selected.follower_id), "leader_id": int(selected.leader_id),
        "sample_interval_s": .1, "sample_count": len(pair),
        "observation_interval_s": [start, start + 60],
        "calibration_interval_s": [start, start + 30], "validation_interval_s": [start + 30, start + 60],
        "calibration_fit_mask": "relative_time_s < 30", "validation_mask": "relative_time_s >= 30",
        "coordinate_semantics": "local_y_m is the observed front-centre longitudinal position; increasing y is ahead; absolute origin is retained",
        "net_gap_definition": "leader_x_m - follower_x_m - leader_length_m",
        "leader_identity": "inferred per frame from all observed vehicles in lane 2; nearest strictly ahead front centre; stable in the full selected run; not an official preceding_id field",
        "units": {"position_and_length": "m", "time": "s", "speed": "m/s", "raw_speed_increment": "m/s^2"},
        "preprocessing": "No smoothing or unit reconversion. Original SI position, supplied speed and vehicle length fields retained; central-position-derivative speeds are additional diagnostics only.",
        "selection_is_independent_of_models": True, "selection_rule": RULE,
        "selected_maximal_run": audit["selected_maximal_run"],
        "eligible_candidates_in_rank_order": [
            {key: value.item() if isinstance(value, np.generic) else value for key, value in row.items()}
            for row in eligible.to_dict(orient="records")
        ],
        "speed_field_note": "Source speeds and positions are separately supplied fields; this does not imply independent measurement processes. For a prescribed observed leader position curve, use that curve's piecewise derivative as the leader velocity in the dynamics; the raw speed remains a measurement target and diagnostic.",
        "source_audit": {key: audit[key] for key in ("source_rows", "source_vehicles", "missing_by_column", "duplicate_vehicle_frame_keys", "lane_2_rows", "stable_leader_runs_total", "eligible_runs", "exclusion_reason_counts")},
        "export_summary": audit["export"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
    print(json.dumps({"follower_id": int(selected.follower_id), "leader_id": int(selected.leader_id), "eligible_runs": len(eligible), "selected_run_duration_s": float(selected.duration_s), "export": audit["export"], "sha256": manifest["subset_sha256"]}, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zip", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/week05_source_reproduced", help="Defaults to an ignored audit directory; use --output week05/data only for an intentional bundled-data refresh")
    parser.add_argument("--audit-output", type=Path, default=ROOT / ".local/week05_data_audit")
    args = parser.parse_args()
    prepare(args.source_zip, args.output, args.audit_output)


if __name__ == "__main__":
    main()
