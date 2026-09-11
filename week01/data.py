"""Pinned downloads and explicit parsing of the upstream 4 x N layout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_data_path(raw_dir: Path, relative: str) -> Path:
    target = (raw_dir / relative).resolve()
    if not target.is_relative_to(raw_dir.resolve()):
        raise ValueError(f"Data path escapes raw directory: {relative}")
    return target


def download_data(root: Path) -> list[dict]:
    """Download only missing files; verify hashes before accepting any file."""
    manifest = read_json(root / "data" / "manifest.json")
    receipt = []
    for entry in manifest["files"]:
        target = safe_data_path(root / "data" / "raw", entry["path"])
        if target.exists():
            if sha256(target) != entry["sha256"]:
                raise ValueError(f"Checksum mismatch, existing file: {entry['path']}")
            status = "verified_cache"
        else:
            expected_prefix = "https://raw.githubusercontent.com/erichhhhho/DataExtraction/" + manifest["commit"] + "/"
            if not entry["url"].startswith(expected_prefix):
                raise ValueError("Unexpected download source in manifest")
            req = urllib.request.Request(entry["url"], headers={"User-Agent": "traj-course/0.1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                content = response.read()
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Checksum mismatch, download: {entry['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(target)
            status = "downloaded"
        receipt.append({"path": entry["path"], "sha256": entry["sha256"], "status": status})
    return receipt


def verify_data(root: Path) -> None:
    for entry in read_json(root / "data" / "manifest.json")["files"]:
        path = safe_data_path(root / "data" / "raw", entry["path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing {entry['path']}; run: python run.py --download")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {entry['path']}")


def load_scene(path: Path, spec: dict) -> pd.DataFrame:
    """Upstream rows are frame, person, y, x; output columns use x, y."""
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    if values.shape[0] != 4 or values.shape[1] == 0:
        raise ValueError(f"Expected 4 x N upstream data, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite coordinate or identifier")
    if not np.equal(values[:2], np.floor(values[:2])).all():
        raise ValueError("Frame and track identifiers must be integers")
    df = pd.DataFrame({
        "frame_id": values[0].astype(np.int64),
        "track_id": values[1].astype(np.int64),
        "x_m": values[3],
        "y_m": values[2],
    })
    if df.duplicated(["track_id", "frame_id"]).any():
        raise ValueError("Duplicate (track_id, frame_id) observations")
    if spec["fps"] <= 0:
        raise ValueError("fps must be positive")
    df.insert(0, "scene", spec["name"])
    df["time_s"] = (df["frame_id"] - df["frame_id"].min()) / spec["fps"]
    return df.sort_values(["track_id", "frame_id"], kind="stable").reset_index(drop=True)


def verify_coordinates(root: Path, spec: dict, df: pd.DataFrame) -> dict:
    """Cross-check the parsed axes against ETH observations or UCY H x pixels."""
    folder = root / "data" / "raw" / spec["source_dir"]
    keys = ["track_id", "frame_id"]
    if spec["family"] == "ETH":
        raw = np.loadtxt(folder / "obsmat.txt")
        reference = pd.DataFrame({"frame_id": raw[:, 0].astype(int), "track_id": raw[:, 1].astype(int), "ref_x": raw[:, 2], "ref_y": raw[:, 4]})
        source = "obsmat columns 3 and 5 (1-based)"
    else:
        raw = np.loadtxt(folder / "pixel.csv", delimiter=",").T
        homography = np.loadtxt(folder / "H.txt")
        homogeneous = np.column_stack([raw[:, 3], raw[:, 2], np.ones(len(raw))]) @ homography.T
        projected = homogeneous[:, :2] / homogeneous[:, 2:3]
        reference = pd.DataFrame({"frame_id": raw[:, 0].astype(int), "track_id": raw[:, 1].astype(int), "ref_x": projected[:, 0], "ref_y": projected[:, 1]})
        source = "pixel.csv [u,v,1] projected through H, homogeneous normalization"
    joined = df.merge(reference, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Coordinate reference keys do not match")
    error = np.hypot(joined.x_m - joined.ref_x, joined.y_m - joined.ref_y)
    # CSV was exported with finite decimal precision; this is a consistency
    # tolerance, not a measurement-accuracy claim.
    if error.max() > 0.002:
        raise ValueError(f"Coordinate cross-check failed: {error.max():.6f} m")
    return {"reference": source, "max_export_rounding_difference_m": float(error.max()), "checked_points": len(joined)}
