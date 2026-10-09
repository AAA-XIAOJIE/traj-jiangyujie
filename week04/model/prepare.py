"""Regenerate the bundled support subset from the supplied classroom ZIP."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_zip", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Use a separate path to compare against the bundled subset")
    args = parser.parse_args()
    manifest = json.loads((ROOT/"data/manifest.json").read_text(encoding="utf8"))
    with zipfile.ZipFile(args.source_zip) as archive:
        raw = archive.read(manifest["source_csv"])
        if hashlib.sha256(raw).hexdigest() != manifest["source_csv_sha256"]:
            raise ValueError("Classroom CSV hash differs from the recorded source")
        import io
        data = pd.read_csv(io.BytesIO(raw))
    if len(data) != manifest["raw_rows"] or data.isna().any().any() or data.duplicated(["vehicle_id", "frame_id"]).any():
        raise ValueError("Unexpected source size, missing data or duplicate keys")
    ids = data.loc[data.lane_id == 2, "vehicle_id"].unique()
    selected = data[data.vehicle_id.isin(ids)][["vehicle_id", "frame_id", "time_s", "lane_id", "local_y_m", "speed_m_s"]]
    selected = selected.sort_values(["vehicle_id", "time_s"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(selected.to_csv(index=False, float_format="%.6f", lineterminator="\n").encode())
    actual = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if actual != manifest["subset_sha256"]:
        raise ValueError("Regenerated subset differs from the committed subset")
    print(f"Verified {len(selected)} records, {len(ids)} vehicles; SHA-256 {actual}")


if __name__ == "__main__":
    main()
