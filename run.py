"""Reproduce Week 01 figures and scene statistics from verified ETH/UCY data."""
import argparse
from pathlib import Path

import pandas as pd

from week01.analysis import build_geometry, select_window, summarize
from week01.data import download_data, load_scene, read_json, verify_coordinates, verify_data
from week01.plotting import plot_overview, plot_scene

ROOT = Path(__file__).resolve().parent


def run(download=False):
    config = read_json(ROOT / "configs/week01.json")
    if download:
        download_data(ROOT)
    verify_data(ROOT)
    week = ROOT / config["week"]
    figures, results = week / "figures", week / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    bundles = []
    for spec in config["scenes"]:
        data = load_scene(ROOT / "data/raw" / spec["source_dir"] / "world_coordinate.csv", spec)
        verify_coordinates(ROOT, spec, data)
        edges, times, tracks, breaks = build_geometry(data, spec)
        window = select_window(data, spec, config["preview_window_s"])
        summary = summarize(data, spec, edges, tracks, breaks, window)
        bundle = {"spec": spec, "data": data, "edges": edges, "times": times,
                  "tracks": tracks, "breaks": breaks, "summary": summary}
        bundles.append(bundle)
        plot_scene(bundle, figures / f"{spec['name']}_trajectories", config["scene_dpi"])
        print(f"{spec['name']}: {summary['tracks']} tracks, {len(data)} points, {len(breaks)} unlinked intervals", flush=True)
    plot_overview(bundles, figures / "eth_ucy_trajectory_overview", config["overview_dpi"])
    rows = [{k: v for k, v in b["summary"].items() if k != "window"} for b in bundles]
    pd.DataFrame(rows).to_csv(results / "scene_summary.csv", index=False, lineterminator="\n", float_format="%.6f")
    print("Done: six PNG figures + scene_summary.csv. PDF/SVG copies are available locally.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing source files and check SHA-256.")
    run(parser.parse_args().download)
