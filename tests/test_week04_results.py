"""Audit the submitted scientific evidence and an optional independent rerun."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "week04/results"
RESULTS = Path(os.environ.get("WEEK04_RESULTS", str(COMMITTED))).resolve()


def test_input_config_and_output_hashes_match_the_recorded_provenance():
    record = json.loads((RESULTS / "reproducibility.json").read_text(encoding="utf8"))
    assert hashlib.sha256((ROOT / "week04/data/lane2_support.csv.gz").read_bytes()).hexdigest() == record["input_sha256"]
    assert hashlib.sha256((ROOT / "week04/config.json").read_bytes()).hexdigest() == record["config_sha256"]
    for name, expected in record["source_code_sha256"].items():
        assert hashlib.sha256((ROOT / "week04" / name).read_bytes()).hexdigest() == expected, name
    for name, expected in record["output_sha256"].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == expected, name


def test_all_four_methods_use_the_same_complete_valid_windows_and_units():
    points = pd.read_csv(RESULTS / "fundamental_points.csv")
    assert set(points.method) == {"edie", "detector", "voronoi", "kernel"}
    for duration, expected in [(10, 84), (30, 28), (60, 14)]:
        group = points[points.window_s == duration]
        for _, method in group.groupby("method"):
            np.testing.assert_allclose(method.start_s, np.arange(0, 840, duration))
            assert len(method) == expected
            assert (method.duration_s == duration).all()
            assert np.isfinite(method[["k_veh_km", "q_veh_h", "v_km_h"]]).all().all()
            np.testing.assert_allclose(method.coverage, 1, atol=1e-8)
            np.testing.assert_allclose(method.q_veh_h, method.k_veh_km * method.v_km_h, rtol=2e-9)


def test_scale_changes_preserve_total_residence_and_distance():
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf8"))
    points = pd.read_csv(RESULTS / "fundamental_points.csv")
    for _, group in points[points.method == "edie"].groupby("window_s"):
        assert group.vehicle_seconds.sum() == pytest.approx(audit["roi_vehicle_seconds"], rel=1e-9)
        assert group.vehicle_metres.sum() == pytest.approx(audit["roi_vehicle_metres"], rel=1e-9)
    assert max(item["max_abs_error"] for item in audit["spatial_additivity"]) < 1e-8
    assert audit["voronoi_invalid_in_analysis"] == 0
    assert max(item["max_q_difference"] for item in audit["quadrature_convergence"]) < .01


@pytest.mark.skipif(RESULTS == COMMITTED, reason="Set WEEK04_RESULTS to check a fresh independent output directory")
def test_fresh_numerical_outputs_match_the_committed_results():
    for reference in COMMITTED.glob("*.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(RESULTS / reference.name), pd.read_csv(reference),
            check_dtype=False, rtol=1e-8, atol=1e-7,
        )
