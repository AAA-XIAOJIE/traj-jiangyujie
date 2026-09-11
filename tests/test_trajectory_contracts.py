"""Small analytic fixtures test identity, units, timing and no false bridges."""
import json

import numpy as np
import pandas as pd
import pytest

from week01.analysis import build_geometry, clip_segments, select_window
from week01.data import load_scene, safe_data_path, verify_data


ETH = {"name": "fixture", "family": "ETH", "fps": 15, "expected_step_frames": 6, "max_link_s": .4}
UCY = {"name": "fixture", "family": "UCY", "fps": 25, "expected_step_frames": None, "max_link_s": 4}


def frame_table(frames, ids, x, y, spec=ETH):
    return pd.DataFrame({"scene": "fixture", "frame_id": frames, "track_id": ids,
                         "x_m": x, "y_m": y, "time_s": np.asarray(frames) / spec["fps"]})


def test_four_rows_become_explicit_xy_and_seconds(tmp_path):
    source = tmp_path / "world.csv"
    np.savetxt(source, [[12, 6], [1, 1], [20, 10], [4, 2]], delimiter=",")
    df = load_scene(source, ETH)
    assert df.x_m.tolist() == [2, 4]
    assert df.y_m.tolist() == [10, 20]
    assert df.time_s.tolist() == [0, .4]


@pytest.mark.parametrize("values, error", [
    ([[0, 0], [1, 1], [2, 2], [3, 3]], "Duplicate"),
    ([[0, 6.5], [1, 1], [2, 2], [3, 3]], "integers"),
    ([[0, 6], [1, 1], [2, float('nan')], [3, 3]], "Non-finite"),
    ([[0, 6], [1, 1], [2, 2]], "4 x N"),
])
def test_invalid_input_is_rejected(tmp_path, values, error):
    source = tmp_path / "world.csv"
    np.savetxt(source, values, delimiter=",")
    with pytest.raises(ValueError, match=error):
        load_scene(source, ETH)


def test_no_cross_person_or_gap_bridges_and_sorted_time():
    df = frame_table([18, 0, 6, 0, 6], [1, 1, 1, 2, 2], [18, 0, 6, 100, 106], [0]*5)
    edges, times, tracks, breaks = build_geometry(df, ETH)
    assert edges.tolist() == [[[0, 0], [6, 0]], [[100, 0], [106, 0]]]
    assert len(breaks) == 1
    assert breaks.iloc[0].left_frame == 6
    assert breaks.iloc[0].right_frame == 18
    assert tracks.annotation_points.sum() == 5
    assert np.all(times[:, 1] > times[:, 0])


def test_ucy_irregular_intervals_are_allowed_but_long_intervals_break():
    df = frame_table([0, 25, 125, 226], [1]*4, [0, 1, 5, 9], [0]*4, UCY)
    edges, _, _, breaks = build_geometry(df, UCY)
    assert len(edges) == 2  # exactly 4 s is accepted
    assert breaks.iloc[0].interval_s == pytest.approx(4.04)


def test_window_clips_line_without_extrapolation():
    edges = np.array([[[0., 0.], [10., 20.]]])
    times = np.array([[0., 10.]])
    clipped, color_time = clip_segments(edges, times, 2, 6)
    np.testing.assert_allclose(clipped, [[[2, 4], [6, 12]]])
    np.testing.assert_allclose(color_time, [2])
    empty, _ = clip_segments(edges, times, 12, 15)
    assert len(empty) == 0


def test_window_selection_counts_identity_once_and_uses_earliest_tie():
    spec = {**UCY, "max_link_s": 20}
    df = frame_table([0, 250, 0, 250], [1, 1, 2, 2], [0, 10, 0, 10], [0, 0, 2, 2], spec)
    window = select_window(df, spec, 4)
    assert window["start_s"] == 0
    assert window["tracks_with_visible_link"] == 2


def test_manifest_path_cannot_escape_data_directory(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_data_path(tmp_path, "../outside.txt")


def test_modified_data_is_not_silently_accepted(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "test.csv").write_text("changed")
    manifest = {"files": [{"path": "test.csv", "sha256": "0"*64}]}
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Checksum"):
        verify_data(tmp_path)
