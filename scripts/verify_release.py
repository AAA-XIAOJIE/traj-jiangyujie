"""Validate generated figures and independent count-conservation invariants."""
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from PIL import Image
import pandas as pd

from traj_course.io import sha256, verify_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    verify_data(ROOT)
    results = ROOT / "week01" / "results"
    scenes = pd.read_csv(results / "scene_summary.csv")
    tracks = pd.read_csv(results / "track_summary.csv")
    gaps = pd.read_csv(results / "unlinked_intervals.csv")
    assert scenes.tracks.sum() == len(tracks) == 1536
    assert scenes.annotation_points.sum() == tracks.annotation_points.sum() == 25126
    assert scenes.unlinked_long_intervals.sum() == len(gaps) == 137
    assert (tracks.annotation_points - tracks.components == tracks.rendered_edges).all()
    assert scenes.rendered_edges.sum() == tracks.rendered_edges.sum() == 23453
    receipt = json.loads((results / "run_receipt.json").read_text(encoding="utf8"))
    checks = []
    for relative, digest in receipt["figures"].items():
        path = ROOT / relative
        assert sha256(path) == digest, f"Modified output: {relative}"
        check = {"file": relative, "bytes": path.stat().st_size}
        if path.suffix == ".png":
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                check["pixels"] = list(im.size)
                check["dpi"] = [round(v) for v in im.info.get("dpi", [])]
                assert im.width >= 3500 and im.height >= 1700
        elif path.suffix == ".svg":
            node = ET.parse(path)
            ns = {"s": "http://www.w3.org/2000/svg"}
            check["editable_text_elements"] = len(node.findall(".//s:text", ns))
            check["vector_paths"] = len(node.findall(".//s:path", ns))
            assert check["editable_text_elements"] > 20
            assert check["vector_paths"] > 100
        else:
            content = path.read_bytes()
            assert content.startswith(b"%PDF-") and b"%%EOF" in content[-32:]
        checks.append(check)
    assert len(checks) == 18
    for md in [ROOT / "README.md", ROOT / "week01/README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        for target in re.findall(r"\]\(([^)]+)\)", md.read_text(encoding="utf8")):
            if not target.startswith(("https://", "http://", "#")):
                assert (md.parent / target.split("#")[0]).exists(), f"Broken link in {md.name}: {target}"
    audit = {"source_checksums": "pass", "count_conservation": "pass", "local_document_links": "pass", "figures": checks}
    (results / "artifact_checks.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
    print(f"Verified {len(checks)} figure files, {len(tracks)} tracks, 25126 annotation points, 137 unlinked intervals.")


if __name__ == "__main__":
    main()
