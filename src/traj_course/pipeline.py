"""End-to-end Week 01 production, audit tables and a Chinese course conclusion."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform

import pandas as pd

from .analysis import build_geometry, select_window, summarize
from .io import load_scene, read_json, sha256, verify_coordinates, verify_data
from .plotting import plot_overview, plot_scene


def run(root: Path, config_path: Path) -> dict:
    config = read_json(config_path)
    verify_data(root)
    week_dir = root / config["week"]
    figure_dir, result_dir = week_dir / "figures", week_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    bundles, checks, sensitivity = [], {}, []
    for spec in config["scenes"]:
        data = load_scene(root / "data" / "raw" / spec["source_dir"] / "world_coordinate.csv", spec)
        checks[spec["name"]] = verify_coordinates(root, spec, data)
        edges, times, tracks, breaks = build_geometry(data, spec)
        window = select_window(data, spec, config["preview_window_s"])
        summary = summarize(data, spec, edges, tracks, breaks, window)
        bundle = {"spec": spec, "data": data, "edges": edges, "times": times,
                  "tracks": tracks, "breaks": breaks, "summary": summary}
        bundles.append(bundle)
        data.to_csv(root / "data" / "processed" / f"{spec['name']}.csv", index=False, lineterminator="\n", float_format="%.8f")
        for threshold in ([0.4] if spec["family"] == "ETH" else [2.0, 4.0, 8.0]):
            altered = {**spec, "max_link_s": threshold}
            test_edges, _, _, test_breaks = build_geometry(data, altered)
            sensitivity.append({"scene": spec["name"], "max_link_s": threshold,
                                "rendered_edges": len(test_edges), "unlinked_intervals": len(test_breaks)})
        plot_scene(bundle, figure_dir / f"{spec['name']}_trajectories", config["scene_dpi"])
        print(f"{spec['name']}: {summary['tracks']} tracks, {len(data)} points, {len(breaks)} unlinked intervals", flush=True)
    plot_overview(bundles, figure_dir / "eth_ucy_trajectory_overview", config["overview_dpi"])
    summary = [b["summary"] for b in bundles]
    flat = [{k: v for k, v in s.items() if k != "window"} for s in summary]
    pd.DataFrame(flat).to_csv(result_dir / "scene_summary.csv", index=False, lineterminator="\n", float_format="%.6f")
    pd.concat([b["tracks"] for b in bundles]).to_csv(result_dir / "track_summary.csv", index=False, lineterminator="\n", float_format="%.6f")
    pd.concat([b["breaks"] for b in bundles]).to_csv(result_dir / "unlinked_intervals.csv", index=False, lineterminator="\n", float_format="%.6f")
    pd.DataFrame(sensitivity).to_csv(result_dir / "link_threshold_sensitivity.csv", index=False, lineterminator="\n")
    (result_dir / "scene_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_report(week_dir, summary)
    receipt = {
        "schema_version": 1,
        "package_version": importlib.metadata.version("traj-course"),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "matplotlib"]},
        "data_commit": read_json(root / "data" / "manifest.json")["commit"],
        "config_sha256": sha256(config_path),
        "manifest_sha256": sha256(root / "data" / "manifest.json"),
        "tracks": sum(s["tracks"] for s in summary),
        "annotation_points": sum(s["annotation_points"] for s in summary),
        "coordinate_consistency": checks,
        "figures": {p.relative_to(root).as_posix(): sha256(p) for p in sorted(figure_dir.iterdir()) if p.suffix in [".png", ".pdf", ".svg"]},
    }
    (result_dir / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Generated {len(receipt['figures'])} figure files in {week_dir.name}/figures", flush=True)
    return receipt


def write_report(week_dir: Path, summary: list[dict]):
    by_name = {s["scene"]: s for s in summary}
    total_tracks = sum(s["tracks"] for s in summary)
    total_points = sum(s["annotation_points"] for s in summary)
    total_gaps = sum(s["unlinked_long_intervals"] for s in summary)
    hotel = by_name["seq_hotel"]
    conclusion = (
        f"本周基于 ETH 的 seq_eth、seq_hotel 和 UCY 的 zara01、zara02、students03 五个场景，"
        f"完成了 {total_tracks:,} 条场景内个体轨迹、{total_points:,} 个原生标注点的整理与绘图。"
        "五个场景分别包含 360、390、148、204、434 个个体。"
        f"Hotel 的轨迹覆盖在空间上较狭长，x、y 方向跨度分别为 {hotel['x_max_m'] - hotel['x_min_m']:.2f} m 和 {hotel['y_max_m'] - hotel['y_min_m']:.2f} m；"
        "Zara01 和 Zara02 以横向通行路径为主，Students03 的轨迹在多条通行方向上交织，说明同一绘图流程需要保留各场景的坐标与标注特征。"
        "处理时按场景和个体 ID 分组、按时间排序，ETH 保留 0.4 s 采样，UCY 保留不等间隔控制点，"
        f"并对 {total_gaps} 处超过 4 s 的控制点间隔断开连线，防止把长时间未观测区间画成连续运动。"
        "全时段叠加图用于观察通行路径的空间分布；路径交叉不等于同一时刻发生接触或避让，"
        "各场景的覆盖时长和标注方式也不同，因此本周不据此判断密度高低、碰撞风险或群体行为。"
    )
    (week_dir / "conclusion.md").write_text("# 第 1 周结论：A1 轨迹可视化\n\n" + conclusion + "\n", encoding="utf-8", newline="\n")
    bullets = "\n".join(f"- **{s['scene']}**：{s['tracks']} 个个体，{s['annotation_points']:,} 个标注点，覆盖 {s['coverage_s']:.2f} s；示例窗口 {s['window']['start_s']:.0f}–{s['window']['end_s']:.0f} s。" for s in summary)
    report = f"""# Week 01 · A1：每个个体一条轨迹

本周提交三项：[代表图](figures/eth_ucy_trajectory_overview.png)、[一段结论](conclusion.md)、[可复现代码](../src/traj_course)。

![五场景轨迹总览](figures/eth_ucy_trajectory_overview.png)

## 数据与读图

{bullets}

每条路径按场景内的个体 ID 独立构建，颜色表示时间，不表示身份或速度。不同场景 ID 不互通。总览图颜色表示各场景自身覆盖时段的 0–100%；五个子图使用相同的 26 × 20 m 显示跨度与等比例坐标，保留原坐标原点。显示范围不是密度计算 ROI。

单场景图的左图覆盖全部标注，右图为 12 s 细节；右图浅灰线是全时段路径背景，彩色线属于选中时间窗。窗口在整秒网格上选择含有效轨迹最多的 12 s，并列取最早；这是可复现的展示规则，不代表统计抽样。

## 本周结论

{conclusion}

## 方法与边界

源 CSV 是 4 × N，行顺序为 frame、ID、y、x。统一表格显式换为 x、y；坐标使用米，未做归一化、旋转、ROI 裁剪或额外平滑。ETH 的源帧间隔分别为 6、10 帧，结合源说明的 0.4 s 标注间隔得到 15、25 fps 的帧索引时间基准；UCY 用源视频 25 fps。时间零点取该场景的首个标注帧，不代表视频开始。

ETH 只连接相邻的原生采样点。UCY 连线是相邻控制点之间的直线几何插值，默认只连接间隔不超过 4 s 的点；这不产生新的观测点，也不声称恢复原始样条。阈值影响保留的连线，[2/4/8 s 敏感性结果](results/link_threshold_sensitivity.csv)与[全部被断开的间隔](results/unlinked_intervals.csv)均保留。4 s 是保守的显示参数，不是已标注的遮挡或跟踪错误判据；断开后的孤立点仍以小圆点显示。

总览模仿已有 PEM-Flow 论文的分面、字体、时间色条和矢量导出方式。这里采用有顺序意义的 viridis 时间色板，且只展示已核实的世界坐标，不添加未经配准验证的照片背景。

## 文件

- `figures/eth_ucy_trajectory_overview.png/.pdf/.svg`：课程代表图（PNG 为 600 dpi）。
- `figures/*_trajectories.png/.pdf/.svg`：五张全场景 + 12 s 细节图（PNG 为 300 dpi）。
- `results/scene_summary.csv`、`track_summary.csv`：场景与个体统计。
- `results/scene_summary.json`：包含每场景选中窗口的结构化结果。
- `results/run_receipt.json`：版本、数据/配置/输出哈希及坐标一致性核验。

坐标核验比较的是同一来源的表示方式：ETH 与 obsmat 对照，UCY 与 pixel.csv + H 对照。数值差异主要来自 CSV 输出精度，**不能作为独立定位精度验证**。

## 复现

在仓库根目录运行 `python scripts/run_week01.py --download`。环境安装方法见[根目录 README](../README.md)，字段和来源见 [DATA_CARD](../docs/DATA_CARD.md)。
"""
    (week_dir / "README.md").write_text(report, encoding="utf-8", newline="\n")
