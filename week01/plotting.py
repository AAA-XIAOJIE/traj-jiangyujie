"""Publication-style A1 plots of every individual and a transparent time window."""
from pathlib import Path

from . import style
from .analysis import clip_segments, contiguous_groups

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
import numpy as np

CMAP = "viridis"
INK = "#152A3A"
MUTED = "#65747E"


def setup_axis(ax, bundle, span=(26.0, 20.0)):
    df = bundle["data"]
    cx = (df.x_m.min() + df.x_m.max()) / 2
    cy = (df.y_m.min() + df.y_m.max()) / 2
    ax.set_xlim(cx - span[0] / 2, cx + span[0] / 2)
    ax.set_ylim(cy - span[1] / 2, cy + span[1] / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World x (m)", color=INK, labelpad=3)
    ax.set_ylabel("World y (m)", color=INK, labelpad=3)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, color="#E6EAED", linewidth=0.45, zorder=0)
    ax.tick_params(length=2.5, width=0.5, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#ACB7BE")


def draw_all(ax, bundle, normalized=True, context=False):
    edges, times = bundle["edges"], bundle["times"]
    duration = bundle["summary"]["coverage_s"]
    if context:
        collection = LineCollection(edges, colors="#BFC8CE", linewidths=0.40, alpha=0.24, zorder=1)
    else:
        norm = Normalize(0, 1 if normalized else duration)
        collection = LineCollection(edges, cmap=CMAP, norm=norm,
                                    linewidths=0.68, alpha=0.65, zorder=2)
        collection.set_array(times.mean(axis=1) / duration if normalized else times.mean(axis=1))
    ax.add_collection(collection)
    isolated = []
    for _, track in bundle["data"].groupby("track_id"):
        isolated.extend(g.iloc[0] for g in contiguous_groups(track, bundle["spec"]) if len(g) == 1)
    if isolated and not context:
        ax.scatter([p.x_m for p in isolated], [p.y_m for p in isolated], s=4,
                   c=[p.time_s / duration if normalized else p.time_s for p in isolated],
                   cmap=CMAP, norm=norm, linewidths=0, zorder=3)
    return collection


def plot_overview(bundles: list[dict], output: Path, dpi: int):
    style.apply()
    fig = plt.figure(figsize=(11.7, 8.6))
    grid = fig.add_gridspec(2, 3, left=0.075, right=0.975, top=0.83,
                           bottom=0.17, wspace=0.33, hspace=0.46)
    fig.text(0.075, 0.94, "Pedestrian trajectories across five ETH / UCY scenes",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(0.075, 0.899, "WEEK 01  /  A1     Native annotations  ·  one path per person  ·  metric world coordinates",
             fontsize=10, color=MUTED)
    for i, bundle in enumerate(bundles):
        ax = fig.add_subplot(grid[i // 3, i % 3])
        setup_axis(ax, bundle)
        draw_all(ax, bundle)
        style.panel_label(ax, chr(97 + i))
        s = bundle["summary"]
        ax.set_title(f"{s['label']}  |  {s['scene']}", loc="left", pad=21,
                     fontweight="bold", color=INK, fontsize=10)
        ax.text(0, 1.026, f"{s['tracks']:,} people  ·  {s['annotation_points']:,} points  ·  {s['coverage_s']:.1f} s",
                transform=ax.transAxes, color=MUTED, fontsize=8)
    ax = fig.add_subplot(grid[1, 2])
    ax.set_axis_off()
    style.panel_label(ax, "f", x=-0.06, y=1.025)
    total_n = sum(b["summary"]["tracks"] for b in bundles)
    total_p = sum(b["summary"]["annotation_points"] for b in bundles)
    total_gap = sum(b["summary"]["unlinked_long_intervals"] for b in bundles)
    ax.text(0, 1.055, "A reproducible trajectory view", transform=ax.transAxes,
            fontweight="bold", fontsize=10, color=INK)
    ax.text(0, .85, f"{total_n:,}", fontsize=29, fontweight="bold", color=style.C_PED, transform=ax.transAxes)
    ax.text(0, .745, f"scene-local individuals / {total_p:,} annotation points", fontsize=8.5, color=MUTED, transform=ax.transAxes)
    notes = [
        ("01  Read the source", "Pinned GitHub commit + SHA-256 checksums"),
        ("02  Keep identity and time", "Sort by person ID, then source frame"),
        ("03  Preserve temporal breaks", f"{total_gap} long control-point intervals left unlinked"),
        ("04  Export editable figures", "PNG 600 dpi / PDF / SVG"),
    ]
    for j, (title, subtitle) in enumerate(notes):
        y = .59 - j * .17
        ax.text(0, y, title, fontsize=9, fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(0, y - .072, subtitle, fontsize=8, color=MUTED, transform=ax.transAxes)
    cax = fig.add_axes([0.075, 0.087, 0.265, 0.014])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([0, .25, .5, .75, 1], labels=["0", "25", "50", "75", "100%"])
    cb.ax.tick_params(length=2, width=.4, pad=2, labelsize=8)
    cb.outline.set_linewidth(.4)
    fig.text(0.075, 0.117, "Time within each recording (normalized)", fontsize=9, fontweight="bold", color=INK)
    fig.text(.39, .112, "Every scene uses the same metric span (26 × 20 m); origins are scene-specific.", fontsize=8.5, color=MUTED)
    fig.text(.39, .087, "ETH: 0.4 s observations. UCY: irregular control points; links >4 s omitted.", fontsize=8.5, color=MUTED)
    fig.text(.075, .031, "Source: erichhhhho/DataExtraction @ 74006729b1ba  |  Full annotation coverage; spatial overlap does not establish interaction.", fontsize=8, color=MUTED)
    style.save(fig, output, dpi)


def plot_scene(bundle: dict, output: Path, dpi: int):
    style.apply()
    s, window = bundle["summary"], bundle["summary"]["window"]
    width = window["end_s"] - window["start_s"]
    fig = plt.figure(figsize=(11.7, 5.8))
    grid = fig.add_gridspec(1, 2, left=.075, right=.97, bottom=.29, top=.80, wspace=.22)
    fig.text(.075, .928, f"{s['label']} / {s['scene']}", fontsize=17, color=INK, fontweight="bold")
    fig.text(.075, .876, f"{s['tracks']:,} individuals  ·  {s['annotation_points']:,} native annotation points  ·  {s['coverage_s']:.2f} s coverage", fontsize=10, color=MUTED)
    ax1, ax2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    setup_axis(ax1, bundle)
    setup_axis(ax2, bundle)
    draw_all(ax1, bundle, normalized=False)
    draw_all(ax2, bundle, context=True)
    cropped, times = clip_segments(bundle["edges"], bundle["times"], window["start_s"], window["end_s"])
    collection = LineCollection(cropped, cmap=CMAP, norm=Normalize(0, width), linewidths=1.45, zorder=3)
    collection.set_array(times)
    ax2.add_collection(collection)
    style.panel_label(ax1, "a", x=-.11)
    style.panel_label(ax2, "b", x=-.11)
    ax1.set_title("All individual trajectories", loc="left", pad=17, fontweight="bold", color=INK)
    ax2.set_title(f"{width:g} s detail  |  {window['start_s']:.0f}–{window['end_s']:.0f} s", loc="left", pad=17, fontweight="bold", color=INK)
    ax2.text(.02, .03, f"{window['tracks_with_visible_link']} individuals with visible links", transform=ax2.transAxes, fontsize=8, color=INK, bbox={"fc": "white", "ec": "none", "alpha": .9})
    for left, duration, label in [(.075, s["coverage_s"], "Time since first scene annotation (s)"), (.575, width, "Time since detail-window start (s)")]:
        cax = fig.add_axes([left, .135, .24, .018])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, duration), cmap=CMAP), cax=cax, orientation="horizontal")
        cb.set_ticks(np.linspace(0, duration, 5))
        cb.ax.tick_params(length=2, width=.4, pad=2, labelsize=8)
        cb.outline.set_linewidth(.4)
        fig.text(left, .178, label, fontsize=8.5, color=INK)
    fig.text(.075, .049, f"Detail: earliest {width:g} s window maximizing visible individuals on a 1 s grid. Gray: full-scene context.", fontsize=8, color=MUTED)
    fig.text(.075, .020, "Links join successive accepted annotations of the same person. UCY links are geometric interpolation, not new observations.", fontsize=8, color=MUTED)
    style.save(fig, output, dpi)
