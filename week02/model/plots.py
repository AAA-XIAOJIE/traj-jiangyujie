"""Course figures reuse Week 01's PEM-Flow typography and export helper."""
import numpy as np

from week01 import style

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

COLORS = {"observed": "#0072B2", "straight": "#999999", "orca": "#E69F00", "orca_right": "#009E73", "radius_030": "#CC79A7"}
LABELS = {"observed": "Observed", "straight": "Goal only", "orca": "ORCA", "orca_right": "ORCA + side preference", "radius_030": "Larger radius (0.30 m)"}
INK = "#152A3A"


def panel(ax, letter, title):
    ax.set_title(title, loc="left", fontsize=10, pad=12, fontweight="bold", color=INK)
    ax.text(-.13, 1.06, letter, transform=ax.transAxes, fontsize=13, weight="bold", color=INK)
    ax.grid(alpha=.15, linewidth=.5)


def map_axis(ax, experiment):
    center = experiment["center"]
    ax.set(xlim=(center[0] - 11, center[0] + 11), ylim=(center[1] - 11, center[1] + 11),
           xlabel="World x (m)", ylabel="World y (m)", aspect="equal")
    ax.add_patch(Circle(center, 10, fill=False, linestyle="--", lw=.6, color="#A7B1B8"))
    ax.scatter(*center, marker="+", s=20, color="#526773", zorder=4)


def trajectories(ax, positions, experiment, color=None):
    map_axis(ax, experiment)
    if color is None:
        edges = np.stack([positions[:-1], positions[1:]], axis=2).reshape(-1, 2, 2)
        lines = LineCollection(edges, cmap="viridis", norm=Normalize(0, experiment["times"][-1]),
                               linewidths=.65, alpha=.82)
        lines.set_array(np.repeat(experiment["times"][:-1], positions.shape[1]))
    else:
        lines = LineCollection(positions.transpose(1, 0, 2), colors=color, linewidths=.7, alpha=.62)
    ax.add_collection(lines)
    ax.scatter(positions[0, :, 0], positions[0, :, 1], s=9, facecolor="white", edgecolor="#526773", linewidth=.5, zorder=3)
    return lines


def observed_figure(experiment, observed, out):
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.8))
    fig.subplots_adjust(left=.09, right=.97, bottom=.1, top=.86, wspace=.32, hspace=.43)
    fig.suptitle("Circle antipode experiment: evidence before modeling", x=.09, y=.967, ha="left", fontsize=17, weight="bold", color=INK)
    fig.text(.09, .926, "64 people  /  425 frames  /  16.96 s  /  nominal radius 10 m  /  25 fps (Xiao et al., 2018)", fontsize=9, color="#65747E")
    ax = axes[0, 0]
    lines = trajectories(ax, experiment["positions"], experiment)
    panel(ax, "a", "Measured paths; circles mark initial positions")
    cb = fig.colorbar(lines, ax=ax, fraction=.035, pad=.025)
    cb.set_label("Time since first supplied frame (s)", fontsize=8)
    ax = axes[0, 1]
    speed = observed["speed"]
    t = experiment["times"]
    ax.fill_between(t, *np.quantile(speed, [.1, .9], axis=1), color=COLORS["observed"], alpha=.14, label="10–90% of people")
    ax.plot(t, speed.mean(axis=1), color=COLORS["observed"], label="Population mean")
    ax.axvspan(0, 2, color="#777777", alpha=.09, label="Initialization / speed calibration")
    ax.set(xlabel="Time (s)", ylabel="Speed (m/s)", xlim=(0, t[-1]), ylim=(0, None))
    panel(ax, "b", "Acceleration, interaction slowdown, then arrival")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax = axes[1, 0]
    initial = experiment["positions"][0]
    inward = experiment["goals"] - initial
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    progress = np.einsum("tij,ij->ti", experiment["positions"] - initial, inward)
    for i in range(len(initial)):
        ax.plot(progress[:, i], observed["lateral"][:, i], lw=.65, alpha=.55, color=COLORS["observed"])
    ax.axhline(0, color="#555555", lw=.6, linestyle="--")
    ax.set(xlabel="Progress along individual start–goal axis (m)", ylabel="Rightward lateral displacement (m)")
    panel(ax, "c", "Detours with both right and left choices")
    ax = axes[1, 1]
    d = observed["individual"]
    ax.scatter(d.approach_speed_mps, d.closest_speed_mps, s=24, color=COLORS["observed"], alpha=.8, edgecolor="white", lw=.4)
    upper = max(d.approach_speed_mps.max(), d.closest_speed_mps.max()) + .2
    ax.plot([0, upper], [0, upper], "--", color="#888888", lw=.8)
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="Approach speed, 1.6–2.6 s (m/s)", ylabel="Speed near closest approach to center (m/s)")
    panel(ax, "d", "Paired evidence for interaction slowdown")
    s = observed["stats"]
    ax.text(.04, .94, f"{s['slower_at_closest_fraction']:.1%} below equality line\nMean: {s['approach_speed_mps']:.2f} → {s['closest_approach_speed_mps']:.2f} m/s", transform=ax.transAxes, va="top", fontsize=9)
    fig.text(.09, .035, "Speed: centered 0.4 s displacement. Closest-approach speed: ±0.4 s around each person's nearest point to the fitted center.", fontsize=8, color="#65747E")
    style.save(fig, out, dpi=300)


def overview(experiment, cases, out):
    style.apply()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.1))
    fig.subplots_adjust(left=.07, right=.985, bottom=.12, top=.85, wspace=.31, hspace=.48)
    fig.suptitle("From observed behavior to reciprocal collision avoidance", x=.07, y=.97, ha="left", fontsize=17, weight="bold", color=INK)
    fig.text(.07, .925, "WEEK 02  /  measured initial positions  /  geometric antipodal goals  /  open-loop simulation after initialization", fontsize=9, color="#65747E")
    for j, key in enumerate(["observed", "orca", "orca_right"]):
        ax = axes[0, j]
        trajectories(ax, cases[key]["positions"], experiment, COLORS[key])
        panel(ax, chr(97 + j), LABELS[key])
        s = cases[key]["stats"]
        ax.text(.02, .025, f"Path within record: {s['mean_path_m']:.2f} m\nArrived {s['arrived_by_end']}/64 by 16.96 s", transform=ax.transAxes, fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    for j, column, ylabel, title in [(0, "mean_speed_mps", "Mean speed (m/s)", "Speed evolution"), (1, "mean_radius_m", "Mean distance to center (m)", "Convergence and dispersal")]:
        ax = axes[1, j]
        for key in ["straight", "observed", "orca", "orca_right"]:
            df = cases[key]["series"]
            ax.plot(df.time_s, df[column], label=LABELS[key], color=COLORS[key], lw=1.5,
                    linestyle="--" if key == "straight" else "-")
        ax.axvspan(0, 2, color="#777777", alpha=.07)
        ax.set(xlabel="Time (s)", ylabel=ylabel, xlim=(0, experiment["times"][-1]), ylim=(0, None))
        panel(ax, chr(100 + j), title)
        if j == 0:
            ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 2]
    keys = ["observed", "orca", "orca_right"]
    right = np.array([cases[k]["stats"]["right_fraction"] for k in keys])
    left = np.array([cases[k]["stats"]["left_fraction"] for k in keys])
    x = np.arange(3)
    ax.bar(x, right, color="#0072B2", width=.62, label="Right")
    ax.bar(x, 1 - right - left, bottom=right, color="#D9DFE3", width=.62, label="Near axis")
    ax.bar(x, left, bottom=1 - left, color="#CC79A7", width=.62, label="Left")
    ax.set(xticks=x, xticklabels=["Observed", "ORCA", "ORCA + side"], ylabel="Fraction of people", ylim=(0, 1.13))
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    panel(ax, "f", "Side preference is not guaranteed by avoidance")
    fig.text(.07, .05, "Right/left: sign of mean lateral displacement, ±0.10 m neutral band. All comparisons use the same 16.96 s interval.", fontsize=8, color="#65747E")
    fig.text(.07, .025, "Gray goal-only paths are a counterfactual control. Circle outlines indicate the experiment layout, not physical walls.", fontsize=8, color="#65747E")
    style.save(fig, out, dpi=300)


def failures(experiment, cases, metrics, out):
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.7))
    fig.subplots_adjust(left=.065, right=.985, bottom=.21, top=.74, wspace=.31)
    fig.suptitle("Where the model remains wrong", x=.065, y=.965, ha="left", fontsize=17, weight="bold", color=INK)
    fig.text(.065, .875, "Failure cases and stability checks are retained; avoiding modeled disks is not the same as explaining human route choice.", fontsize=9, color="#65747E")
    model = cases["orca_right"]
    errors = np.linalg.norm(model["positions"] - experiment["positions"], axis=-1)
    worst = int(np.argmax(errors[experiment["times"] > 2].mean(axis=0)))
    ax = axes[0]
    for key in ["observed", "orca", "orca_right"]:
        p = cases[key]["positions"][:, worst]
        ax.plot(p[:, 0], p[:, 1], label=LABELS[key], color=COLORS[key], lw=1.5)
    ax.scatter(*experiment["positions"][0, worst], marker="o", s=30, color=INK)
    ax.scatter(*experiment["goals"][worst], marker="*", s=70, color=INK)
    ax.set(xlabel="World x (m)", ylabel="World y (m)", aspect="equal")
    panel(ax, "a", f"Largest mean position error: person {experiment['ids'][worst]}")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    t = experiment["times"]
    ax.fill_between(t, *np.quantile(errors, [.1, .9], axis=1), color=COLORS["orca_right"], alpha=.15, label="10–90%")
    ax.plot(t, errors.mean(axis=1), color=COLORS["orca_right"], label="Mean")
    ax.axvspan(0, 2, color="#777777", alpha=.07)
    ax.set(xlabel="Time (s)", ylabel="Position error to observed (m)", xlim=(0, t[-1]), ylim=(0, None))
    panel(ax, "b", "Individual paths are not recovered")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[2]
    for key in ["observed", "orca_right", "radius_030"]:
        df = cases[key]["series"]
        ax.plot(t, df.arrived_fraction, color=COLORS[key], label=LABELS[key], lw=1.5)
    ax.set(xlabel="Time (s)", ylabel="Cumulative arrived fraction", xlim=(0, t[-1]), ylim=(0, 1.04))
    panel(ax, "c", "A larger body radius brings back congestion")
    ax.legend(frameon=False, fontsize=7)
    s = model["stats"]
    fig.text(.065, .065, f"After-2 s ADE = {s['ade_after_calibration_m']:.2f} m. Body radius 0.25 m is an assumed disk proxy, not a measured shoulder shape.", fontsize=8, color="#65747E")
    fig.text(.065, .028, "Single recording: behavior-informed model development, not independent validation. Full parameter and perturbation results: metrics.csv.", fontsize=8, color="#65747E")
    style.save(fig, out, dpi=300)


def replay(experiment, cases, output):
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    fig.subplots_adjust(left=.055, right=.99, bottom=.17, top=.82, wspace=.22)
    artists = []
    keys = ["observed", "orca", "orca_right"]
    for ax, key in zip(axes, keys):
        map_axis(ax, experiment)
        ax.set_title(LABELS[key], fontsize=10, weight="bold")
        ax.add_collection(LineCollection(cases[key]["positions"].transpose(1, 0, 2), colors=COLORS[key], alpha=.08, linewidths=.5))
        points = ax.scatter([], [], s=12, color=COLORS[key])
        artists.append(points)
    title = fig.text(.055, .94, "", fontsize=12, weight="bold", color=INK)
    fig.text(.055, .035, "Circle-antipode crossing | same initial positions and time | dots denote tracked centers, not body outlines", fontsize=8, color="#65747E")
    def update(frame):
        for artist, key in zip(artists, keys):
            artist.set_offsets(cases[key]["positions"][frame])
        title.set_text(f"Circle antipode experiment   t = {experiment['times'][frame]:.2f} s")
        return artists + [title]
    frames = np.unique(np.r_[np.arange(0, len(experiment["times"]), 5), len(experiment["times"]) - 1])
    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    animation.save(output, writer=PillowWriter(fps=5), dpi=105)
    plt.close(fig)
