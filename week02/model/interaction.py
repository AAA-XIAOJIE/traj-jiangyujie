"""A measured encounter compared with a constant-velocity counterfactual."""
import json
import numpy as np
import matplotlib.pyplot as plt

from week01 import style
from .analysis import velocity
from .plots import panel, INK


def time_to_collision(relative_position, relative_velocity, combined_radius=0.5):
    a = np.sum(relative_velocity**2, axis=-1)
    b = np.sum(relative_position * relative_velocity, axis=-1)
    c = np.sum(relative_position**2, axis=-1) - combined_radius**2
    discriminant = b * b - a * c
    valid = (a > 1e-12) & (b > 0) & (c > 0) & (discriminant >= 0)
    forecast = np.where(valid, (b - np.sqrt(np.maximum(discriminant, 0))) / np.maximum(a, 1e-12), np.inf)
    return np.where(c <= 0, 0.0, forecast)


def make_encounter_figure(experiment, out):
    p, times = experiment["positions"], experiment["times"]
    dt = times[1] - times[0]
    v = velocity(p, dt)
    a, b = np.triu_indices(p.shape[1], 1)
    best = None
    # Display selection only; none of this episode is passed to the simulator.
    for k in np.flatnonzero((times >= 2) & (times <= 8))[::2]:
        relative = p[k, b] - p[k, a]
        relative_v = v[k, a] - v[k, b]
        ttc = time_to_collision(relative, relative_v)
        spacing = np.linalg.norm(relative, axis=-1)
        candidates = np.flatnonzero((ttc > .3) & (ttc < 1.5) & (spacing > .7) & (spacing < 3))
        for pair in candidates:
            i, j = a[pair], b[pair]
            future = min(len(times), k + int(round(2 / dt)) + 1)
            observed_min = np.linalg.norm(p[k:future, i] - p[k:future, j], axis=-1).min()
            if observed_min < .6:
                continue
            after = min(k + int(round(.8 / dt)), len(times) - 1)
            initial_speed = np.linalg.norm(v[k, [i, j]], axis=-1)
            later_speed = np.linalg.norm(v[after, [i, j]], axis=-1)
            angle = np.rad2deg(np.arccos(np.clip(np.sum(v[k, [i, j]] * v[after, [i, j]], axis=-1) /
                                             np.maximum(initial_speed * later_speed, 1e-12), -1, 1)))
            drop = initial_speed - later_speed
            if angle.max() < 12 or drop.max() < .15:
                continue
            score = observed_min + .01 * min(angle.max(), 60) + .2 * drop.max()
            if best is None or score > best[0]:
                best = (score, k, i, j, float(ttc[pair]), float(observed_min), angle, drop)
    if best is None:
        raise ValueError("No encounter satisfies the stated display-selection criteria")
    _, k, i, j, ttc, minimum, angle, drop = best
    begin, end = max(0, k - 15), min(len(times), k + 51)
    local_time = times[begin:end] - times[k]
    future_time = np.linspace(0, 2, 101)
    colors = ["#0072B2", "#D55E00"]
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.7))
    fig.subplots_adjust(left=.075, right=.98, bottom=.22, top=.75, wspace=.34)
    fig.suptitle("A local conflict forecast and the observed response", x=.075, y=.965, ha="left", fontsize=17, weight="bold", color=INK)
    fig.text(.075, .875, f"People {experiment['ids'][i]} and {experiment['ids'][j]} at t = {times[k]:.2f} s | frozen-velocity TTC = {ttc:.2f} s | later measured minimum spacing = {minimum:.2f} m", fontsize=9, color="#65747E")
    for index, color in zip([i, j], colors):
        label = f"Person {experiment['ids'][index]}"
        axes[0].plot(p[begin:end, index, 0], p[begin:end, index, 1], color=color, label=label)
        projected = p[k, index] + future_time[:, None] * v[k, index]
        axes[0].plot(projected[:, 0], projected[:, 1], "--", color=color, alpha=.6)
        axes[0].scatter(*p[k, index], color=color, s=28)
        axes[2].plot(local_time, np.linalg.norm(v[begin:end, index], axis=-1), color=color, label=label)
    axes[0].set(xlabel="World x (m)", ylabel="World y (m)", aspect="equal")
    panel(axes[0], "a", "Measured and projected paths")
    axes[0].legend(frameon=False, fontsize=7)
    real_spacing = np.linalg.norm(p[begin:end, i] - p[begin:end, j], axis=-1)
    prediction = np.linalg.norm(p[k, j] - p[k, i] - future_time[:, None] * (v[k, i] - v[k, j]), axis=-1)
    axes[1].plot(local_time, real_spacing, color="#009E73", label="Observed")
    axes[1].plot(future_time, prediction, "--", color="#777777", label="Frozen-velocity forecast")
    axes[1].axhline(.5, color="#CC79A7", linestyle=":", label="Assumed disk diameter")
    axes[1].set(xlabel="Time from selected instant (s)", ylabel="Center-to-center distance (m)", ylim=(0, None))
    panel(axes[1], "b", "Forecast conflict did not persist")
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].set(xlabel="Time from selected instant (s)", ylabel="Speed (m/s)", ylim=(0, None))
    axes[2].axvline(0, color="#888888", lw=.7, linestyle=":")
    panel(axes[2], "c", "Speed adjustment accompanies turning")
    axes[2].legend(frameon=False, fontsize=7)
    fig.text(.075, .065, "TTC is a constant-velocity prediction with a 0.5 m disk threshold, not a measured collision time. Dashed paths are counterfactual.", fontsize=8, color="#65747E")
    fig.text(.075, .03, "Episode selected retrospectively for a clear turn and slowdown; trajectories alone do not establish perception, intention, or causal response.", fontsize=8, color="#65747E")
    style.save(fig, out / "interaction_detail", dpi=300)
    receipt = {"people": [int(experiment["ids"][i]), int(experiment["ids"][j])], "time_s": float(times[k]),
               "predicted_ttc_s": ttc, "observed_min_next_2s_m": minimum,
               "heading_change_over_0p8s_deg": angle.tolist(), "speed_drop_over_0p8s_mps": drop.tolist(),
               "selection": "t=2..8s every 0.08s; TTC 0.3..1.5s; initial spacing 0.7..3m; later min >=0.6m; max turn >=12deg; max slowdown >=0.15m/s; maximize clearance+0.01*capped_turn+0.2*slowdown"}
    (out / "interaction.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    return receipt
