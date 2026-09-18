"""Goal navigation plus persistent heterogeneous side preference, then ORCA."""
import numpy as np

from .orca import step


def preferred_velocities(positions, velocities, goals, desired_speeds, t, initial_speeds,
                         dt, bias=0.0, anticipation=3.0, start_response=0.5):
    toward = goals - positions
    remaining = np.linalg.norm(toward, axis=1)
    heading = toward / np.maximum(remaining[:, None], 1e-12)
    nominal_speed = desired_speeds + (initial_speeds - desired_speeds) * np.exp(-t / start_response)
    base = heading * nominal_speed[:, None]
    if np.any(bias):
        relative_position = positions[None, :, :] - positions[:, None, :]
        relative_velocity = base[:, None, :] - velocities[None, :, :]
        closing = np.sum(relative_position * relative_velocity, axis=-1)
        tca = closing / np.maximum(np.sum(relative_velocity**2, axis=-1), 1e-12)
        miss = np.linalg.norm(relative_position - np.clip(tca, 0, anticipation)[..., None] * relative_velocity, axis=-1)
        ahead = np.sum(relative_position * heading[:, None, :], axis=-1) > 0
        risk = np.exp(-(miss / 1.0)**2) * np.clip(1 - tca / anticipation, 0, 1)
        risk *= (tca > 0) & (tca < anticipation) & ahead
        np.fill_diagonal(risk, 0)
        angle = bias * risk.max(axis=1)
        right = np.column_stack([heading[:, 1], -heading[:, 0]])
        heading = np.cos(angle)[:, None] * heading + np.sin(angle)[:, None] * right
    # A goal is a destination, not a prescribed measured endpoint; avoid overshoot.
    speed = np.minimum(nominal_speed, remaining / max(0.4, dt))
    return heading * speed[:, None]


def simulate(initial, initial_velocity, goals, desired, times, config, bias=0.0,
             horizon=None, radius=None, avoid=True, speed_seed=None, substeps=1,
             side_seed=None, right_probability=0.65):
    positions, velocities = initial.copy(), initial_velocity.copy()
    desired = desired.copy()
    signed_bias = bias
    if side_seed is not None:
        signs = np.where(np.random.default_rng(side_seed).random(len(desired)) < right_probability, 1.0, -1.0)
        signed_bias = bias * signs
    if speed_seed is not None:
        desired *= np.random.default_rng(speed_seed).uniform(0.9, 1.1, len(desired))
    max_speeds = np.maximum(config["max_speed_factor"] * desired, np.linalg.norm(initial_velocity, axis=1))
    dt = (times[1] - times[0]) / substeps
    horizon = config["time_horizon_s"] if horizon is None else horizon
    radius = config["body_radius_m"] if radius is None else radius
    output = np.empty((len(times), len(initial), 2))
    output[0] = initial
    failures = 0
    max_violation = 0.0
    max_accel = 0.0
    integration_min_spacing = float("inf")
    integration_overlap_steps = 0
    pair_i, pair_j = np.triu_indices(len(initial), 1)
    for frame in range(1, len(times)):
        for sub in range(substeps):
            t = times[frame - 1] + sub * dt
            preferred = preferred_velocities(positions, velocities, goals, desired, t,
                                            np.linalg.norm(initial_velocity, axis=1), dt,
                                            signed_bias, config["anticipation_s"], config["start_response_s"])
            if avoid:
                new_velocity, failed, violation = step(positions, velocities, preferred, max_speeds,
                                                      radius, horizon, dt)
                failures += failed
                max_violation = max(max_violation, violation)
            else:
                new_velocity = preferred
            max_accel = max(max_accel, float(np.linalg.norm(new_velocity - velocities, axis=1).max() / dt))
            relative = positions[pair_i] - positions[pair_j]
            relative_step = dt * (new_velocity[pair_i] - new_velocity[pair_j])
            fraction = np.clip(-np.sum(relative * relative_step, axis=1) / np.maximum(np.sum(relative_step**2, axis=1), 1e-20), 0, 1)
            swept = np.linalg.norm(relative + fraction[:, None] * relative_step, axis=1)
            integration_min_spacing = min(integration_min_spacing, float(swept.min()))
            integration_overlap_steps += int(np.sum(swept < 2 * radius - 1e-5))
            positions = positions + dt * new_velocity
            velocities = new_velocity
        output[frame] = positions
    return output, {"lp_fallback_agent_steps": failures, "max_constraint_violation_mps": max_violation,
                    "integration_min_spacing_m": integration_min_spacing,
                    "integration_overlap_pair_steps": integration_overlap_steps,
                    "integration_overlap_pair_seconds": integration_overlap_steps * dt,
                    "max_disk_penetration_mm": max(0.0, 2 * radius - integration_min_spacing) * 1000,
                    "max_command_acceleration_mps2": max_accel, "integration_dt_s": dt,
                    "horizon_s": horizon, "radius_m": radius, "right_bias_rad": bias,
                    "speed_seed": speed_seed, "side_seed": side_seed,
                    "right_preference_probability": right_probability if side_seed is not None else None}
