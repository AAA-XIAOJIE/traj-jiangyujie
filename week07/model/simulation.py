"""Shared synchronous simulator and interval-level metrics for all controllers."""
import time
import numpy as np
from .core import CONTROLLERS, fibonacci_sphere, swept_distances


def simulate(cfg, name, verbose=False):
    started = time.perf_counter()
    n = cfg['agents']; dt = cfg['dt_s']
    steps = int(round(cfg['duration_s']/dt))
    initial = cfg['sphere_radius_m'] * fibonacci_sphere(n)
    goal = -initial
    p = initial.copy(); old_v = np.zeros_like(p)
    controller = CONTROLLERS[name](cfg, initial)
    pairs = np.triu_indices(n, 1)
    arrived = np.full(n, np.nan)
    stopped = np.zeros(n, bool)
    positions = [p.copy()]; velocities = [old_v.copy()]
    distance_history = []; count_history = []; path_length = np.zeros(n)
    offending = np.zeros(len(pairs[0]), bool)
    pair_steps = 0; fallback = 0; solver_failure = 0
    for k in range(steps):
        velocity, diagnostic = controller.step(p, old_v, goal, stopped)
        if not np.isfinite(velocity).all():
            raise FloatingPointError(f'{name}: nonfinite velocity at step {k}')
        if np.linalg.norm(velocity, axis=1).max() > cfg['maximum_speed_m_s'] + 1e-7:
            raise AssertionError(f'{name}: speed limit exceeded')
        distances = swept_distances(p, velocity, dt, pairs)
        intrusion = distances < 2*cfg['body_radius_m'] - cfg['distance_tolerance_m']
        offending |= intrusion; pair_steps += int(intrusion.sum())
        distance_history.append(float(distances.min()))
        path_length += np.linalg.norm(velocity, axis=1) * dt
        p = p + dt * velocity
        newly = (~stopped) & (np.linalg.norm(p - goal, axis=1) <= cfg['goal_tolerance_m'] + 1e-10)
        arrived[newly] = (k+1)*dt
        stopped |= newly
        old_v = velocity.copy()
        old_v[stopped] = 0
        # Velocities stored at each sample describe the state at that sample;
        # actual segment displacement is unambiguously given by position differences.
        positions.append(p.copy()); velocities.append(old_v.copy())
        count_history.append(int(stopped.sum()))
        fallback += diagnostic['fallback']; solver_failure += diagnostic['solver_failure']
        if verbose and (k+1) % 200 == 0:
            print(f'{name}: t={(k+1)*dt:.1f}s arrived={stopped.sum()}/{n} min={min(distance_history):.4f}m', flush=True)
        if stopped.all():
            break
    metrics = dict(model=name, agents=n, arrived=int(stopped.sum()), arrival_rate=float(stopped.mean()),
                   minimum_separation_m=float(min(distance_history)), violating_unique_pairs=int(offending.sum()),
                   violating_pair_steps=pair_steps, all_arrival_time_s=float(np.nanmax(arrived)) if stopped.all() else None,
                   mean_path_length_m=float(path_length.mean()), simulated_time_s=(k+1)*dt,
                   max_speed_m_s=float(np.linalg.norm(np.diff(np.asarray(positions), axis=0)/dt, axis=2).max()),
                   fallback_agent_or_system_steps=fallback, solver_failure_steps=solver_failure,
                   runtime_s=time.perf_counter()-started,
                   collision_free_complete=bool(stopped.all() and not offending.any()))
    trace = dict(positions=np.asarray(positions), velocities=np.asarray(velocities), goals=goal,
                 times=np.arange(k+2)*dt, arrival_times=arrived, path_lengths=path_length,
                 minimum_separation=np.asarray(distance_history), arrival_counts=np.asarray(count_history),
                 violation_pairs=np.column_stack(pairs)[offending])
    return metrics, trace
