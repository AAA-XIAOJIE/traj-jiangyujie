"""Agent-only ORCA, adapted to NumPy from RVO2 src/Agent.cc.

SPDX-FileCopyrightText: 2008 University of North Carolina at Chapel Hill
SPDX-License-Identifier: Apache-2.0
Source: snape/RVO2, commit 1c25b27258d191e26c2df83ff6e31abc8efbdaed.
Modifications (2026-09-18): Python port, vectorized pair constraints, no static
obstacles/KD tree, explicit feasibility diagnostics. See RVO2-LICENSE.txt.
"""
import numpy as np

EPS = 1e-9


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def lp1(lines, index, radius, preferred, direction_only):
    point, direction = lines[index]
    dot = float(point @ direction)
    disc = dot * dot + radius * radius - float(point @ point)
    if disc < -EPS:
        return None
    extent = np.sqrt(max(0.0, disc))
    left, right = -dot - extent, -dot + extent
    for other_point, other_direction in lines[:index]:
        denominator = det(direction, other_direction)
        numerator = det(other_direction, point - other_point)
        if abs(denominator) <= EPS:
            if numerator < -EPS:
                return None
            continue
        value = numerator / denominator
        if denominator >= 0:
            right = min(right, value)
        else:
            left = max(left, value)
        if left > right + EPS:
            return None
    if direction_only:
        t = right if preferred @ direction > 0 else left
    else:
        t = np.clip(direction @ (preferred - point), left, right)
    return point + t * direction


def lp2(lines, radius, preferred, direction_only=False):
    norm = np.linalg.norm(preferred)
    if direction_only:
        result = preferred * radius
    else:
        result = preferred * min(1.0, radius / max(norm, EPS))
    for i, (point, direction) in enumerate(lines):
        if det(direction, point - result) > EPS:
            candidate = lp1(lines, i, radius, preferred, direction_only)
            if candidate is None:
                return i, result
            result = candidate
    return len(lines), result


def lp3(lines, begin, radius, result):
    """RVO2 least-violation fallback; not a guarantee of feasible avoidance."""
    distance = 0.0
    for i in range(begin, len(lines)):
        point, direction = lines[i]
        if det(direction, point - result) <= distance + EPS:
            continue
        projected = []
        for other_point, other_direction in lines[:i]:
            determinant = det(direction, other_direction)
            if abs(determinant) <= EPS:
                if direction @ other_direction > 0:
                    continue
                intersection = 0.5 * (point + other_point)
            else:
                intersection = point + det(other_direction, point - other_point) / determinant * direction
            difference = other_direction - direction
            projected.append((intersection, difference / max(np.linalg.norm(difference), EPS)))
        failed, candidate = lp2(projected, radius, np.array([-direction[1], direction[0]]), True)
        if failed == len(projected):
            result = candidate
        distance = det(direction, point - result)
    return result


def pair_constraints(positions, velocities, radius, horizon, dt):
    """All ordered pairs; direction's left half-plane is feasible."""
    if radius <= 0 or horizon <= 0 or dt <= 0:
        raise ValueError("radius, horizon and dt must be positive")
    n = len(positions)
    relative_position = positions[None, :, :] - positions[:, None, :]
    relative_velocity = velocities[:, None, :] - velocities[None, :, :]
    d2 = np.sum(relative_position**2, axis=-1)
    combined = 2 * radius
    w = relative_velocity - relative_position / horizon
    w2 = np.sum(w**2, axis=-1)
    dot = np.sum(w * relative_position, axis=-1)
    cut = (dot < 0) & (dot**2 > combined**2 * w2)
    length = np.sqrt(np.maximum(w2, EPS**2))
    unit = w / length[..., None]
    direction = np.stack([unit[..., 1], -unit[..., 0]], axis=-1)
    correction = (combined / horizon - length)[..., None] * unit
    leg = np.sqrt(np.maximum(d2 - combined**2, 0))
    x, y = relative_position[..., 0], relative_position[..., 1]
    left = x * w[..., 1] - y * w[..., 0] > 0
    sign = np.where(left, 1.0, -1.0)
    legs = np.stack([sign * x * leg - y * combined,
                     x * combined + sign * y * leg], axis=-1) / np.maximum(d2, EPS)[..., None]
    leg_correction = np.sum(relative_velocity * legs, axis=-1)[..., None] * legs - relative_velocity
    direction = np.where(cut[..., None], direction, legs)
    correction = np.where(cut[..., None], correction, leg_correction)
    overlap = d2 <= combined**2
    overlap_w = relative_velocity - relative_position / dt
    overlap_len = np.linalg.norm(overlap_w, axis=-1)
    overlap_unit = overlap_w / np.maximum(overlap_len, EPS)[..., None]
    # Deterministic antisymmetric separation for exactly coincident agents.
    zero = overlap_len < EPS
    signs = np.where(np.arange(n)[:, None] < np.arange(n)[None, :], -1.0, 1.0)
    overlap_unit[zero, 0] = signs[zero]
    overlap_unit[zero, 1] = 0.0
    overlap_direction = np.stack([overlap_unit[..., 1], -overlap_unit[..., 0]], axis=-1)
    overlap_correction = (combined / dt - overlap_len)[..., None] * overlap_unit
    direction = np.where(overlap[..., None], overlap_direction, direction)
    correction = np.where(overlap[..., None], overlap_correction, correction)
    points = velocities[:, None, :] + 0.5 * correction
    return points, direction, d2


def step(positions, velocities, preferred, max_speeds, radius, horizon, dt):
    """Synchronous velocity selection; no post-projection velocity clipping."""
    points, directions, distances = pair_constraints(positions, velocities, radius, horizon, dt)
    result = np.empty_like(velocities)
    failures = 0
    max_violation = 0.0
    for i in range(len(positions)):
        neighbors = np.argsort(distances[i], kind="stable")
        neighbors = neighbors[neighbors != i]
        lines = list(zip(points[i, neighbors], directions[i, neighbors]))
        fail, candidate = lp2(lines, max_speeds[i], preferred[i])
        if fail < len(lines):
            failures += 1
            candidate = lp3(lines, fail, max_speeds[i], candidate)
        delta = points[i, neighbors] - candidate
        violation = directions[i, neighbors, 0] * delta[:, 1] - directions[i, neighbors, 1] * delta[:, 0]
        max_violation = max(max_violation, float(np.max(violation, initial=0)))
        result[i] = candidate
    return result, failures, max_violation
