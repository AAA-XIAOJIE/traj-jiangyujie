"""Three velocity controllers. No mass, thrust, forces or acceleration model."""
from __future__ import annotations
import numpy as np
from scipy import sparse
import osqp

EPS = 1e-12


def fibonacci_sphere(n):
    """Equal-area latitude bands with golden-angle azimuth, not exact equidistance."""
    k = np.arange(n, dtype=float)
    z = 1 - 2 * (k + 0.5) / n
    theta = np.pi * (3 - np.sqrt(5)) * k
    r = np.sqrt(1 - z * z)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta), z))


def cap_speed(v, limit):
    return v * np.minimum(1.0, limit / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), EPS))


def preferred(p, goal, cfg):
    delta = goal - p
    # Do not overshoot the goal even if a future configuration uses a larger dt.
    return cap_speed(delta / cfg['dt_s'], cfg['maximum_speed_m_s'])


def swept_distances(p, velocity, dt, pairs):
    """Exact minimum center distances over the entire piecewise-linear time step."""
    i, j = pairs
    relative = p[i] - p[j]
    dv = velocity[i] - velocity[j]
    t = np.clip(-np.sum(relative * dv, axis=1) / np.maximum(np.sum(dv * dv, axis=1), EPS), 0, dt)
    return np.linalg.norm(relative + t[:, None] * dv, axis=1)


class APF:
    """Potential-gradient-inspired *velocity* correction, not a physical force."""
    def __init__(self, cfg, initial):
        self.cfg = cfg

    def step(self, p, old_v, goal, stopped):
        c = self.cfg
        q = p[:, None] - p[None, :]
        d = np.linalg.norm(q, axis=-1)
        np.fill_diagonal(d, np.inf)
        influence = c['apf']['influence_m']
        # Dimension: gain m^2/s times inverse distance 1/m gives velocity m/s.
        weight = c['apf']['gain_m2_s'] * np.maximum(1 / np.maximum(d, .05) - 1 / influence, 0)
        correction = np.sum(weight[..., None] * q / np.maximum(d[..., None], EPS), axis=1)
        velocity = cap_speed(preferred(p, goal, c) + correction, c['maximum_speed_m_s'])
        velocity[stopped] = 0
        return velocity, {'fallback': 0, 'solver_failure': 0}


class VO:
    """Finite-horizon 3D velocity-obstacle sampling with constant-neighbor prediction.

    This is not ORCA: simultaneous changes of neighbors are not coordinated.
    """
    def __init__(self, cfg, initial):
        self.cfg = cfg
        self.directions = fibonacci_sphere(cfg['vo']['sphere_directions'])
        self.phi = np.arange(cfg['vo']['cone_azimuths']) * 2 * np.pi / cfg['vo']['cone_azimuths']

    def step(self, p, old_v, goal, stopped):
        c = self.cfg
        pref = preferred(p, goal, c)
        vmax = c['maximum_speed_m_s']
        separation = 2 * c['body_radius_m']
        out = np.zeros_like(p)
        fallback = 0
        for i in np.flatnonzero(~stopped):
            e = pref[i] / max(np.linalg.norm(pref[i]), EPS)
            axis = np.eye(3)[np.argmin(np.abs(e))]
            a = np.cross(e, axis); a /= np.linalg.norm(a)
            b = np.cross(e, a)
            cones = []
            for angle in np.deg2rad(c['vo']['cone_angles_deg']):
                cones.append(vmax * (np.cos(angle) * e + np.sin(angle) * (np.cos(self.phi)[:, None] * a + np.sin(self.phi)[:, None] * b)))
            candidates = np.vstack((pref[i], .5 * pref[i], old_v[i], np.zeros(3), *cones, vmax * self.directions, .5 * vmax * self.directions))
            q = p - p[i]
            # Beyond this distance no pair can close to the safety threshold in tau.
            neighbor = (np.linalg.norm(q, axis=1) <= 2 * vmax * c['horizon_s'] + separation)
            neighbor[i] = False
            q = q[neighbor]
            if len(q):
                w = candidates[:, None, :] - old_v[None, neighbor, :]
                t = np.clip(np.sum(w * q, axis=-1) / np.maximum(np.sum(w * w, axis=-1), EPS), 0, c['horizon_s'])
                closest = np.linalg.norm(q - t[..., None] * w, axis=-1).min(axis=1)
            else:
                closest = np.full(len(candidates), np.inf)
            cost = np.sum((candidates - pref[i])**2, axis=1) + c['vo']['inertia_weight'] * np.sum((candidates - old_v[i])**2, axis=1)
            safe = closest >= separation
            if safe.any():
                index = np.argmin(np.where(safe, cost, np.inf))
            else:
                # Honest failed-search behavior: safest sampled candidate, no hidden teleport.
                fallback += 1
                best = closest.max()
                index = np.argmin(np.where(closest >= best - 1e-10, cost, np.inf))
            out[i] = candidates[index]
        return out, {'fallback': fallback, 'solver_failure': 0}


class CBF:
    """Centralized joint velocity QP with a distance barrier and circulation guidance.

    n_ij dot (v_i-v_j) >= -(d_ij-D)/tau guarantees projected separation >= D
    within one dt <= tau, when current d_ij >= D and the constraints are feasible.
    Stopped agents remain obstacles with exactly zero velocity.
    """
    def __init__(self, cfg, initial):
        self.cfg = cfg
        self.n = len(initial)
        self.i, self.j = np.triu_indices(self.n, 1)
        self.m = len(self.i)
        # Fixed sparse pattern permits solver warm starts while normals change.
        rows = np.repeat(np.arange(self.m), 6)
        columns = np.column_stack((3*self.i, 3*self.i+1, 3*self.i+2, 3*self.j, 3*self.j+1, 3*self.j+2)).ravel()
        self.rows, self.columns = rows, columns
        self.solver = None
        self.axis = np.asarray(cfg['cbf']['axis'], float)
        self.axis /= np.linalg.norm(self.axis)

    def step(self, p, old_v, goal, stopped):
        c = self.cfg
        dgoal = goal - p
        nominal = preferred(p, goal, c)
        radius = np.linalg.norm(p, axis=1)
        swirl = np.cross(self.axis, p)
        swirl /= np.maximum(np.linalg.norm(swirl, axis=1, keepdims=True), EPS)
        # Fade circulation near the goal; no future ground truth is available/used.
        progress = np.clip(np.linalg.norm(dgoal, axis=1) / 5, 0, 1)
        strength = c['cbf']['swirl_speed_m_s'] * np.exp(-(radius / c['cbf']['swirl_radius_m'])**2) * progress
        nominal = cap_speed(nominal + strength[:, None] * swirl, c['maximum_speed_m_s'])
        nominal[stopped] = 0
        q = p[self.i] - p[self.j]
        distance = np.linalg.norm(q, axis=1)
        normal = q / np.maximum(distance[:, None], EPS)
        values = np.column_stack((normal, -normal)).ravel()
        pair_matrix = sparse.csc_matrix((values, (self.rows, self.columns)), shape=(self.m, 3*self.n))
        matrix = sparse.vstack((pair_matrix, sparse.eye(3*self.n, format='csc')), format='csc')
        safe_distance = 2*c['body_radius_m'] + c['cbf']['separation_margin_m']
        bounds = -(distance - safe_distance) / c['horizon_s']
        limits = np.full((self.n, 3), c['maximum_speed_m_s'])
        limits[stopped] = 0
        lower = np.r_[bounds, -limits.ravel()]
        upper = np.r_[np.full(self.m, np.inf), limits.ravel()]
        if self.solver is None:
            self.solver = osqp.OSQP()
            self.solver.setup(P=sparse.eye(3*self.n, format='csc'), q=-nominal.ravel(), A=matrix, l=lower, u=upper,
                              verbose=False, eps_abs=c['cbf']['eps_abs'], eps_rel=c['cbf']['eps_rel'],
                              max_iter=c['cbf']['max_iter'], polishing=False, adaptive_rho=True)
        else:
            self.solver.update(q=-nominal.ravel(), Ax=matrix.data, l=lower, u=upper)
        result = self.solver.solve(raise_error=False)
        if result.info.status_val not in (1, 2) or result.x is None or not np.isfinite(result.x).all():
            return np.zeros_like(p), {'fallback': 1, 'solver_failure': 1}
        velocity = result.x.reshape(self.n, 3)
        velocity[stopped] = 0
        # A common contraction preserves ALL pair inequalities with nonpositive RHS;
        # clipping individual velocities would not preserve joint constraints.
        scale = min(1., c['maximum_speed_m_s'] / max(np.linalg.norm(velocity, axis=1).max(), EPS))
        velocity *= scale
        projected = np.sum(normal * (velocity[self.i] - velocity[self.j]), axis=1)
        if np.min(projected - bounds) < -1e-5:
            return np.zeros_like(p), {'fallback': 1, 'solver_failure': 1}
        return velocity, {'fallback': 0, 'solver_failure': 0}


CONTROLLERS = {'apf': APF, 'vo': VO, 'cbf': CBF}
