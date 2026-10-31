"""Collision-avoidance maneuver planning.

An along-track impulse is applied to the primary some lead time before TCA. Its effect
is computed with a numerical two-body + J2 propagator as a *differential correction*
(burned minus un-burned trajectory) added on top of the SGP4 ephemeris, so the
secondary and the baseline primary keep their SGP4 motion. A Clohessy-Wiltshire
closed form is reported alongside as an analytic cross-check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from .objects import MU, R_EARTH, SpaceObject
from .probability import pc_for
from .screening import Conjunction, state

J2 = 1.08262668e-3


def _accel(t, y):
    r = y[:3]
    x, yy, z = r
    rn = np.linalg.norm(r)
    k = 1.5 * J2 * MU * R_EARTH**2 / rn**5
    zz = 5 * z**2 / rn**2
    a = -MU * r / rn**3 + k * np.array([x * (zz - 1), yy * (zz - 1), z * (zz - 3)])
    return np.concatenate([y[3:], a])


def _propagate(r0, v0, t_span_s, t_eval=None):
    return solve_ivp(_accel, (0.0, t_span_s), np.concatenate([r0, v0]), method="DOP853",
                     rtol=1e-11, atol=1e-9, dense_output=True, t_eval=t_eval)


@dataclass
class ManeuverResult:
    dv_ms: float                 # along-track delta-v, m/s (+ = prograde)
    lead_time_s: float
    new_tca_jd: float
    new_miss_km: float
    new_pc: float
    cw_along_shift_km: float     # analytic Clohessy-Wiltshire along-track shift at TCA
    cw_radial_shift_km: float


class AvoidancePlanner:
    def __init__(self, primary: SpaceObject, conj: Conjunction, lead_orbits: float = 1.5,
                 window_s: float = 120.0):
        self.primary = primary
        self.conj = conj
        n = primary.sat.no_kozai / 60.0                   # mean motion, rad/s
        self.n = n
        self.period = 2 * np.pi / n
        self.lead = lead_orbits * self.period
        self.window = window_s
        self.t_burn_jd = conj.tca_jd - self.lead / 86400.0
        self.rb, self.vb = state(primary, self.t_burn_jd)
        self.base = _propagate(self.rb, self.vb, self.lead + window_s)
        self.T_hat = self.vb / np.linalg.norm(self.vb)

    def _delta(self, dv_kms: float):
        burned = _propagate(self.rb, self.vb + dv_kms * self.T_hat, self.lead + self.window)
        return lambda s: burned.sol(s)[:3] - self.base.sol(s)[:3], \
            lambda s: burned.sol(s)[3:] - self.base.sol(s)[3:]

    def evaluate(self, dv_ms: float) -> ManeuverResult:
        dv = dv_ms / 1000.0
        dpos, dvel = self._delta(dv)
        sec = self.conj.secondary

        def rng(s_rel):  # s_rel: seconds relative to the original TCA
            jd = self.conj.tca_jd + s_rel / 86400.0
            r1, _ = state(self.primary, jd)
            r2, _ = state(sec, jd)
            return float(np.linalg.norm(r2 - (r1 + dpos(self.lead + s_rel))))

        res = minimize_scalar(rng, bounds=(-self.window, self.window), method="bounded",
                              options={"xatol": 1e-4})
        jd = self.conj.tca_jd + res.x / 86400.0
        r1, v1 = state(self.primary, jd)
        r2, v2 = state(sec, jd)
        r1 = r1 + dpos(self.lead + res.x)
        v1 = v1 + dvel(self.lead + res.x)
        pc = pc_for(r1, v1, r2, v2)

        nt = self.n * self.lead
        cw_along = (4 * np.sin(nt) / self.n - 3 * self.lead) * dv     # km
        cw_radial = (2 / self.n) * (1 - np.cos(nt)) * dv
        return ManeuverResult(dv_ms, self.lead, jd, float(res.fun), pc, float(cw_along), float(cw_radial))

    def sweep(self, dv_values_ms: np.ndarray) -> list[ManeuverResult]:
        return [self.evaluate(float(x)) for x in dv_values_ms]

    def minimum_dv(self, pc_target: float = 1e-4, max_dv_ms: float = 0.5,
                   n: int = 51) -> tuple[ManeuverResult | None, list[ManeuverResult]]:
        """Smallest |dv| (either direction) whose post-maneuver Pc is below ``pc_target``."""
        if self.conj.pc < pc_target:            # already acceptable: no maneuver needed
            return None, []
        grid = np.linspace(0.0, max_dv_ms, n)[1:]
        best = None
        results = []
        for sign in (+1, -1):
            for x in grid:
                r = self.evaluate(sign * float(x))
                results.append(r)
                if r.new_pc < pc_target:
                    if best is None or abs(r.dv_ms) < abs(best.dv_ms):
                        best = r
                    break
        return best, results
