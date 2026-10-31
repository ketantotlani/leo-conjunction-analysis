"""SGP4 propagation and conjunction screening.

Pipeline: apogee/perigee pre-filter -> vectorised SGP4 on a coarse time grid ->
local minima of range -> Brent refinement of the time of closest approach (TCA).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.optimize import minimize_scalar
from sgp4.api import SatrecArray

from .objects import MU, R_EARTH, SpaceObject

JD_UNIX_EPOCH = 2440587.5


@dataclass
class Conjunction:
    secondary: SpaceObject
    tca_jd: float            # full Julian date of TCA (UTC)
    miss_km: float
    r1: np.ndarray           # primary position at TCA (TEME, km)
    v1: np.ndarray           # primary velocity at TCA (km/s)
    r2: np.ndarray
    v2: np.ndarray
    pc: float = float("nan")

    @property
    def rel_speed(self) -> float:
        return float(np.linalg.norm(self.v2 - self.v1))

    @property
    def tca_utc(self) -> datetime:
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=self.tca_jd - JD_UNIX_EPOCH)


def state(obj: SpaceObject, jd_full: float) -> tuple[np.ndarray, np.ndarray]:
    jd = np.floor(jd_full - 0.5) + 0.5
    err, r, v = obj.sat.sgp4(jd, jd_full - jd)
    if err:
        raise RuntimeError(f"SGP4 error {err} for {obj.name}")
    return np.array(r), np.array(v)


def _perigee_apogee(obj: SpaceObject) -> tuple[float, float]:
    n = obj.sat.no_kozai / 60.0                 # rad/s
    a = (MU / n**2) ** (1 / 3)
    e = obj.sat.ecco
    return a * (1 - e) - R_EARTH, a * (1 + e) - R_EARTH


def prefilter(primary: SpaceObject, others: list[SpaceObject], margin_km: float = 25.0) -> list[SpaceObject]:
    """Drop objects whose altitude band can never overlap the primary's."""
    pp, pa = _perigee_apogee(primary)
    keep = []
    for o in others:
        op, oa = _perigee_apogee(o)
        if op <= pa + margin_km and oa >= pp - margin_km:
            keep.append(o)
    return keep


def screen(primary: SpaceObject, others: list[SpaceObject], start_jd: float, days: float,
           threshold_km: float = 5.0, step_s: float = 30.0, batch: int = 200) -> list[Conjunction]:
    """Return all refined conjunctions with miss distance below ``threshold_km``."""
    n = int(days * 86400 / step_s) + 1
    t = start_jd + np.arange(n) * step_s / 86400.0
    jd = np.floor(t - 0.5) + 0.5
    fr = t - jd

    err_p, rp, _ = SatrecArray([primary.sat]).sgp4(jd, fr)
    rp = rp[0]
    coarse_gate = threshold_km + 16.0 * step_s     # max LEO closing speed ~16 km/s

    found: list[Conjunction] = []
    for b in range(0, len(others), batch):
        chunk = others[b:b + batch]
        err, rs, _ = SatrecArray([o.sat for o in chunk]).sgp4(jd, fr)
        d = np.linalg.norm(rs - rp[None, :, :], axis=2)
        d[err != 0] = np.inf
        for k, o in enumerate(chunk):
            dk = d[k]
            idx = np.where((dk[1:-1] <= dk[:-2]) & (dk[1:-1] <= dk[2:]) & (dk[1:-1] < coarse_gate))[0] + 1
            for i in idx:
                c = _refine(primary, o, t[i], step_s)
                if c is not None and c.miss_km < threshold_km:
                    found.append(c)
    found.sort(key=lambda c: c.miss_km)
    return found


def _refine(primary: SpaceObject, other: SpaceObject, t_guess: float, step_s: float) -> Conjunction | None:
    """Brent search for the range minimum, parameterised in seconds from ``t_guess``
    (Julian dates are too large for sub-second tolerances in float64)."""
    def rng(s: float) -> float:
        x = t_guess + s / 86400.0
        return float(np.linalg.norm(state(other, x)[0] - state(primary, x)[0]))

    try:
        res = minimize_scalar(rng, bounds=(-step_s, step_s), method="bounded", options={"xatol": 1e-4})
    except RuntimeError:
        return None
    tca = t_guess + res.x / 86400.0
    r1, v1 = state(primary, tca)
    r2, v2 = state(other, tca)
    return Conjunction(other, float(tca), float(np.linalg.norm(r2 - r1)), r1, v1, r2, v2)


def range_history(primary: SpaceObject, other: SpaceObject, center_jd: float,
                  half_window_s: float = 300.0, n: int = 601) -> tuple[np.ndarray, np.ndarray]:
    """Range vs time (seconds from TCA) around an encounter, for plotting."""
    ts = np.linspace(-half_window_s, half_window_s, n)
    d = np.array([np.linalg.norm(state(other, center_jd + s / 86400)[0] - state(primary, center_jd + s / 86400)[0])
                  for s in ts])
    return ts, d
