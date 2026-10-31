"""Space-object catalog: load real TLEs from CelesTrak, or build a synthetic demo cloud.

Every object is wrapped as a ``SpaceObject`` holding an ``sgp4`` ``Satrec`` so the
rest of the pipeline uses one propagation path (SGP4) for real and demo data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, WGS72, jday

MU = 398600.4418          # km^3/s^2
R_EARTH = 6378.137        # km
CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php"


@dataclass
class SpaceObject:
    name: str
    norad_id: int
    sat: Satrec
    is_primary: bool = False
    seeded_tca_jd: float | None = None   # demo only: when the crosser was placed


# --------------------------------------------------------------------------- real data
def _download(params: dict, cache: Path, max_age_h: float = 12.0) -> str:
    """Download a TLE set from CelesTrak, caching it for ``max_age_h`` hours.

    TLEs go stale within days, so the cache is refreshed after ``max_age_h``. If the
    refresh fails (offline, rate-limited) an existing cached copy is used instead.
    """
    import time

    if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_h * 3600:
        return cache.read_text()
    import requests

    try:
        resp = requests.get(CELESTRAK, params={**params, "FORMAT": "tle"}, timeout=30)
        if resp.status_code == 404 or "No GP data found" in resp.text:
            raise LookupError(f"CelesTrak has no data for {params} (check the NORAD ID or group name)")
        resp.raise_for_status()
        text = resp.text.strip()
        if not text:
            raise LookupError(f"CelesTrak returned an empty response for {params}")
    except LookupError:
        raise
    except Exception:
        if cache.exists():                      # offline: fall back to the stale copy
            return cache.read_text()
        raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text


def _parse_tle_text(text: str) -> list[SpaceObject]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    objs = []
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue
        sat = Satrec.twoline2rv(l1, l2)
        objs.append(SpaceObject(name=name, norad_id=sat.satnum, sat=sat))
    return objs


def load_real(primary_id: int, debris_groups: list[str], data_dir: Path) -> tuple[SpaceObject, list[SpaceObject]]:
    """Primary satellite by NORAD ID + one or more CelesTrak GROUP names for secondaries."""
    ptext = _download({"CATNR": primary_id}, data_dir / f"primary_{primary_id}.tle")
    primary = _parse_tle_text(ptext)[0]
    primary.is_primary = True

    secondaries: list[SpaceObject] = []
    for g in debris_groups:
        secondaries += _parse_tle_text(_download({"GROUP": g}, data_dir / f"group_{g}.tle"))
    secondaries = [s for s in secondaries if s.norad_id != primary.norad_id]
    return primary, secondaries


# --------------------------------------------------------------------------- synthetic demo
def _epoch_days_since_1949(jd: float, fr: float) -> float:
    return (jd + fr) - 2433281.5


def _make_sat(name: str, norad: int, a_km: float, e: float, inc: float, raan: float,
              argp: float, M: float, jd: float, fr: float, bstar: float = 2e-5) -> SpaceObject:
    n_rad_min = math.sqrt(MU / a_km**3) * 60.0
    sat = Satrec()
    sat.sgp4init(WGS72, "i", norad, _epoch_days_since_1949(jd, fr), bstar, 0.0, 0.0,
                 e, argp, inc, M, n_rad_min, raan)
    return SpaceObject(name=name, norad_id=norad, sat=sat)


def _rv2coe(r: np.ndarray, v: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Cartesian state -> classical elements (a, e, i, RAAN, argp, M), radians."""
    h = np.cross(r, v)
    nvec = np.cross([0, 0, 1.0], h)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    evec = ((vn**2 - MU / rn) * r - np.dot(r, v) * v) / MU
    e = np.linalg.norm(evec)
    a = 1.0 / (2.0 / rn - vn**2 / MU)
    inc = math.acos(h[2] / np.linalg.norm(h))
    raan = math.atan2(nvec[1], nvec[0]) % (2 * math.pi)
    nn = np.linalg.norm(nvec)
    if e > 1e-6:
        argp = math.acos(np.clip(np.dot(nvec, evec) / (nn * e), -1, 1))
        if evec[2] < 0:
            argp = 2 * math.pi - argp
        nu = math.acos(np.clip(np.dot(evec, r) / (e * rn), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * math.pi - nu
    else:  # near-circular: measure from the node
        argp = 0.0
        nu = math.acos(np.clip(np.dot(nvec, r) / (nn * rn), -1, 1))
        if r[2] < 0:
            nu = 2 * math.pi - nu
    E = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(nu / 2))
    M = (E - e * math.sin(E)) % (2 * math.pi)
    return a, e, inc, raan, argp, M


def build_demo(n_background: int = 400, n_crossers: int = 6, seed: int = 7,
               start=(2026, 11, 1, 0, 0, 0.0)) -> tuple[SpaceObject, list[SpaceObject]]:
    """Synthetic scenario: an ISS-like primary plus a fragmentation-style debris cloud.

    A handful of 'crosser' fragments are seeded so that they pass close to the primary
    at known times; the rest form a realistic background population at similar altitudes.
    This exists only so the pipeline can be tested without network access — real runs
    should use ``load_real``.
    """
    rng = np.random.default_rng(seed)
    jd, fr = jday(*start)

    primary = _make_sat("DEMO-PRIMARY (ISS-like)", 90000, R_EARTH + 420.0, 0.0005,
                        math.radians(51.64), math.radians(120.0), math.radians(30.0), 0.0, jd, fr)
    primary.is_primary = True

    debris: list[SpaceObject] = []
    norad = 91000
    # background cloud (e.g. a breakup at ~800 km decaying through lower shells)
    for _ in range(n_background):
        alt = rng.uniform(380, 900)
        debris.append(_make_sat(
            f"DEMO-DEB {norad}", norad, R_EARTH + alt, rng.uniform(0.0005, 0.02),
            math.radians(rng.normal(98.0, 6.0) if rng.random() < 0.6 else rng.uniform(45, 75)),
            rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi),
            jd, fr, bstar=rng.uniform(1e-5, 2e-4)))
        norad += 1

    # crossers: take the primary's state at a chosen time, offset it slightly, rotate the
    # velocity to a different orbit plane, convert to elements and back-date to the epoch.
    for k in range(n_crossers):
        t_cross_min = rng.uniform(0.6, 6.5) * 1440.0
        jd_c, fr_c = jd, fr + t_cross_min / 1440.0
        err, r, v = primary.sat.sgp4(jd_c, fr_c)
        r, v = np.array(r), np.array(v)
        rhat = r / np.linalg.norm(r)
        offset = rng.normal(0, 1.0, 3)
        offset -= offset.dot(rhat) * rhat * 0.7
        r2 = r + offset * rng.uniform(0.2, 1.5)
        ang = math.radians(rng.uniform(20, 160))          # crossing angle
        k_axis = rhat
        v2 = (v * math.cos(ang) + np.cross(k_axis, v) * math.sin(ang)
              + k_axis * np.dot(k_axis, v) * (1 - math.cos(ang)))
        # The fragment's element set is epoched at the crossing time (SGP4 propagates it
        # backwards/forwards). SGP4 treats inputs as *mean* elements, so a few fixed-point
        # iterations remove the mean/osculating mismatch and land it on the target state.
        r_tgt, v_tgt = r2.copy(), v2.copy()
        for _ in range(8):
            a, e, inc, raan, argp, M = _rv2coe(r_tgt, v_tgt)
            frag = _make_sat(f"DEMO-FRAG {norad}", norad, a, max(e, 1e-4), inc, raan, argp,
                             M, jd_c, fr_c, bstar=1e-5)
            _, rg, vg = frag.sat.sgp4(jd_c, fr_c)
            r_tgt += r2 - np.array(rg)
            v_tgt += v2 - np.array(vg)
        frag.seeded_tca_jd = jd_c + fr_c
        debris.append(frag)
        norad += 1
    return primary, debris
