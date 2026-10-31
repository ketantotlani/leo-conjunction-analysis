"""Tests for the LEO conjunction pipeline (offline: uses the synthetic demo catalog)."""
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from sgp4.api import jday

from leoca.maneuver import AvoidancePlanner
from leoca.objects import _download, _parse_tle_text, build_demo
from leoca.probability import Encounter, pc_foster, pc_for
from leoca.screening import prefilter, screen, state

START = sum(jday(2026, 11, 1, 0, 0, 0))
ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""


@pytest.fixture(scope="module")
def demo():
    primary, debris = build_demo()
    conjs = screen(primary, prefilter(primary, debris), START, 7.0, 5.0)
    for c in conjs:
        c.pc = pc_for(c.r1, c.v1, c.r2, c.v2)
    conjs.sort(key=lambda c: -c.pc)
    return primary, debris, conjs


# ----------------------------------------------------------------------------- catalog
def test_parse_tle():
    objs = _parse_tle_text(ISS_TLE)
    assert len(objs) == 1 and objs[0].norad_id == 25544 and objs[0].name == "ISS (ZARYA)"


def test_demo_crossers_hit_their_targets():
    primary, debris = build_demo()
    seeded = [o for o in debris if o.seeded_tca_jd]
    assert len(seeded) == 6
    for o in seeded:
        r1, _ = state(primary, o.seeded_tca_jd)
        r2, _ = state(o, o.seeded_tca_jd)
        assert np.linalg.norm(r2 - r1) < 3.0          # km


def test_prefilter_drops_non_overlapping_shells():
    primary, debris = build_demo()
    kept = prefilter(primary, debris)
    assert 0 < len(kept) < len(debris)


def test_cache_is_used_when_fresh(tmp_path: Path):
    cache = tmp_path / "x.tle"
    cache.write_text(ISS_TLE)
    assert _download({"CATNR": 25544}, cache) == ISS_TLE     # no network needed


def test_stale_cache_used_when_offline(tmp_path: Path, monkeypatch):
    cache = tmp_path / "x.tle"
    cache.write_text(ISS_TLE)
    old = time.time() - 3 * 86400
    os.utime(cache, (old, old))
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", boom)
    assert _download({"CATNR": 25544}, cache) == ISS_TLE


# ----------------------------------------------------------------------------- screening
def test_screening_finds_all_seeded_crossers(demo):
    _, debris, conjs = demo
    found = {c.secondary.norad_id for c in conjs}
    seeded = {o.norad_id for o in debris if o.seeded_tca_jd}
    assert seeded <= found


def test_tca_is_a_local_minimum(demo):
    primary, _, conjs = demo
    c = conjs[0]
    for dt in (-0.05, 0.05):                            # +-50 ms
        jd = c.tca_jd + dt / 86400
        d = np.linalg.norm(state(c.secondary, jd)[0] - state(primary, jd)[0])
        assert d >= c.miss_km - 1e-6


def test_relative_speed_is_physical(demo):
    for c in demo[2]:
        assert 0 < c.rel_speed < 16.0                   # km/s, LEO upper bound


# ----------------------------------------------------------------------------- probability
def test_pc_matches_isotropic_closed_form():
    s, R = 0.5, 0.02                                     # km
    enc = Encounter(miss_2d=np.zeros(2), cov_2d=np.eye(2) * s**2, basis=np.eye(2, 3))
    expected = 1 - math.exp(-R**2 / (2 * s**2))
    assert pc_foster(enc, R) == pytest.approx(expected, rel=1e-4)


def test_pc_decreases_with_miss_distance():
    cov = np.diag([0.3, 0.8]) ** 2
    pcs = [pc_foster(Encounter(np.array([m, 0.0]), cov, np.eye(2, 3))) for m in (0, 0.5, 1, 2, 4)]
    assert all(a > b for a, b in zip(pcs, pcs[1:]))


# ----------------------------------------------------------------------------- maneuver
def test_zero_dv_reproduces_original_encounter(demo):
    primary, _, conjs = demo
    r = AvoidancePlanner(primary, conjs[0]).evaluate(0.0)
    assert r.new_miss_km == pytest.approx(conjs[0].miss_km, abs=1e-3)


def test_minimum_dv_meets_target(demo):
    primary, _, conjs = demo
    best, _ = AvoidancePlanner(primary, conjs[0]).minimum_dv(pc_target=1e-4)
    assert best is not None and best.new_pc < 1e-4 and abs(best.dv_ms) <= 0.5


def test_no_maneuver_when_already_safe(demo):
    primary, _, conjs = demo
    safe = min(conjs, key=lambda c: c.pc)
    best, tried = AvoidancePlanner(primary, safe).minimum_dv(pc_target=max(safe.pc * 10, 1e-12))
    assert best is None and tried == []


def test_cw_prograde_burn_drifts_behind(demo):
    """Raising the orbit slows the satellite: a prograde burn gives a negative along-track shift."""
    primary, _, conjs = demo
    r = AvoidancePlanner(primary, conjs[0]).evaluate(+0.05)
    assert r.cw_along_shift_km < 0 and r.cw_radial_shift_km > 0
