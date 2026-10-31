"""Collision probability (2D Foster method) in the encounter (B-) plane.

TLEs carry no covariance, so position uncertainties are *assumed* per object in the
radial / along-track / cross-track (RTN) frame. Results are therefore illustrative,
not operational — which the README states explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate


@dataclass
class Sigma:
    """1-sigma position uncertainty in RTN, km."""
    radial: float
    along: float
    cross: float


# Typical orders of magnitude for TLE-derived states over a few days.
PRIMARY_SIGMA = Sigma(radial=0.10, along=0.60, cross=0.10)
SECONDARY_SIGMA = Sigma(radial=0.30, along=1.50, cross=0.30)
HARD_BODY_RADIUS_KM = 0.020   # combined object radius (20 m)


def rtn_matrix(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows are R, T, N unit vectors expressed in the inertial frame."""
    R = r / np.linalg.norm(r)
    N = np.cross(r, v)
    N /= np.linalg.norm(N)
    T = np.cross(N, R)
    return np.vstack([R, T, N])


def inertial_cov(r: np.ndarray, v: np.ndarray, s: Sigma) -> np.ndarray:
    Q = rtn_matrix(r, v)
    return Q.T @ np.diag([s.radial**2, s.along**2, s.cross**2]) @ Q


@dataclass
class Encounter:
    miss_2d: np.ndarray      # miss vector in encounter plane (km)
    cov_2d: np.ndarray       # combined 2x2 covariance (km^2)
    basis: np.ndarray        # rows: x_hat, y_hat of the encounter plane (inertial)


def encounter_plane(r1, v1, r2, v2, s1: Sigma = PRIMARY_SIGMA, s2: Sigma = SECONDARY_SIGMA) -> Encounter:
    dr = r2 - r1
    dv = v2 - v1
    z = dv / np.linalg.norm(dv)
    x = dr - dr.dot(z) * z
    if np.linalg.norm(x) < 1e-12:
        x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    P = np.vstack([x, y])
    C = inertial_cov(r1, v1, s1) + inertial_cov(r2, v2, s2)
    return Encounter(miss_2d=P @ dr, cov_2d=P @ C @ P.T, basis=P)


def pc_foster(enc: Encounter, hbr: float = HARD_BODY_RADIUS_KM) -> float:
    """Integrate the 2D Gaussian over the hard-body circle centred on the origin."""
    Ci = np.linalg.inv(enc.cov_2d)
    norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(enc.cov_2d)))
    mx, my = enc.miss_2d

    def f(y, x):
        d = np.array([x - mx, y - my])
        return norm * np.exp(-0.5 * d @ Ci @ d)

    val, _ = integrate.dblquad(f, -hbr, hbr, lambda x: -np.sqrt(max(hbr**2 - x**2, 0.0)),
                               lambda x: np.sqrt(max(hbr**2 - x**2, 0.0)), epsabs=1e-14, epsrel=1e-8)
    return float(max(val, 0.0))


def pc_for(r1, v1, r2, v2, hbr: float = HARD_BODY_RADIUS_KM) -> float:
    return pc_foster(encounter_plane(r1, v1, r2, v2), hbr)
