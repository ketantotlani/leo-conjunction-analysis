"""Figures for the conjunction analysis (matplotlib, saved as PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse

from .maneuver import AvoidancePlanner, ManeuverResult
from .objects import MU, R_EARTH, SpaceObject
from .probability import encounter_plane
from .screening import Conjunction, range_history, state

PRIMARY_C = "#1f4e79"
ACCENT = "#c0392b"
GREY = "#8a8f98"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
})


def _save(fig, out: Path, name: str) -> Path:
    p = out / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _track(obj: SpaceObject, jd0: float, minutes: float, n: int = 400) -> np.ndarray:
    return np.array([state(obj, jd0 + m / 1440.0)[0] for m in np.linspace(0, minutes, n)])


def orbits_3d(primary: SpaceObject, conjs: list[Conjunction], out: Path, top: int = 3) -> Path:
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(R_EARTH * np.cos(u) * np.sin(v), R_EARTH * np.sin(u) * np.sin(v), R_EARTH * np.cos(v),
                    color="#d6e4f0", alpha=0.35, linewidth=0, zorder=0)
    period_min = 2 * np.pi / primary.sat.no_kozai
    seen, colors = set(), ["#e67e22", "#8e44ad", "#16a085", "#7f8c8d"]
    shown = [c for c in conjs if not (c.secondary.norad_id in seen or seen.add(c.secondary.norad_id))][:top]
    for i, c in enumerate(shown):
        jd0 = c.tca_jd - period_min / 2 / 1440
        tr = _track(c.secondary, jd0, period_min)
        ax.plot(*tr.T, color=colors[i % 4], lw=1.0, label=f"{c.secondary.name} (miss {c.miss_km*1000:.0f} m)")
        ax.scatter(*c.r1, color=ACCENT, s=40, marker="x", depthshade=False)
    tp = _track(primary, shown[0].tca_jd - period_min / 2 / 1440 if shown else 0, period_min)
    ax.plot(*tp.T, color=PRIMARY_C, lw=2.0, label=primary.name)
    lim = R_EARTH + 1000
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), xlabel="x (km)", ylabel="y (km)", zlabel="z (km)")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Primary orbit and highest-risk secondaries (× = TCA)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    return _save(fig, out, "fig1_orbits_3d.png")


def gabbard(primary: SpaceObject, secondaries: list[SpaceObject], conj_ids: set[int], out: Path) -> Path:
    def pa(o):
        n = o.sat.no_kozai / 60.0
        a = (MU / n**2) ** (1 / 3)
        return 2 * np.pi / n / 60.0, a * (1 + o.sat.ecco) - R_EARTH, a * (1 - o.sat.ecco) - R_EARTH

    fig, ax = plt.subplots(figsize=(7, 4.5))
    P, A, Pe = (np.array([pa(o) for o in secondaries]).T if secondaries else (np.empty(0),) * 3)
    ax.scatter(P, A, s=6, color="#e59866", label="apogee")
    ax.scatter(P, Pe, s=6, color="#5dade2", label="perigee")
    hit = np.array([o.norad_id in conj_ids for o in secondaries])
    if hit.any():
        ax.scatter(P[hit], A[hit], s=40, facecolors="none", edgecolors=ACCENT, lw=1.2, label="in conjunction")
        ax.scatter(P[hit], Pe[hit], s=40, facecolors="none", edgecolors=ACCENT, lw=1.2)
    pp, pap, ppe = pa(primary)
    ax.axhspan(ppe, pap, color=PRIMARY_C, alpha=0.15, label=f"{primary.name} altitude band")
    ax.set(xlabel="Orbital period (min)", ylabel="Altitude (km)",
           title="Gabbard diagram of the screened population")
    ax.legend(fontsize=8, frameon=False, markerscale=1.5)
    return _save(fig, out, "fig2_gabbard.png")


def timeline(conjs: list[Conjunction], threshold_km: float, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    t = [c.tca_utc for c in conjs]
    pcs = np.clip([c.pc for c in conjs], 1e-12, 1)
    sc = ax.scatter(t, [c.miss_km for c in conjs], c=pcs, cmap="inferno_r", norm=LogNorm(1e-10, 1e-2),
                    s=[20 + 12 * c.rel_speed for c in conjs], edgecolors="k", linewidths=0.4)
    ax.axhline(threshold_km, color=GREY, ls="--", lw=1)
    ax.set(xlabel="Time of closest approach (UTC)", ylabel="Miss distance (km)",
           title=f"Conjunctions below {threshold_km:g} km  (marker size ∝ relative speed)")
    fig.colorbar(sc, ax=ax, label="Collision probability $P_c$")
    fig.autofmt_xdate()
    return _save(fig, out, "fig3_conjunction_timeline.png")


def range_profile(primary: SpaceObject, c: Conjunction, out: Path) -> Path:
    ts, d = range_history(primary, c.secondary, c.tca_jd, half_window_s=60, n=481)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(ts, d, color=PRIMARY_C, lw=1.8)
    ax.axvline(0, color=ACCENT, ls=":", lw=1)
    ax.annotate(f"TCA  {c.tca_utc:%Y-%m-%d %H:%M:%S} UTC\nmiss {c.miss_km*1000:.0f} m, "
                f"v_rel {c.rel_speed:.1f} km/s", xy=(0, c.miss_km), xytext=(-57, max(d) * 0.55),
                arrowprops=dict(arrowstyle="->", color=ACCENT), fontsize=9)
    ax.set(xlabel="Time from TCA (s)", ylabel="Range (km)", title=f"Range profile: {c.secondary.name}")
    ins = ax.inset_axes([0.62, 0.40, 0.33, 0.45])
    hw = max(3 * c.miss_km / max(c.rel_speed, 1e-3), 0.002)        # seconds
    ts2, d2 = range_history(primary, c.secondary, c.tca_jd, half_window_s=hw, n=201)
    ins.plot(ts2 * 1000, d2 * 1000, color=PRIMARY_C, lw=1.5)
    ins.set_title("zoom near TCA", fontsize=8)
    ins.set_xlabel("ms", fontsize=7)
    ins.set_ylabel("m", fontsize=7)
    ins.tick_params(labelsize=7)
    return _save(fig, out, "fig4_range_profile.png")


def _ellipse(ax, mean, cov, k, **kw):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mean, 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0]), angle=ang, fill=False, **kw))


def encounter_plane_plot(c: Conjunction, planner: AvoidancePlanner, best: ManeuverResult | None, out: Path) -> Path:
    enc = encounter_plane(c.r1, c.v1, c.r2, c.v2)
    fig, ax = plt.subplots(figsize=(6, 6))
    for k, a in zip((1, 2, 3), (0.9, 0.6, 0.35)):
        _ellipse(ax, (0, 0), enc.cov_2d, k, color=PRIMARY_C, lw=1.2, alpha=a, label="1/2/3σ combined covariance" if k == 1 else None)
    ax.scatter(*enc.miss_2d, color=ACCENT, zorder=5, label=f"before: miss {c.miss_km*1000:.0f} m, Pc {c.pc:.1e}")
    ax.add_patch(Circle(enc.miss_2d, 0.02, color=ACCENT, alpha=0.3))
    if best is not None:
        jd = best.new_tca_jd
        dpos, dvel = planner._delta(best.dv_ms / 1000.0)
        s = (jd - c.tca_jd) * 86400
        r1, v1 = state(planner.primary, jd)
        r2, _ = state(c.secondary, jd)
        r1 = r1 + dpos(planner.lead + s)
        m_after = enc.basis @ (r2 - r1)
        ax.scatter(*m_after, color="#27ae60", marker="D", zorder=5,
                   label=f"after Δv={best.dv_ms*100:+.0f} cm/s: miss {best.new_miss_km*1000:.0f} m, Pc {best.new_pc:.1e}")
        ax.annotate("", xy=m_after, xytext=enc.miss_2d, arrowprops=dict(arrowstyle="->", color="#27ae60"))
    lim = 3.3 * np.sqrt(np.max(np.linalg.eigvalsh(enc.cov_2d)))
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="Encounter-plane x (km)",
           ylabel="Encounter-plane y (km)", title="Encounter (B-) plane: secondary position relative to primary")
    ax.legend(fontsize=7.5, loc="lower left", frameon=True)
    return _save(fig, out, "fig5_encounter_plane.png")


def maneuver_tradeoff(results: list[ManeuverResult], best: ManeuverResult | None, pc_target: float, out: Path,
                      lead_orbits: float = 1.5) -> Path:
    res = sorted(results, key=lambda r: r.dv_ms)
    dv = np.array([r.dv_ms for r in res]) * 100
    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    ax1.semilogy(dv, np.clip([r.new_pc for r in res], 1e-12, 1), color=ACCENT, lw=1.8, label="$P_c$")
    ax1.axhline(pc_target, color=GREY, ls="--", lw=1, label=f"threshold {pc_target:g}")
    ax1.set(xlabel="Along-track Δv (cm/s)", ylabel="Collision probability $P_c$")
    ax2 = ax1.twinx()
    ax2.plot(dv, [r.new_miss_km for r in res], color=PRIMARY_C, lw=1.5, label="miss distance")
    ax2.set_ylabel("Miss distance (km)", color=PRIMARY_C)
    ax2.grid(False)
    if best is not None:
        ax1.axvline(best.dv_ms * 100, color="#27ae60", lw=1.2)
        ax1.annotate(f"min |Δv| = {abs(best.dv_ms)*100:.0f} cm/s", xy=(best.dv_ms * 100, pc_target),
                     xytext=(10, 20), textcoords="offset points", color="#27ae60", fontsize=9)
    lines = ax1.get_legend_handles_labels()
    l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + l2[0], lines[1] + l2[1], fontsize=8, frameon=True, loc="lower left")
    ax1.set_title(f"Avoidance maneuver trade-off (burn {lead_orbits:g} orbits before TCA)")
    return _save(fig, out, "fig6_maneuver_tradeoff.png")
