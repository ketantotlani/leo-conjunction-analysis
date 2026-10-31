#!/usr/bin/env python3
"""LEO conjunction screening, collision probability, and avoidance-maneuver planning.

Examples
--------
Real data (needs internet; downloads TLEs from CelesTrak and caches them in ./data):
    python run_analysis.py --primary 25544 --groups cosmos-2251-debris iridium-33-debris fengyun-1c-debris

Offline demo with a synthetic debris cloud:
    python run_analysis.py --demo
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sgp4.api import jday

from leoca import visualize as viz
from leoca.maneuver import AvoidancePlanner
from leoca.objects import build_demo, load_real
from leoca.probability import HARD_BODY_RADIUS_KM, PRIMARY_SIGMA, SECONDARY_SIGMA, pc_for
from leoca.screening import prefilter, screen

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use a synthetic debris cloud (no network)")
    ap.add_argument("--primary", type=int, default=25544, help="NORAD ID of the protected satellite (default ISS)")
    ap.add_argument("--groups", nargs="+",
                    default=["cosmos-2251-debris", "iridium-33-debris", "fengyun-1c-debris"],
                    help="CelesTrak GROUP names for secondary objects")
    ap.add_argument("--start", default=None, help="screening start, UTC 'YYYY-MM-DD' (default: now / demo epoch)")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--threshold-km", type=float, default=5.0)
    ap.add_argument("--pc-target", type=float, default=1e-4, help="maneuver until Pc falls below this")
    ap.add_argument("--lead-orbits", type=float, default=1.5, help="burn time before TCA, in primary orbits")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        primary, secondaries = build_demo()
        start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    else:
        primary, secondaries = load_real(args.primary, args.groups, ROOT / "data")
        start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    jd, fr = jday(start.year, start.month, start.day, start.hour, start.minute, start.second)

    print(f"Primary: {primary.name} (NORAD {primary.norad_id})")
    cand = prefilter(primary, secondaries)
    print(f"Secondaries: {len(secondaries)} loaded, {len(cand)} overlap the primary's altitude band")

    conjs = screen(primary, cand, jd + fr, args.days, args.threshold_km)
    for c in conjs:
        c.pc = pc_for(c.r1, c.v1, c.r2, c.v2)
    conjs.sort(key=lambda c: -c.pc)
    print(f"Conjunctions < {args.threshold_km:g} km over {args.days:g} days: {len(conjs)}")

    with open(out / "conjunctions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["secondary", "norad_id", "tca_utc", "miss_m", "rel_speed_kms", "pc"])
        for c in conjs:
            w.writerow([c.secondary.name, c.secondary.norad_id, c.tca_utc.isoformat(timespec="seconds"),
                        f"{c.miss_km*1000:.1f}", f"{c.rel_speed:.3f}", f"{c.pc:.3e}"])

    figs = [viz.gabbard(primary, cand, {c.secondary.norad_id for c in conjs}, out)]
    best = None
    if not conjs:
        print("No conjunctions found — try a longer window, a larger threshold, or other debris groups.")
    else:
        worst = conjs[0]
        figs += [viz.orbits_3d(primary, conjs, out), viz.timeline(conjs, args.threshold_km, out),
                 viz.range_profile(primary, worst, out)]
        planner = AvoidancePlanner(primary, worst, lead_orbits=args.lead_orbits)
        best, _ = planner.minimum_dv(pc_target=args.pc_target)
        sweep = planner.sweep(np.linspace(-0.4, 0.4, 81))
        figs += [viz.encounter_plane_plot(worst, planner, best, out),
                 viz.maneuver_tradeoff(sweep, best, args.pc_target, out, args.lead_orbits)]
        _report(out, args, primary, cand, conjs, worst, best, start)

    print("\nTop conjunctions by Pc:")
    for c in conjs[:10]:
        print(f"  {c.tca_utc:%Y-%m-%d %H:%M:%S}  {c.secondary.name:28s} miss {c.miss_km*1000:7.0f} m  "
              f"v_rel {c.rel_speed:5.2f} km/s  Pc {c.pc:.2e}")
    if best:
        print(f"\nMinimum along-track Δv for Pc < {args.pc_target:g}: {best.dv_ms*100:+.1f} cm/s "
              f"-> miss {best.new_miss_km*1000:.0f} m, Pc {best.new_pc:.2e}")
    print("\nFigures:", *[f"  {p.name}" for p in figs], sep="\n")


def _report(out, args, primary, cand, conjs, worst, best, start) -> None:
    lines = [
        "# Conjunction Assessment Summary", "",
        f"- Primary: **{primary.name}** (NORAD {primary.norad_id})",
        f"- Window: {start:%Y-%m-%d %H:%M} UTC + {args.days:g} days; screening threshold {args.threshold_km:g} km",
        f"- Secondaries screened: {len(cand)}; conjunctions found: {len(conjs)}",
        f"- Assumed 1σ (R/T/N, m): primary {PRIMARY_SIGMA.radial*1e3:.0f}/{PRIMARY_SIGMA.along*1e3:.0f}/"
        f"{PRIMARY_SIGMA.cross*1e3:.0f}, secondary {SECONDARY_SIGMA.radial*1e3:.0f}/{SECONDARY_SIGMA.along*1e3:.0f}/"
        f"{SECONDARY_SIGMA.cross*1e3:.0f}; hard-body radius {HARD_BODY_RADIUS_KM*1e3:.0f} m",
        "", "## Highest-risk event", "",
        f"- {worst.secondary.name} at {worst.tca_utc:%Y-%m-%d %H:%M:%S} UTC",
        f"- Miss distance {worst.miss_km*1000:.0f} m, relative speed {worst.rel_speed:.2f} km/s, Pc {worst.pc:.2e}",
    ]
    if best:
        lines += [f"- Avoidance: along-track Δv {best.dv_ms*100:+.1f} cm/s applied {args.lead_orbits:g} orbits "
                  f"before TCA -> miss {best.new_miss_km*1000:.0f} m, Pc {best.new_pc:.2e}",
                  f"- Clohessy-Wiltshire check: along-track shift {best.cw_along_shift_km*1000:.0f} m, "
                  f"radial shift {best.cw_radial_shift_km*1000:.0f} m at TCA"]
    lines += ["", "## Top 10 events by Pc", "", "| TCA (UTC) | Secondary | Miss (m) | v_rel (km/s) | Pc |",
              "|---|---|---:|---:|---:|"]
    lines += [f"| {c.tca_utc:%Y-%m-%d %H:%M:%S} | {c.secondary.name} | {c.miss_km*1000:.0f} | "
              f"{c.rel_speed:.2f} | {c.pc:.2e} |" for c in conjs[:10]]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
