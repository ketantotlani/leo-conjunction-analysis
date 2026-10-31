"""Streamlit GUI for LEO conjunction screening and avoidance planning.

Run:  streamlit run app.py
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sgp4.api import jday

from leoca import visualize as viz
from leoca.maneuver import AvoidancePlanner
from leoca.objects import build_demo, load_real
from leoca.probability import pc_for
from leoca.screening import prefilter, screen

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "outputs" / "gui"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEBRIS_GROUPS = {
    "Cosmos 2251 debris": "cosmos-2251-debris",
    "Iridium 33 debris": "iridium-33-debris",
    "Fengyun-1C debris": "fengyun-1c-debris",
    "Cosmos 1408 debris (2021 ASAT test)": "cosmos-1408-debris",
    "Starlink": "starlink",
    "Last 30 days' launches": "last-30-days",
}
# Debris from the big breakups sits mostly at ~700-900 km, so satellites in that band see far
# more conjunctions than the ISS (~420 km).
PRIMARIES = {"NOAA 19 (~850 km, Fengyun-1C band)": 33591, "Sentinel-2A (~786 km)": 40697,
             "Terra (~705 km)": 25994, "ISS (ZARYA) (~420 km)": 25544, "Hubble (~520 km)": 20580,
             "Custom NORAD ID": None}
FALLBACK_KM = 50.0

st.set_page_config(page_title="LEO Conjunction Analysis", layout="wide")
st.title("LEO Conjunction Screening & Collision Avoidance")
st.caption("SGP4 propagation · conjunction screening · Foster 2D collision probability · "
           "minimum-Δv along-track avoidance maneuver (J2 + Clohessy-Wiltshire check)")

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Scenario")
    mode = st.radio("Data source", ["Real (CelesTrak)", "Synthetic demo"], index=0)
    if mode.startswith("Real"):
        pname = st.selectbox("Protected satellite", list(PRIMARIES))
        primary_id = PRIMARIES[pname] or st.number_input("NORAD ID", min_value=1, value=25544, step=1)
        groups = st.multiselect("Debris populations", list(DEBRIS_GROUPS),
                                default=["Cosmos 2251 debris", "Iridium 33 debris", "Fengyun-1C debris"])
        start_d = st.date_input("Start (UTC)", value=date.today())
    else:
        st.info("Synthetic ISS-like primary with a 406-object fragment cloud. Works offline.")
        start_d = date(2026, 11, 1)

    st.header("Screening")
    days = st.slider("Window (days)", 1.0, 14.0, 7.0, 0.5)
    threshold = st.slider("Miss-distance threshold (km)", 1.0, 25.0, 5.0, 0.5)

    st.header("Maneuver")
    pc_target = st.select_slider("Pc target", options=[1e-3, 1e-4, 1e-5, 1e-6], value=1e-4,
                                 format_func=lambda x: f"{x:.0e}")
    lead = st.slider("Burn lead time (orbits before TCA)", 0.5, 5.0, 1.5, 0.5)
    run = st.button("Run analysis", type="primary", width="stretch")


@st.cache_resource(show_spinner=False)
def _run(mode: str, primary_id: int, groups: tuple, start_iso: str, days: float, threshold: float):
    if mode.startswith("Real"):
        primary, secs = load_real(int(primary_id), [DEBRIS_GROUPS[g] for g in groups], ROOT / "data")
    else:
        primary, secs = build_demo()
    start = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
    jd, fr = jday(start.year, start.month, start.day, 0, 0, 0)
    cand = prefilter(primary, secs)
    conjs = screen(primary, cand, jd + fr, days, threshold)
    fallback = False
    if not conjs and cand:
        # Nothing inside the threshold: still show the closest approaches so every tab works.
        conjs = screen(primary, cand, jd + fr, days, FALLBACK_KM)[:25]
        fallback = True
    for c in conjs:
        c.pc = pc_for(c.r1, c.v1, c.r2, c.v2)
    conjs.sort(key=lambda c: -c.pc)
    return primary, secs, cand, conjs, fallback


if run and mode.startswith("Real") and not groups:
    st.warning("Select at least one debris population in the sidebar.")
    st.stop()

if run:
    st.session_state["params"] = (mode, int(primary_id) if mode.startswith("Real") else 0,
                                  tuple(groups) if mode.startswith("Real") else (),
                                  start_d.isoformat(), days, threshold)

if "params" not in st.session_state:
    st.markdown("Choose a scenario in the sidebar and press **Run analysis**.")
    st.stop()

try:
    with st.spinner("Downloading TLEs and screening… (first real run can take a minute)"):
        primary, secs, cand, conjs, fallback = _run(*st.session_state["params"])
except LookupError as e:  # bad NORAD ID / group
    st.error(f"{e}.")
    st.stop()
except Exception as e:  # network / CelesTrak errors
    st.error(f"Could not load data: {e}\n\nCheck your internet connection, or switch to the synthetic demo.")
    st.stop()

# ----------------------------------------------------------------------------- summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Primary", primary.name)
c2.metric("Objects loaded", f"{len(secs):,}")
c3.metric("In altitude band", f"{len(cand):,}")
thr = FALLBACK_KM if fallback else st.session_state['params'][5]
c4.metric(f"Conjunctions < {st.session_state['params'][5]:g} km", 0 if fallback else len(conjs))
if not cand:
    st.warning("None of the selected debris shares the primary's altitude band, so no close approaches are "
               "possible. Pick a primary at 700-900 km (e.g. NOAA 19) or different debris groups.")
elif fallback:
    st.info(f"No approaches inside {st.session_state['params'][5]:g} km — that is normal for a few days of real "
            f"data. Showing the {len(conjs)} closest approaches within {FALLBACK_KM:g} km instead.")

tab_over, tab_events, tab_man = st.tabs(["Overview", "Conjunctions", "Avoidance maneuver"])

with tab_over:
    a, b = st.columns(2)
    a.image(str(viz.gabbard(primary, cand, {c.secondary.norad_id for c in conjs}, FIG_DIR)),
            width="stretch")
    if conjs:
        b.image(str(viz.orbits_3d(primary, conjs, FIG_DIR)), width="stretch")
        st.image(str(viz.timeline(conjs, thr, FIG_DIR)), width="stretch")
    else:
        b.warning("No conjunctions in this window. Try more days, a larger threshold, or other debris groups.")

if not conjs:
    st.stop()

df = pd.DataFrame([{
    "TCA (UTC)": c.tca_utc.strftime("%Y-%m-%d %H:%M:%S"), "Secondary": c.secondary.name,
    "NORAD": c.secondary.norad_id, "Miss (m)": round(c.miss_km * 1000), "v_rel (km/s)": round(c.rel_speed, 2),
    "Pc": c.pc} for c in conjs])

with tab_events:
    st.dataframe(df.style.format({"Pc": "{:.2e}"}).background_gradient(subset=["Pc"], cmap="Reds"),
                 width="stretch", height=420)
    st.download_button("Download CSV", df.to_csv(index=False), "conjunctions.csv", "text/csv")

with tab_man:
    labels = [f"{r['TCA (UTC)']} · {r['Secondary']} · miss {r['Miss (m)']} m · Pc {r['Pc']:.1e}"
              for _, r in df.iterrows()]
    idx = st.selectbox("Event to analyse", range(len(labels)), format_func=lambda i: labels[i])
    conj = conjs[idx]
    with st.spinner("Planning maneuver…"):
        planner = AvoidancePlanner(primary, conj, lead_orbits=lead)
        best, _ = planner.minimum_dv(pc_target=pc_target)
        sweep = planner.sweep(np.linspace(-0.4, 0.4, 61))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Miss distance", f"{conj.miss_km*1000:.0f} m")
    m2.metric("Relative speed", f"{conj.rel_speed:.2f} km/s")
    m3.metric("Pc before", f"{conj.pc:.1e}")
    if conj.pc < pc_target:
        m4.metric("Min. Δv", "none needed")
        st.info(f"Pc {conj.pc:.1e} is already below the {pc_target:.0e} target — no maneuver required. "
                "The trade-off chart below still shows how a burn would change the risk.")
    elif best:
        m4.metric("Min. Δv (along-track)", f"{best.dv_ms*100:+.1f} cm/s",
                  f"Pc after {best.new_pc:.1e}", delta_color="off")
        st.success(f"A {abs(best.dv_ms)*100:.1f} cm/s {'prograde' if best.dv_ms > 0 else 'retrograde'} burn "
                   f"{lead:g} orbits before TCA changes the miss distance from {conj.miss_km*1000:.0f} m to "
                   f"{best.new_miss_km*1000:.0f} m and lowers Pc from {conj.pc:.1e} to {best.new_pc:.1e}. "
                   f"Clohessy-Wiltshire check: along-track shift {best.cw_along_shift_km*1000:.0f} m, "
                   f"radial {best.cw_radial_shift_km*1000:.0f} m.")
    else:
        m4.metric("Min. Δv", "> 50 cm/s")
        st.warning("No along-track burn up to 50 cm/s reaches the target. Try a longer lead time.")
    a, b = st.columns(2)
    a.image(str(viz.encounter_plane_plot(conj, planner, best, FIG_DIR)), width="stretch")
    b.image(str(viz.range_profile(primary, conj, FIG_DIR)), width="stretch")
    st.image(str(viz.maneuver_tradeoff(sweep, best, pc_target, FIG_DIR, lead)), width="stretch")

st.caption("Covariances are assumed (TLEs carry none), so Pc values are illustrative, not operational.")
