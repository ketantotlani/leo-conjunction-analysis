"""Streamlit GUI smoke tests (offline, synthetic demo)."""
from streamlit.testing.v1 import AppTest


def _run_demo():
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("Synthetic demo")
    at.run()
    at.sidebar.button[0].click()
    at.run()
    return at


def test_demo_runs_without_errors():
    at = _run_demo()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Objects loaded"] == "406"
    assert int(labels["Conjunctions < 5 km"]) > 0
    assert at.success                                  # a maneuver was planned


def test_empty_debris_selection_is_rejected():
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    at.sidebar.multiselect[0].set_value([])
    at.run()
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception
    assert any("at least one debris population" in w.value for w in at.warning)
