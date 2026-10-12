"""Smoke tests: the app boots, the login gate works, every tab renders."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=240)


def test_welcome_page_is_shown_first():
    at = _app().run()
    assert not at.exception
    assert len(at.tabs) == 0                      # dashboard hidden
    assert "Sign in" in [b.label for b in at.button]


def test_wrong_password_is_rejected():
    at = _app().run()
    at.text_input[0].set_value("admin")
    at.text_input[1].set_value("wrong")
    next(b for b in at.button if b.label == "Sign in").click()
    at.run()
    assert at.error and len(at.tabs) == 0


def test_correct_login_opens_dashboard():
    at = _app().run()
    at.text_input[0].set_value("admin")
    at.text_input[1].set_value("admin123")
    next(b for b in at.button if b.label == "Sign in").click()
    at.run()
    assert not at.exception
    assert len(at.tabs) == 6


def test_dashboard_renders_without_errors_for_guest():
    at = _app()
    at.session_state["user"] = "Guest"
    at.run()
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    assert "Total Revenue" in labels and "Test ROC-AUC" in labels