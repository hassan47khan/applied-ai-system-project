"""The Streamlit app (smoke tests with AppTest, offline) and the Deezer matcher."""

import pytest

from src.enrich import pick_match, search_title

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pathlib import Path

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def fresh_app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def html_text(at) -> str:
    return " ".join(str(getattr(el, "body", "")) for el in at.get("html"))


def test_first_run_shows_the_title_examples_and_the_offline_status():
    at = fresh_app()
    text = html_text(at)
    assert 'class="vf-title"' in text and ">Discover<" in text
    assert "Offline (no Gemini key)" in text
    assert any("Something calm for studying" in b.label for b in at.button)


def test_a_request_renders_the_lead_pick_confidence_and_trace():
    at = fresh_app()
    at.text_input[0].input("I need something calm for studying late at night")
    next(b for b in at.button if b.label == "Find tracks").click()
    at.run()
    assert not at.exception, at.exception
    text = html_text(at)
    assert "Top pick for you" in text and "Confidence" in text and "Where the picks land" in text
    assert "late-night-study" not in text or "Late-night studying" in text   # card titles, not ids
    assert "Late-night studying" in text


def test_a_follow_up_refines_instead_of_starting_over():
    at = fresh_app()
    at.text_input[0].input("I need something calm for studying late at night")
    next(b for b in at.button if b.label == "Find tracks").click()
    at.run()
    at.text_input[0].input("more upbeat")
    next(b for b in at.button if b.label == "Find tracks").click()
    at.run()
    assert not at.exception, at.exception
    assert "more upbeat -&gt; more energy" in html_text(at) or "more upbeat -> more energy" in html_text(at)


def test_an_unusable_request_explains_itself_without_crashing():
    at = fresh_app()
    at.text_input[0].input("asdfghjkl qwerty")
    next(b for b in at.button if b.label == "Find tracks").click()
    at.run()
    assert not at.exception, at.exception
    assert "No tracks" in html_text(at)


def test_example_buttons_run_a_request():
    at = fresh_app()
    next(b for b in at.button if b.label.startswith("Romantic dinner")).click()
    at.run()
    assert not at.exception and "Top pick for you" in html_text(at)


# ---- Deezer matching -----------------------------------------------------------------
def _hit(title, artist):
    return {"title": title, "artist": {"name": artist}, "album": {"cover_medium": "c"}, "preview": "p"}


def test_match_prefers_the_right_artist():
    results = [_hit("Bad Habits", "Someone Else"), _hit("Bad Habits", "Ed Sheeran")]
    assert pick_match(results, "Bad Habits", "Ed Sheeran")["artist"]["name"] == "Ed Sheeran"


def test_match_folds_accents():
    assert pick_match([_hit("Tacones Rojos", "Sebastián Yatra")], "Tacones Rojos", "Sebastian Yatra")


def test_a_weak_match_is_rejected_rather_than_showing_the_wrong_cover():
    assert pick_match([_hit("Totally Different", "Nobody")], "Bad Habits", "Ed Sheeran") is None


@pytest.mark.parametrize("title, expected", [
    ("Song - Remastered 2011", "Song"), ("Song (feat. X)", "Song"), ("Song [Live]", "Song"), ("Plain", "Plain"),
])
def test_search_title_drops_version_noise(title, expected):
    assert search_title(title) == expected


def test_media_lookups_are_off_in_the_test_suite():
    from src.enrich import lookup
    assert lookup("Bad Habits", "Ed Sheeran") is None
