"""Regressions from the build/vibefinder-2 code review: each case once gave a wrong answer."""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.agent import VibeFinderAgent
from src.main import main
from src.profile import parse_explicit, strip_negated
from src.recommender import load_songs, score_song
from src.retrieval import Retriever, load_cards

GENRES = ["pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop",
          "ambient electronic", "classic jazz", "indie folk", "latin pop", "world fusion"]
MOODS = ["happy", "chill", "intense", "relaxed", "moody", "focused", "melancholy", "energetic"]
FIELDS = ["id", "title", "artist", "genre", "mood", "energy", "tempo_bpm", "valence",
          "danceability", "acousticness"]


@pytest.fixture(scope="module")
def agent():
    return VibeFinderAgent()


def parse(text):
    return parse_explicit(text, GENRES, MOODS)


# ---- rejecting a genre ---------------------------------------------------------
@pytest.mark.parametrize("text", [
    "anything but jazz", "I don\u2019t want jazz", "I don't want jazz", "dont want jazz",
    "don't play jazz", "don't want to hear jazz", "not in the mood for jazz",
    "I can't stand jazz", "sick of jazz", "no more jazz", "without any jazz",
    "not a fan of jazz", "something other than jazz", "stop playing jazz",
])
def test_every_way_of_rejecting_a_genre(text):
    p = parse(text)
    assert p.genres == [] and "jazz" in p.avoid_genres


@pytest.mark.parametrize("text", [
    "not jazz or rock", "I hate jazz and rock", "neither jazz nor rock",
    "no jazz, rock, or pop", "no jazz, rock or pop",
])
def test_rejection_covers_a_list(text):
    p = parse(text)
    assert p.genres == [] and {"jazz", "rock"} <= set(p.avoid_genres)


@pytest.mark.parametrize("text, genre", [
    ("nothing but jazz", "jazz"), ("why not some jazz", "jazz"), ("I don't mind jazz", "jazz"),
    ("not sure, maybe jazz", "jazz"), ("no, I want jazz", "jazz"), ("not only jazz", "jazz"),
])
def test_phrases_that_only_look_negative_keep_the_genre(text, genre):
    p = parse(text)
    assert p.genres == [genre] and p.avoid_genres == []


@pytest.mark.parametrize("text, wanted, rejected", [
    ("not jazz but rock", "rock", "jazz"),
    ("no jazz, rock please", "rock", "jazz"),
    ("I don't want jazz, I want rock", "rock", "jazz"),
    ("I hate jazz and love rock", "rock", "jazz"),
    ("no jazz, give me rock or pop", "rock", "jazz"),
])
def test_rejection_stops_at_the_end_of_its_clause(text, wanted, rejected):
    p = parse(text)
    assert wanted in p.genres and rejected in p.avoid_genres


def test_rejection_does_not_swallow_the_context():
    assert "gym" in strip_negated("no pop for the gym")
    assert "gym" not in strip_negated("not for the gym")


def test_i_cant_sleep_is_a_request_not_a_rejection(agent):
    assert agent.run("I can't sleep").retrieved[0][0] == "sleep-wind-down"


# ---- rejecting a feature ----------------------------------------------------------
@pytest.mark.parametrize("text, attr, check", [
    ("not loud music", "energy", lambda t: t[1] <= 0.6),
    ("not too loud", "energy", lambda t: t[1] <= 0.6),
    ("not calm", "energy", lambda t: t[0] >= 0.5),
    ("I don't want anything acoustic", "acousticness", lambda t: t == 0.10),
    ("avoid acoustic songs", "acousticness", lambda t: t == 0.10),
    ("non-acoustic", "acousticness", lambda t: t == 0.10),
    ("no dance music", "danceability", lambda t: t == 0.15),
    ("not danceable", "danceability", lambda t: t == 0.15),
    ("undanceable", "danceability", lambda t: t == 0.15),
])
def test_rejected_feature_never_sets_the_requested_value(text, attr, check):
    assert check(parse(text).targets[attr])


def test_what_was_asked_for_outranks_what_was_ruled_out():
    assert parse("calm, not loud").targets["energy"] == 0.25


def test_rejected_feature_alone_is_not_a_confident_answer(agent):
    r = agent.run("not loud music")
    assert r.needs_review and "exclusion_only" in r.issues
    assert r.recommendations and all(rec.song["energy"] <= 0.6 for rec in r.recommendations)


# ---- rejected genres and moods are hard filters -----------------------------------
@pytest.mark.parametrize("query, banned", [
    ("pump me up for the gym but no pop", {"pop", "indie pop", "latin pop"}),
    ("relaxing music but not jazz", {"jazz", "classic jazz"}),
    ("anything but jazz", {"jazz", "classic jazz"}),
    ("don't play rock", {"rock"}),
    ("not folk", {"indie folk"}),
])
def test_rejected_genre_never_appears_at_any_rank(agent, query, banned):
    r = agent.run(query)
    assert r.recommendations
    assert not banned & {rec.song["genre"] for rec in r.recommendations}


def test_specific_rejection_leaves_the_broader_genre():
    p = parse("pop but no latin pop")
    assert p.genres == ["pop"] and p.avoid_genres == ["latin pop"]


def test_rejecting_everything_returns_nothing_and_says_why(agent):
    r = agent.run("no pop, no lofi, no rock, no ambient, no jazz, no synthwave, no folk, no fusion")
    assert r.issues == ["no_candidates"] and r.recommendations == [] and r.needs_review
    assert r.message


@pytest.mark.parametrize("query, mood", [("not happy", "happy"), ("I'm not chill", "chill")])
def test_rejected_mood_is_filtered_and_pulls_in_no_card(agent, query, mood):
    r = agent.run(query)
    assert r.retrieved == [] and r.needs_review
    assert r.recommendations and all(rec.song["mood"] != mood for rec in r.recommendations)


def test_rejected_mood_is_not_an_explicit_preference():
    p = parse("not happy")
    assert p.moods == [] and p.avoid_moods == ["happy"] and not p.has_explicit_signal


# ---- a request that only excludes -----------------------------------------------------
@pytest.mark.parametrize("query", ["no jazz", "not rock"])
def test_exclusion_only_request_answers_and_is_flagged(agent, query):
    r = agent.run(query)
    assert r.recommendations and r.needs_review
    assert "exclusion_only" in r.issues and r.confidence < 0.6
    assert "avoid" in r.message


# ---- partial genre names ----------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("folk music", {"indie folk"}),
    ("something indie", {"indie folk", "indie pop"}),
    ("latin", {"latin pop"}),
])
def test_partial_genre_name_is_understood(text, expected):
    assert set(parse(text).genres) == expected


def test_partial_genre_name_can_be_rejected():
    assert parse("not folk").avoid_genres == ["indie folk"]


def test_ambiguous_genre_word_alone_is_not_a_genre():
    assert parse("around the world").genres == []
    assert parse("classic rock").genres == ["rock"]


def test_full_genre_name_is_not_read_twice():
    assert parse("indie pop").genres == ["indie pop"]


# ---- retrieval must not guess from one incidental word ---------------------------------
@pytest.mark.parametrize("query", [
    "I am late for my dentist appointment", "what time does the library open",
    "is the gym open today", "how do I fix my car",
])
def test_off_topic_request_is_refused(agent, query):
    r = agent.run(query)
    assert r.issues == ["no_signal"] and r.recommendations == []


@pytest.mark.parametrize("query, card", [
    ("coffee shop on sunday", "dinner-cafe-jazz"),
    ("something for a long drive home", "night-drive"),
    ("study session", "late-night-study"),
])
def test_on_topic_request_with_unknown_words_still_retrieves(query, card):
    assert Retriever(load_cards()).retrieve(query, min_score=0.12)[0][0].id == card


def test_unknown_words_lower_the_retrieval_score():
    retriever = Retriever(load_cards())
    focused = retriever.retrieve("road trip")[0][1]
    diluted = retriever.retrieve("road trip with grandma")[0][1]
    assert diluted < focused


# ---- contradictory context cards ----------------------------------------------------------
def test_opposite_cards_are_flagged_not_merged(agent):
    r = agent.run("sleep then gym")
    assert "mixed_context" in r.issues and r.needs_review
    assert "energy=0.15-0.35" in r.profile          # the stronger card, not a 0.15-1.00 union
    assert {"pop", "rock"}.isdisjoint(rec.song["genre"] for rec in r.recommendations[:3])
    assert "sleep" in r.message.lower() or "wind" in r.message.lower()


def test_compatible_cards_still_blend_without_a_flag(agent):
    r = agent.run("beach party this summer")
    assert "mixed_context" not in r.issues


def test_explicit_feature_settles_a_card_clash():
    cards = {c.id: c for c in load_cards()}
    from src.profile import _clashing_features
    clash = _clashing_features(cards["sleep-wind-down"], cards["gym-workout"], {"energy"})
    assert "energy" not in clash


# ---- odd environments never crash the CLI -------------------------------------------------
def test_empty_catalog_does_not_crash():
    r = VibeFinderAgent(songs=[]).run("calm music for studying")
    assert r.issues == ["no_candidates"] and r.recommendations == [] and "empty" in r.message


def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return path


def test_header_only_catalog_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="no songs"):
        load_songs(write_csv(tmp_path / "songs.csv", []))


@pytest.mark.parametrize("row", [
    [1, "T", "A", "pop", "happy", "loud", 100, 0.5, 0.5, 0.5],   # non-numeric field
    [1, "T", "A", "pop"],                                        # short row
    [1, "T", "A", "", "happy", 0.5, 100, 0.5, 0.5, 0.5],         # empty genre
])
def test_bad_row_names_the_file_and_line(tmp_path, row):
    with pytest.raises(ValueError, match=r"line 2"):
        load_songs(write_csv(tmp_path / "songs.csv", [row]))


def test_bad_catalog_gives_the_startup_message(tmp_path, monkeypatch, capsys):
    bad = write_csv(tmp_path / "songs.csv", [[1, "T", "A", "pop", "happy", "x", 1, 1, 1, 1]])
    monkeypatch.setattr("src.agent.load_songs", lambda: load_songs(bad))
    assert main(["calm music"]) == 2
    assert "Could not start VibeFinder" in capsys.readouterr().err


def test_character_the_console_cannot_encode_does_not_crash():
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    proc = subprocess.run([sys.executable, "-m", "src.main", "calm \U0001f600 study"],
                          capture_output=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert b"VibeFinder" in proc.stdout


# ---- second review: a rejection followed by a request after a comma ----------------
@pytest.mark.parametrize("text, wanted, rejected", [
    ("no jazz, upbeat rock or pop", {"rock", "pop"}, {"jazz"}),
    ("skip the jazz, chill lofi or ambient", {"lofi", "ambient"}, {"jazz"}),
    ("not jazz, rock or pop for the gym", set(), {"jazz", "rock", "pop"}),
])
def test_comma_continues_a_rejection_only_across_a_real_list(text, wanted, rejected):
    p = parse(text)
    assert wanted <= set(p.genres) and rejected <= set(p.avoid_genres)
    assert not wanted & set(p.avoid_genres)


def test_comma_list_needs_vocabulary_or_it_ends_the_rejection():
    # strip_negated without terms can't tell a list from a new request, so it stops early.
    assert "rock" in strip_negated("no jazz, upbeat rock or pop")


def test_rejection_then_request_gives_the_request(agent):
    r = agent.run("skip the jazz, chill lofi or ambient")
    assert r.recommendations[0].song["genre"] in {"lofi", "ambient", "ambient electronic"}
    assert parse("skip the jazz, chill lofi or ambient").moods == ["chill"]
    assert not {"jazz", "classic jazz"} & {rec.song["genre"] for rec in r.recommendations}


# ---- second review: avoiding a genre must not penalise a related one ---------------
def test_avoid_penalty_is_exact_match_only():
    pop = {"genre": "pop", "mood": "happy", "energy": 0.5, "danceability": 0.5, "acousticness": 0.5}
    score, reasons = score_song({"genres": ["pop"], "avoid_genres": ["indie pop"]}, pop)
    assert score == 1.5 and not any("avoided" in r for r in reasons)


def test_wanting_pop_but_not_indie_pop(agent):
    r = agent.run("I want pop, not indie pop")
    assert r.recommendations[0].song["genre"] == "pop"
    assert not any("avoided" in reason for rec in r.recommendations for reason in rec.reasons)


# ---- second review: "less X" is a preference, not a rejection ------------------------
@pytest.mark.parametrize("text, attr, check", [
    ("less acoustic please", "acousticness", lambda t: t == (0.00, 0.50)),
    ("a bit less electronic", "acousticness", lambda t: t == (0.50, 1.00)),
    ("less danceable", "danceability", lambda t: t == (0.00, 0.55)),
    ("less loud", "energy", lambda t: t == (0.00, 0.60)),
    ("less mellow", "energy", lambda t: t == (0.50, 1.00)),
])
def test_less_x_leans_away_without_rejecting(text, attr, check):
    p = parse(text)
    assert check(p.targets[attr]) and attr in p.explicit_numeric
    assert attr not in p.rejected_numeric and p.has_explicit_signal


def test_less_intense_is_not_a_request_for_the_intense_mood():
    p = parse("less intense")
    assert p.moods == [] and p.avoid_moods == [] and p.targets["energy"] == (0.00, 0.60)


def test_less_on_a_genre_still_rejects_it():
    assert "jazz" in parse("less jazz").avoid_genres


def test_less_acoustic_is_a_confident_answer(agent):
    r = agent.run("less acoustic please")
    assert "exclusion_only" not in r.issues
    assert all(rec.song["acousticness"] <= 0.5 for rec in r.recommendations[:3])
