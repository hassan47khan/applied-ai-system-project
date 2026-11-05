from src.recommender import (
    genre_affinity, load_songs, max_score, numeric_distance, numeric_similarity,
    rank_songs, recommend_songs, score_song,
)


def song(**overrides):
    base = dict(id=1, title="T", artist="A", genre="pop", mood="happy", energy=0.8,
                tempo_bpm=120, valence=0.8, danceability=0.8, acousticness=0.2)
    base.update(overrides)
    return base


def test_catalog_loads():
    songs = load_songs()
    assert len(songs) == 18
    assert all(0.0 <= s["energy"] <= 1.0 for s in songs)


def test_genre_affinity_exact_related_and_none():
    assert genre_affinity("pop", "pop") == 1.0
    assert genre_affinity("pop", "indie pop") == 0.5
    assert genre_affinity("pop", "jazz") == 0.0


def test_range_target_is_zero_distance_inside_and_edge_distance_outside():
    assert numeric_distance(0.3, (0.2, 0.4)) == 0.0
    assert abs(numeric_distance(0.5, (0.2, 0.4)) - 0.1) < 1e-9
    assert abs(numeric_distance(0.1, (0.2, 0.4)) - 0.1) < 1e-9


def test_mismatch_is_penalised_not_just_zero():
    assert numeric_similarity(0.5, 0.5) == 1.0
    assert numeric_similarity(0.0, 0.5) == 0.0
    assert numeric_similarity(0.0, 1.0) == -1.0   # worst case subtracts points


def test_far_numeric_miss_lowers_score_below_zero_contribution():
    near, _ = score_song({"energy": 0.9}, song(energy=0.9))
    far, reasons = score_song({"energy": 0.9}, song(energy=0.1))
    assert near > 0 > far
    assert any("misses" in r for r in reasons)


def test_explicit_genre_outweighs_context_genre():
    explicit, _ = score_song({"genres": ["pop"]}, song())
    context, _ = score_song({"context_genres": ["pop"]}, song())
    assert explicit > context > 0


def test_avoided_genre_is_penalised():
    score, reasons = score_song({"avoid_genres": ["jazz"]}, song(genre="jazz"))
    assert score < 0 and any("avoided" in r for r in reasons)
    assert score_song({"avoid_genres": ["jazz"]}, song(genre="pop"))[0] == 0


def test_genre_scale_relaxes_genre_weight():
    full, _ = score_song({"genres": ["pop"], "genre_scale": 1.0}, song())
    half, _ = score_song({"genres": ["pop"], "genre_scale": 0.5}, song())
    assert abs(half - full / 2) < 1e-9


def test_max_score_is_the_perfect_song_score():
    prefs = {"genres": ["pop"], "moods": ["happy"], "energy": 0.8, "danceability": 0.8}
    perfect, _ = score_song(prefs, song())
    assert abs(perfect - max_score(prefs)) < 1e-9


def test_ranking_is_deterministic_and_sorted():
    songs = load_songs()
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    a = [s["title"] for s, _, _ in rank_songs(prefs, songs)]
    b = [s["title"] for s, _, _ in rank_songs(prefs, songs)]
    assert a == b
    scores = [sc for _, sc, _ in rank_songs(prefs, songs)]
    assert scores == sorted(scores, reverse=True)


def test_module3_recommend_songs_shape_still_works():
    out = recommend_songs({"genre": "pop", "mood": "happy", "energy": 0.8}, load_songs(), k=3)
    assert len(out) == 3 and out[0][0]["title"] == "Sunrise City"
    assert isinstance(out[0][2], str) and out[0][2]
