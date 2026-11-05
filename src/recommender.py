"""
Scoring core, carried over from the Module 3 Music Recommender and extended.

Changes relative to the base project (see model_card.md for the reasoning):
  * Numeric targets can be a point OR a (low, high) range.
  * Numeric mismatches now cost points (similarity runs from +1 down to -1)
    instead of silently scoring zero.
  * Genre matching gives partial credit to related genres ("pop" ~ "indie pop").
  * "Soft" preferences (genres/moods inferred from retrieved context) are weighted
    below explicit ones the user typed.
  * The OOP `Recommender` class the tests expect is now wired to the same scorer.
"""

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SONGS_PATH = ROOT / "data" / "songs.csv"

# --- Scoring weights --------------------------------------------------------
GENRE_WEIGHT = 1.5           # genre the user explicitly named
MOOD_WEIGHT = 1.0            # mood the user explicitly named
CONTEXT_GENRE_WEIGHT = 0.75  # genre suggested by retrieved context cards
CONTEXT_MOOD_WEIGHT = 0.5    # mood suggested by retrieved context cards
NUMERIC_WEIGHT = 1.5         # per numeric feature (energy / danceability / acousticness)
PARTIAL_GENRE_CREDIT = 0.5   # share of the genre weight for a related genre
NUMERIC_ATTRS = ("energy", "danceability", "acousticness")
# ---------------------------------------------------------------------------

Target = Union[float, Tuple[float, float]]


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


def load_songs(csv_path: Union[str, Path] = DEFAULT_SONGS_PATH) -> List[Dict]:
    """Load the catalog into dicts with numeric fields coerced. Raises a clear error if missing."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Song catalog not found at {path}")
    songs: List[Dict] = []
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            songs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def genre_affinity(preferred: str, actual: str) -> float:
    """1.0 for an exact match, 0.5 if the genres share a word ('pop' ~ 'indie pop'), else 0."""
    preferred, actual = preferred.lower(), actual.lower()
    if preferred == actual:
        return 1.0
    if set(preferred.split()) & set(actual.split()):
        return PARTIAL_GENRE_CREDIT
    return 0.0


def numeric_distance(value: float, target: Target) -> float:
    """Distance from a point target, or from the nearest edge of a (low, high) range (0 inside)."""
    if isinstance(target, (tuple, list)):
        lo, hi = target
        if lo <= value <= hi:
            return 0.0
        return min(abs(value - lo), abs(value - hi))
    return abs(value - float(target))


def numeric_similarity(value: float, target: Target) -> float:
    """+1 on target, 0 at distance 0.5, -1 at distance 1.0 (mismatches are penalized)."""
    return 1.0 - 2.0 * numeric_distance(value, target)


def _best_affinity(preferred: Sequence[str], actual: str) -> float:
    return max((genre_affinity(p, actual) for p in preferred), default=0.0)


def score_song(prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Score one song against `prefs` and return (score, reasons).

    prefs keys (all optional):
      genre / genres, mood / moods         explicit, typed by the user
      context_genres, context_moods        soft, from retrieved context cards
      energy, danceability, acousticness   float or (low, high)
      genre_scale                          multiplier on every genre weight (used when relaxing)
    """
    score = 0.0
    reasons: List[str] = []
    genre_scale = float(prefs.get("genre_scale", 1.0))

    explicit_genres = _as_list(prefs.get("genre")) + _as_list(prefs.get("genres"))
    if explicit_genres:
        aff = _best_affinity(explicit_genres, song["genre"])
        if aff:
            pts = GENRE_WEIGHT * genre_scale * aff
            score += pts
            kind = "genre match" if aff == 1.0 else "related genre"
            reasons.append(f"{kind}: {song['genre']} (+{pts:.2f})")

    context_genres = _as_list(prefs.get("context_genres"))
    if context_genres:
        aff = _best_affinity(context_genres, song["genre"])
        if aff:
            pts = CONTEXT_GENRE_WEIGHT * genre_scale * aff
            score += pts
            reasons.append(f"genre fits the context: {song['genre']} (+{pts:.2f})")

    avoid_genres = _as_list(prefs.get("avoid_genres"))
    if avoid_genres:
        aff = _best_affinity(avoid_genres, song["genre"])
        if aff:
            pts = GENRE_WEIGHT * aff
            score -= pts
            reasons.append(f"avoided genre: {song['genre']} misses your request (-{pts:.2f})")

    explicit_moods = _as_list(prefs.get("mood")) + _as_list(prefs.get("moods"))
    if any(m.lower() == song["mood"].lower() for m in explicit_moods):
        score += MOOD_WEIGHT
        reasons.append(f"mood match: {song['mood']} (+{MOOD_WEIGHT:.2f})")

    context_moods = _as_list(prefs.get("context_moods"))
    if any(m.lower() == song["mood"].lower() for m in context_moods):
        score += CONTEXT_MOOD_WEIGHT
        reasons.append(f"mood fits the context: {song['mood']} (+{CONTEXT_MOOD_WEIGHT:.2f})")

    for attr in NUMERIC_ATTRS:
        if attr not in prefs or prefs[attr] is None:
            continue
        target = prefs[attr]
        pts = NUMERIC_WEIGHT * numeric_similarity(song[attr], target)
        score += pts
        if isinstance(target, (tuple, list)):
            tgt = f"{target[0]:.2f}-{target[1]:.2f}"
        else:
            tgt = f"{float(target):.2f}"
        sign = "+" if pts >= 0 else "-"
        verb = "fits" if pts >= 0 else "misses"
        reasons.append(f"{attr} {song[attr]:.2f} {verb} target {tgt} ({sign}{abs(pts):.2f})")

    return score, reasons


def max_score(prefs: Dict) -> float:
    """Highest score any song could earn for these prefs (used to normalise confidence)."""
    genre_scale = float(prefs.get("genre_scale", 1.0))
    total = 0.0
    if _as_list(prefs.get("genre")) + _as_list(prefs.get("genres")):
        total += GENRE_WEIGHT * genre_scale
    if _as_list(prefs.get("context_genres")):
        total += CONTEXT_GENRE_WEIGHT * genre_scale
    if _as_list(prefs.get("mood")) + _as_list(prefs.get("moods")):
        total += MOOD_WEIGHT
    if _as_list(prefs.get("context_moods")):
        total += CONTEXT_MOOD_WEIGHT
    total += NUMERIC_WEIGHT * sum(1 for a in NUMERIC_ATTRS if prefs.get(a) is not None)
    return total


def rank_songs(prefs: Dict, songs: List[Dict]) -> List[Tuple[Dict, float, List[str]]]:
    """Score every song; highest first. Ties broken by title so output is deterministic."""
    scored = [(song, *score_song(prefs, song)) for song in songs]
    scored.sort(key=lambda item: (-item[1], item[0]["title"]))
    return scored


def recommend_songs(prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Top-k (song, score, explanation) tuples, same shape as the Module 3 function."""
    return [
        (song, score, "; ".join(reasons) if reasons else "no matching attributes")
        for song, score, reasons in rank_songs(prefs, songs)[:k]
    ]


class Recommender:
    """OOP wrapper over the same scorer (kept because the Module 3 tests target it)."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    @staticmethod
    def _prefs(user: UserProfile) -> Dict:
        return {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "acousticness": 0.8 if user.likes_acoustic else 0.2,
        }

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        catalog = [asdict(s) for s in self.songs]
        by_id = {s.id: s for s in self.songs}
        return [by_id[song["id"]] for song, _, _ in rank_songs(self._prefs(user), catalog)[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        _, reasons = score_song(self._prefs(user), asdict(song))
        return "; ".join(reasons) if reasons else "no matching attributes"
