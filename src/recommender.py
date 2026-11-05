import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
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
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Load the song catalog from a CSV file into a list of dicts with numeric fields coerced to float/int."""
    songs: List[Dict] = []
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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

# --- Scoring weights (Phase 4 "Weight Shift" experiment) -------------------
# Baseline recipe was: GENRE=2.0, MOOD=1.0, NUMERIC=1.0 per feature.
# Weight Shift: halve the genre weight, double the numeric-feature weight,
# leave mood unchanged. Change these constants to re-run the experiment.
GENRE_WEIGHT = 1.0        # was 2.0
MOOD_WEIGHT = 1.0         # unchanged
NUMERIC_WEIGHT = 2.0      # was 1.0, applied to energy / danceability / acousticness
# ---------------------------------------------------------------------------

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user_prefs and return (total_score, list_of_reason_strings)."""
    score = 0.0
    reasons: List[str] = []

    if "genre" in user_prefs and user_prefs["genre"] == song.get("genre"):
        score += GENRE_WEIGHT
        reasons.append(f"genre match (+{GENRE_WEIGHT:.1f})")

    if "mood" in user_prefs and user_prefs["mood"] == song.get("mood"):
        score += MOOD_WEIGHT
        reasons.append(f"mood match (+{MOOD_WEIGHT:.1f})")

    for attr in ("energy", "danceability", "acousticness"):
        if attr in user_prefs and attr in song:
            target = float(user_prefs[attr])
            value = float(song[attr])
            proximity = max(0.0, 1.0 - abs(target - value)) * NUMERIC_WEIGHT
            if proximity > 0:
                score += proximity
                reasons.append(f"{attr} proximity (+{proximity:.2f})")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Return the top-k (song, score, explanation) tuples ranked by score_song, highest first."""
    scored = [
        (song, *score_song(user_prefs, song))
        for song in songs
    ]
    scored.sort(key=lambda item: item[1], reverse=True)

    return [
        (song, score, "; ".join(reasons) if reasons else "no matching attributes")
        for song, score, reasons in scored[:k]
    ]
