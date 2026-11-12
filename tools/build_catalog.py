"""
Build data/songs.csv from a real, public dataset.

Source: maharshipandya/spotify-tracks-dataset on Hugging Face (BSD license), about 114k
Spotify tracks with the audio features this project scores on. Spotify closed its own
audio-features API to new apps in November 2024, so a published snapshot is the only
practical source of energy / danceability / acousticness for real songs.

    python tools/build_catalog.py                       # download (cached) and build
    python tools/build_catalog.py --source dataset.csv  # build from a local copy
    python tools/build_catalog.py --per-genre 25

The build is deterministic: the same input file always gives the same catalog. Selection
rules (also written to data/CATALOG_PROVENANCE.md):
  * Spotify genre tags are mapped onto a smaller vocabulary (GENRE_MAP below); tags with
    no clear listening "feel" (comedy, children, show tunes...) are dropped.
  * mood is derived from valence / energy / acousticness / instrumentalness (derive_mood).
  * duplicates (same title + lead artist) keep the most popular copy; spoken-word tracks
    (speechiness > 0.66) and tracks under 90 s are dropped.
  * per genre: the most popular tracks, at most 2 per artist, up to --per-genre.
"""

import argparse
import csv
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
DATASET_PAGE = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset"
SOURCE_URL = DATASET_PAGE + "/resolve/main/dataset.csv"
CACHE_PATH = ROOT / ".cache" / "datasets" / "spotify_tracks.csv"
OUT_PATH = ROOT / "data" / "songs.csv"
PROVENANCE_PATH = ROOT / "data" / "CATALOG_PROVENANCE.md"

# Spotify genre tag -> VibeFinder genre. Multi-word genres share a word with their parent
# ("indie pop" ~ "pop") so the scorer's related-genre credit keeps working.
GENRE_MAP: Dict[str, str] = {
    "pop": "pop", "power-pop": "pop", "pop-film": "pop",
    "indie-pop": "indie pop",
    "synth-pop": "synth pop",
    "k-pop": "kpop",
    "latin": "latin pop", "latino": "latin pop",
    "reggaeton": "reggaeton",
    "salsa": "latin", "samba": "latin", "tango": "latin", "mpb": "latin", "forro": "latin",
    "rock": "rock", "alt-rock": "rock", "hard-rock": "rock", "rock-n-roll": "rock",
    "indie": "indie rock", "alternative": "indie rock", "grunge": "indie rock",
    "punk": "punk", "punk-rock": "punk",
    "metal": "metal", "heavy-metal": "metal",
    "hip-hop": "hip hop",
    "r-n-b": "soul", "soul": "soul",
    "funk": "funk", "disco": "funk",
    "jazz": "jazz",
    "blues": "blues",
    "classical": "classical", "opera": "classical",
    "piano": "piano",
    "ambient": "ambient", "new-age": "ambient", "sleep": "ambient",
    "study": "lofi",   # Spotify's "chill" tag mixes ODESZA, EDM remixes and pop; not lofi
    "electronic": "electronic", "idm": "electronic", "trip-hop": "electronic",
    "house": "house", "deep-house": "house", "progressive-house": "house", "chicago-house": "house",
    "techno": "techno", "minimal-techno": "techno", "detroit-techno": "techno",
    "edm": "edm", "dance": "edm", "electro": "edm",
    "drum-and-bass": "drum and bass",
    "folk": "folk", "acoustic": "folk", "singer-songwriter": "folk", "songwriter": "folk",
    "country": "country", "bluegrass": "country", "honky-tonk": "country",
    "reggae": "reggae", "dub": "reggae", "dancehall": "reggae",
    "world-music": "world fusion", "afrobeat": "world fusion", "indian": "world fusion",
}

MIN_DURATION_MS = 90_000
MAX_SPEECHINESS = 0.66
MAX_PER_ARTIST = 2
MAX_TITLE_CHARS = 60
MIN_TEMPO = 40.0
MIN_DANCEABILITY = 0.08
_NOISE = re.compile(r"\b(?:white|brown|pink|green) noise\b|\bnoise\b.*\bsleep|\brain sounds?\b|"
                    r"\bloopable\b|\bfan sounds?\b", re.IGNORECASE)


def derive_mood(valence: float, energy: float, acousticness: float, instrumentalness: float) -> str:
    """First matching rule wins. Thresholds sit near the dataset's quartiles."""
    if energy >= 0.80 and valence < 0.45:
        return "intense"
    if energy >= 0.70 and valence >= 0.45:
        return "energetic"
    # Instrumental, low-energy music is low-valence almost by definition; checking it
    # before "melancholy" stops every piano or ambient piece being labelled sad.
    if instrumentalness >= 0.40 and energy < 0.60:
        return "focused"
    if valence < 0.30 and energy < 0.55:
        return "melancholy"
    if valence >= 0.65:
        return "happy"
    if valence < 0.45 and energy >= 0.45:
        return "moody"
    if energy < 0.50 and acousticness >= 0.50:
        return "relaxed"
    return "chill"


def _candidate(row: Dict) -> Optional[Dict]:
    genre = GENRE_MAP.get(row.get("track_genre", ""))
    if genre is None:
        return None
    title = (row.get("track_name") or "").strip()
    artist = (row.get("artists") or "").split(";")[0].strip()
    if not title or not artist or len(title) > MAX_TITLE_CHARS:
        return None
    try:
        if int(row["duration_ms"]) < MIN_DURATION_MS or float(row["speechiness"]) > MAX_SPEECHINESS:
            return None
        # White/brown noise and rain loops: no beat, no melody, not music to recommend.
        if float(row["tempo"]) < MIN_TEMPO or float(row["danceability"]) < MIN_DANCEABILITY \
                or _NOISE.search(title):
            return None
        song = {
            "title": title, "artist": artist, "genre": genre,
            "energy": round(float(row["energy"]), 3),
            "tempo_bpm": round(float(row["tempo"]), 1),
            "valence": round(float(row["valence"]), 3),
            "danceability": round(float(row["danceability"]), 3),
            "acousticness": round(float(row["acousticness"]), 3),
            "popularity": int(row["popularity"]),
            "track_id": row["track_id"],
        }
        song["mood"] = derive_mood(song["valence"], song["energy"], song["acousticness"],
                                   float(row["instrumentalness"]))
    except (KeyError, TypeError, ValueError):
        return None
    return song


def build(source: Path, per_genre: int) -> List[Dict]:
    best: Dict[tuple, Dict] = {}
    with open(source, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            song = _candidate(row)
            if song is None:
                continue
            key = (song["title"].lower(), song["artist"].lower())
            old = best.get(key)
            # Keep the most popular copy; ties go to the smaller track id (deterministic).
            if old is None or (song["popularity"], old["track_id"]) > (old["popularity"], song["track_id"]):
                best[key] = song

    by_genre: Dict[str, List[Dict]] = defaultdict(list)
    for song in best.values():
        by_genre[song["genre"]].append(song)

    chosen: List[Dict] = []
    for genre in sorted(by_genre):
        per_artist: Dict[str, int] = defaultdict(int)
        ranked = sorted(by_genre[genre], key=lambda s: (-s["popularity"], s["title"], s["artist"]))
        picked = 0
        for song in ranked:
            if picked >= per_genre:
                break
            if per_artist[song["artist"].lower()] >= MAX_PER_ARTIST:
                continue
            per_artist[song["artist"].lower()] += 1
            chosen.append(song)
            picked += 1
    for i, song in enumerate(chosen, start=1):
        song["id"] = i
    return chosen


FIELDS = ["id", "title", "artist", "genre", "mood", "energy", "tempo_bpm", "valence",
          "danceability", "acousticness", "popularity", "track_id"]


def write(songs: List[Dict], out: Path) -> None:
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for song in songs:
            writer.writerow({k: song[k] for k in FIELDS})


def write_provenance(songs: List[Dict], per_genre: int) -> None:
    genres: Dict[str, int] = defaultdict(int)
    moods: Dict[str, int] = defaultdict(int)
    for s in songs:
        genres[s["genre"]] += 1
        moods[s["mood"]] += 1
    mapping: Dict[str, List[str]] = defaultdict(list)
    for tag, genre in sorted(GENRE_MAP.items()):
        mapping[genre].append(tag)
    lines = [
        "# Song catalog provenance",
        "",
        "`data/songs.csv` is generated by `python tools/build_catalog.py`. Do not edit it by hand.",
        "",
        f"- **Source:** [maharshipandya/spotify-tracks-dataset]({DATASET_PAGE}) on Hugging Face, "
        "a snapshot of Spotify track metadata and audio features.",
        "- **License:** BSD (as declared on the dataset card). Only factual metadata (titles, artist "
        "names, audio-feature numbers) is used; no audio is redistributed.",
        "- **Why a snapshot:** Spotify closed its audio-features endpoint to new apps on "
        "2024-11-27, so live energy/danceability/acousticness values are no longer available.",
        f"- **Size:** {len(songs)} tracks, up to {per_genre} per genre and at most {MAX_PER_ARTIST} "
        "per artist, chosen by Spotify popularity.",
        f"- **Filters:** duration >= {MIN_DURATION_MS // 1000}s, speechiness <= {MAX_SPEECHINESS}, "
        f"title <= {MAX_TITLE_CHARS} characters; duplicates (title + lead artist) keep the most popular copy.",
        "- **Mood is derived, not labelled.** `derive_mood()` assigns one of intense / energetic / "
        "melancholy / focused / happy / moody / relaxed / chill from valence, energy, acousticness "
        "and instrumentalness. It is a heuristic and will be wrong for some songs.",
        "- **Known bias:** the source over-represents tracks that were popular on Spotify around 2022 "
        "and English-language releases, and popularity-first selection amplifies that.",
        "",
        "## Genres",
        "",
        "| genre | tracks | Spotify tags |",
        "|---|---:|---|",
    ]
    lines += [f"| {g} | {genres[g]} | {', '.join(mapping[g])} |" for g in sorted(genres)]
    lines += ["", "## Moods", "", "| mood | tracks |", "|---|---:|"]
    lines += [f"| {m} | {moods[m]} |" for m in sorted(moods)]
    PROVENANCE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build data/songs.csv from the HF Spotify dataset")
    parser.add_argument("--source", type=Path, help="local copy of dataset.csv (skips download)")
    parser.add_argument("--per-genre", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    source = args.source
    if source is None:
        source = CACHE_PATH
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {SOURCE_URL} ...")
            urllib.request.urlretrieve(SOURCE_URL, source)
    songs = build(source, args.per_genre)
    if not songs:
        print("No songs selected; is the source file the right dataset?", file=sys.stderr)
        return 1
    write(songs, args.out)
    if args.out == OUT_PATH:
        write_provenance(songs, args.per_genre)
    print(f"Wrote {len(songs)} songs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
