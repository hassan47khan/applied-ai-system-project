"""
Cover art and 30-second previews from Deezer's public search API (no key, no account).

Used only by the Streamlit UI; the recommender core never imports this, so recommendations
and tests stay offline and deterministic. Preview URLs are signed and expire, so they are
fetched when a card is rendered and never written to disk.

    lookup("Bad Habits", "Ed Sheeran") -> TrackMedia(cover=..., preview=..., link=...) or None
"""

import json
import logging
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

log = logging.getLogger("vibefinder.enrich")

SEARCH_URL = "https://api.deezer.com/search?"
TIMEOUT_S = 3.0
USER_AGENT = "VibeFinder/3.0 (student project; music recommender demo)"

# "Song - Remastered 2011", "Song (feat. X)", "Song - From 'Film'" -> "Song" for searching.
_TITLE_NOISE = re.compile(r"\s*(?:-\s.*|\((?:feat|with|ft)\.?[^)]*\)|\[[^\]]*\])\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class TrackMedia:
    cover: Optional[str]
    preview: Optional[str]
    link: Optional[str]
    matched_title: str
    matched_artist: str


def _norm(text: str) -> str:
    """Lower-case, accents folded ("Sebastián" == "Sebastian"), punctuation to spaces."""
    folded = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def search_title(title: str) -> str:
    return _TITLE_NOISE.sub("", title).strip() or title


def _fetch(query: str) -> list:
    url = SEARCH_URL + urllib.parse.urlencode({"q": query, "limit": 5})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:   # noqa: S310 (fixed host)
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, list) else []


def pick_match(results: list, title: str, artist: str) -> Optional[dict]:
    """The result whose artist matches; otherwise one whose title matches; else None.
    A wrong cover is worse than no cover, so a weak match is rejected."""
    want_title, want_artist = _norm(search_title(title)), _norm(artist)
    for item in results:
        if _norm(item.get("artist", {}).get("name", "")) == want_artist:
            return item
    for item in results:
        got = _norm(item.get("title", ""))
        if got and (got == want_title or got.startswith(want_title)) and want_artist.split()[:1] == \
                _norm(item.get("artist", {}).get("name", "")).split()[:1]:
            return item
    return None


@lru_cache(maxsize=512)
def lookup(title: str, artist: str) -> Optional[TrackMedia]:
    """Never raises: any network or parsing problem returns None (the UI shows a placeholder).
    VIBEFINDER_NO_MEDIA=1 switches lookups off (the test suite does)."""
    if os.environ.get("VIBEFINDER_NO_MEDIA") == "1":
        return None
    try:
        results = _fetch(f"{search_title(title)} {artist}")
    except Exception as exc:   # offline, timeout, HTTP error, bad JSON
        log.info("Deezer lookup failed for %r: %s", title, type(exc).__name__)
        return None
    item = pick_match(results, title, artist)
    if item is None:
        return None
    album = item.get("album") or {}
    return TrackMedia(
        cover=album.get("cover_medium") or album.get("cover"),
        preview=item.get("preview") or None,
        link=item.get("link"),
        matched_title=item.get("title", ""),
        matched_artist=(item.get("artist") or {}).get("name", ""),
    )
