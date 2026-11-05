"""
Turns a free-text request plus retrieved context cards into a scoring profile.

Two signal sources are merged:
  * explicit  - words the user typed that match the catalog vocabulary
                (genres, moods) or a small numeric lexicon ("high energy", "not acoustic")
  * context   - genres, moods and feature RANGES from the retrieved context cards

Explicit numeric targets override the range a card suggests for that feature.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .recommender import NUMERIC_ATTRS, Target
from .retrieval import Card

# A second card is only blended in if it is nearly as relevant as the best one.
SECONDARY_CARD_RATIO = 0.6
MAX_CONTEXT_CARDS = 2

_NORMALISE = [(re.compile(r"\blo[- ]fi\b"), "lofi")]

# (regex, value). Negated forms are checked first so "not acoustic" never reads as "acoustic".
_NUMERIC_LEXICON: List[Tuple[str, re.Pattern, float]] = [
    ("acousticness", re.compile(r"\b(?:not|non|no|without|isn't)[ -](?:very |too )?acoustic\b"), 0.10),
    ("acousticness", re.compile(r"\b(?:acoustic|unplugged)\b"), 0.85),
    ("acousticness", re.compile(r"\b(?:electronic|synthetic|digital)\b"), 0.10),
    ("danceability", re.compile(r"\b(?:undanceable|(?:not|non|no)[ -](?:very )?danceable)\b"), 0.15),
    ("danceability", re.compile(r"\b(?:danceable|dance|groovy)\b"), 0.88),
    ("energy", re.compile(r"\b(?:max(?:imum)?|highest|very high|super high|high)[ -]energy\b|\bloud\b|\bpumped\b"), 0.92),
    ("energy", re.compile(r"\b(?:low|lowest|minimal)[ -]energy\b|\bcalm\b|\bmellow\b|\bquiet\b|\bgentle\b|\bsoft\b"), 0.25),
]


@dataclass
class Profile:
    genres: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    context_genres: List[str] = field(default_factory=list)
    context_moods: List[str] = field(default_factory=list)
    targets: Dict[str, Target] = field(default_factory=dict)
    explicit_numeric: Set[str] = field(default_factory=set)
    cards: List[Tuple[Card, float]] = field(default_factory=list)
    genre_scale: float = 1.0

    @property
    def has_explicit_signal(self) -> bool:
        return bool(self.genres or self.moods or self.explicit_numeric)

    @property
    def has_signal(self) -> bool:
        return bool(
            self.has_explicit_signal or self.context_genres or self.context_moods or self.targets
        )

    def to_prefs(self) -> Dict:
        prefs: Dict = {"genre_scale": self.genre_scale}
        if self.genres:
            prefs["genres"] = list(self.genres)
        if self.moods:
            prefs["moods"] = list(self.moods)
        if self.context_genres:
            prefs["context_genres"] = list(self.context_genres)
        if self.context_moods:
            prefs["context_moods"] = list(self.context_moods)
        prefs.update(self.targets)
        return prefs

    def describe(self) -> str:
        parts = []
        if self.genres:
            parts.append("genre=" + "/".join(self.genres))
        if self.moods:
            parts.append("mood=" + "/".join(self.moods))
        if self.context_genres:
            parts.append("context genres~" + "/".join(self.context_genres))
        if self.context_moods:
            parts.append("context moods~" + "/".join(self.context_moods))
        for attr in NUMERIC_ATTRS:
            if attr in self.targets:
                t = self.targets[attr]
                parts.append(f"{attr}=" + (f"{t[0]:.2f}-{t[1]:.2f}" if isinstance(t, tuple) else f"{t:.2f}"))
        if self.genre_scale != 1.0:
            parts.append(f"genre weight x{self.genre_scale:g}")
        return ", ".join(parts) if parts else "(no preferences detected)"

    def copy(self) -> "Profile":
        return Profile(
            genres=list(self.genres), moods=list(self.moods),
            context_genres=list(self.context_genres), context_moods=list(self.context_moods),
            targets=dict(self.targets), explicit_numeric=set(self.explicit_numeric),
            cards=list(self.cards), genre_scale=self.genre_scale,
        )


def parse_explicit(text: str, genre_vocab: Sequence[str], mood_vocab: Sequence[str]) -> Profile:
    """Pull catalog genres/moods and numeric hints straight out of the text."""
    working = text.lower()
    for pattern, repl in _NORMALISE:
        working = pattern.sub(repl, working)

    profile = Profile()
    # Longest genre names first so "ambient electronic" is not also read as "ambient".
    for genre in sorted(genre_vocab, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(genre.lower())}\b")
        if pattern.search(working):
            profile.genres.append(genre)
            working = pattern.sub(" ", working)

    for mood in mood_vocab:
        if re.search(rf"\b{re.escape(mood.lower())}\b", working):
            profile.moods.append(mood)

    for attr, pattern, value in _NUMERIC_LEXICON:
        if attr in profile.explicit_numeric:
            continue  # first (most specific) rule wins
        if pattern.search(working):
            profile.targets[attr] = value
            profile.explicit_numeric.add(attr)
            working = pattern.sub(" ", working)
    return profile


def build_profile(
    text: str,
    retrieved: List[Tuple[Card, float]],
    genre_vocab: Sequence[str],
    mood_vocab: Sequence[str],
) -> Profile:
    """Merge explicit signals with the retrieved context cards into one scoring profile."""
    profile = parse_explicit(text, genre_vocab, mood_vocab)
    if not retrieved:
        return profile

    best = retrieved[0][1]
    chosen = [retrieved[0]] + [
        (c, s) for c, s in retrieved[1:MAX_CONTEXT_CARDS] if s >= SECONDARY_CARD_RATIO * best
    ]
    profile.cards = chosen

    for card, _ in chosen:
        for g in card.genres:
            if g not in profile.context_genres:
                profile.context_genres.append(g)
        for m in card.moods:
            if m not in profile.context_moods:
                profile.context_moods.append(m)

    for attr in NUMERIC_ATTRS:
        if attr in profile.explicit_numeric:
            continue  # what the user typed beats what the card suggests
        spans = [card.ranges[attr] for card, _ in chosen if attr in card.ranges]
        if spans:
            profile.targets[attr] = (min(lo for lo, _ in spans), max(hi for _, hi in spans))
    return profile
