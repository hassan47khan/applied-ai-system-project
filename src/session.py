"""
Multi-turn conversations: "more upbeat", "less jazz", "not #3", "more like #2".

A `Conversation` keeps the previous turn's profile and results. A short follow-up that
reads as a refinement edits that profile and re-ranks (agent.refine) instead of starting
over, so "more upbeat" means "the same thing, with more energy". Anything that doesn't
parse as a refinement is treated as a new request.

The refinement parser is deterministic on purpose: follow-ups are short, their meaning is
narrow, and a rule you can read is easier to trust than a model for "drop #3".
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .agent import AgentResult, VibeFinderAgent
from .profile import Profile, widen_rejection
from .recommender import NUMERIC_ATTRS

STEP = 0.2          # how far one "more"/"less" moves a feature
HALF_WIDTH = 0.15   # a refined feature becomes a range of this half-width
LIKE_HALF_WIDTH = 0.10
MAX_REFINEMENT_WORDS = 8

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "last": -1}
_RANK = r"(?:#\s*|number\s+|no\.?\s*|the\s+)?(\d{1,2}|" + "|".join(_ORDINALS) + r")(?:\s+one)?"

_FEATURE_RULES = [
    ("energy", +1, r"\b(?:more (?:upbeat|energy|energetic|intense|hype|loud|power)|faster|louder|harder|"
                   r"pick it up|turn it up|higher energy|more of a banger)\b"),
    ("energy", -1, r"\b(?:calmer|softer|quieter|slower|mellower|less (?:energy|energetic|intense|loud|upbeat|hype)|"
                   r"more (?:calm|chill|mellow|relaxed|relaxing|gentle)|tone it down|lower energy)\b"),
    ("danceability", +1, r"\b(?:more (?:danceable|dancey|groovy|groove)|danceable(?:r)?)\b"),
    ("danceability", -1, r"\b(?:less (?:danceable|dancey|groovy))\b"),
    ("acousticness", +1, r"\b(?:more acoustic|unplugged|less electronic)\b"),
    ("acousticness", -1, r"\b(?:less acoustic|more electronic|more synth)\b"),
]
_DROP = re.compile(r"\b(?:not|skip|drop|remove|no|hate|lose|without)\s+" + _RANK + r"\b")
_LIKE = re.compile(r"\b(?:more (?:songs )?like|similar to|something like|like)\s+" + _RANK + r"\b")
_FRESH = re.compile(r"\b(?:something else|different (?:ones|songs)|other (?:ones|songs)|new ones|"
                    r"more options|show me more|next batch|again but different)\b")
_LESS_WORDS = r"(?:less|no|not|without|skip|drop|fewer|no more|enough)"
_MORE_WORDS = r"(?:more|add|some|extra|plus)"


@dataclass
class Refinement:
    shifts: Dict[str, int] = field(default_factory=dict)        # feature -> +1 / -1
    add_genres: List[str] = field(default_factory=list)
    avoid_genres: List[str] = field(default_factory=list)
    add_moods: List[str] = field(default_factory=list)
    avoid_moods: List[str] = field(default_factory=list)
    drop_ranks: List[int] = field(default_factory=list)
    like_rank: Optional[int] = None
    fresh: bool = False

    @property
    def empty(self) -> bool:
        return not (self.shifts or self.add_genres or self.avoid_genres or self.add_moods
                    or self.avoid_moods or self.drop_ranks or self.like_rank or self.fresh)

    def describe(self) -> str:
        parts = [f"{'more' if d > 0 else 'less'} {a}" for a, d in self.shifts.items()]
        parts += [f"+{g}" for g in self.add_genres] + [f"-{g}" for g in self.avoid_genres]
        parts += [f"+{m}" for m in self.add_moods] + [f"-{m}" for m in self.avoid_moods]
        parts += [f"drop #{r}" for r in self.drop_ranks]
        if self.like_rank:
            parts.append(f"like #{self.like_rank}")
        if self.fresh:
            parts.append("new options")
        return ", ".join(parts)


def _rank(token: str, shown: int) -> Optional[int]:
    value = _ORDINALS.get(token, None)
    if value is None:
        try:
            value = int(token)
        except ValueError:
            return None
    if value == -1:
        value = shown
    return value if 1 <= value <= shown else None


def parse_refinement(text: str, shown: int, genre_vocab: Sequence[str],
                     mood_vocab: Sequence[str]) -> Optional[Refinement]:
    """A Refinement if `text` is a follow-up to the previous turn, else None."""
    t = re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip(" .!?")
    if not t or len(t.split()) > MAX_REFINEMENT_WORDS:
        return None
    ref = Refinement()
    for attr, direction, pattern in _FEATURE_RULES:
        if attr not in ref.shifts and re.search(pattern, t):
            ref.shifts[attr] = direction
    for m in _DROP.finditer(t):
        r = _rank(m.group(1), shown)
        if r and r not in ref.drop_ranks:
            ref.drop_ranks.append(r)
    like = _LIKE.search(t)
    if like:
        ref.like_rank = _rank(like.group(1), shown)
    ref.fresh = bool(_FRESH.search(t))
    for g in sorted(genre_vocab, key=len, reverse=True):
        name = re.escape(g.lower())
        if re.search(rf"\b{_LESS_WORDS}\s+{name}\b", t):
            ref.avoid_genres.append(g)
        elif re.search(rf"\b{_MORE_WORDS}\s+{name}\b", t):
            ref.add_genres.append(g)
    for m in mood_vocab:
        name = re.escape(m.lower())
        if re.search(rf"\b{_LESS_WORDS}\s+{name}\b", t):
            ref.avoid_moods.append(m)
        elif re.search(rf"\b{_MORE_WORDS}\s+{name}\b", t):
            ref.add_moods.append(m)
    return None if ref.empty else ref


def _center(target) -> float:
    return (target[0] + target[1]) / 2 if isinstance(target, tuple) else float(target)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def apply_refinement(profile: Profile, ref: Refinement, previous: AgentResult,
                     genre_vocab: Sequence[str]) -> Profile:
    p = profile.copy()
    p.genre_scale = 1.0
    shown = [r.song for r in previous.recommendations]
    for attr, direction in ref.shifts.items():
        if attr in p.targets:
            center = _center(p.targets[attr])
        elif shown:
            center = sum(s[attr] for s in shown) / len(shown)
        else:
            center = 0.5
        center = _clamp(center + direction * STEP)
        p.targets[attr] = (round(_clamp(center - HALF_WIDTH), 3), round(_clamp(center + HALF_WIDTH), 3))
        p.explicit_numeric.add(attr)
        p.rejected_numeric.discard(attr)
    for g in ref.avoid_genres:
        for widened in widen_rejection(g, genre_vocab):
            if widened not in p.avoid_genres:
                p.avoid_genres.append(widened)
    p.genres = [g for g in p.genres + [g for g in ref.add_genres if g not in p.genres]
                if g not in p.avoid_genres]
    p.context_genres = [g for g in p.context_genres if g not in p.avoid_genres]
    p.avoid_moods = list(dict.fromkeys(p.avoid_moods + ref.avoid_moods))
    p.moods = [m for m in p.moods + [m for m in ref.add_moods if m not in p.moods] if m not in p.avoid_moods]
    p.context_moods = [m for m in p.context_moods if m not in p.avoid_moods]
    if ref.like_rank:
        seed = shown[ref.like_rank - 1]
        for attr in NUMERIC_ATTRS:
            if attr not in ref.shifts:
                p.targets[attr] = (round(_clamp(seed[attr] - LIKE_HALF_WIDTH), 3),
                                   round(_clamp(seed[attr] + LIKE_HALF_WIDTH), 3))
                p.explicit_numeric.add(attr)
        if seed["genre"] not in p.context_genres and seed["genre"] not in p.avoid_genres:
            p.context_genres.insert(0, seed["genre"])
        if seed["mood"] not in p.context_moods and seed["mood"] not in p.avoid_moods:
            p.context_moods.insert(0, seed["mood"])
    return p


class Conversation:
    """One listener's session. Not thread-safe; the UI keeps one per browser session."""

    def __init__(self, agent: VibeFinderAgent):
        self.agent = agent
        self.turns: List[AgentResult] = []
        self.dismissed: Set[int] = set()

    @property
    def last(self) -> Optional[AgentResult]:
        return self.turns[-1] if self.turns else None

    def send(self, text: str) -> AgentResult:
        last = self.last
        profile = getattr(last, "_profile", None) if last else None
        ref = None
        if last is not None and profile is not None and last.recommendations and isinstance(text, str):
            ref = parse_refinement(text, len(last.recommendations), self.agent.genre_vocab,
                                   self.agent.mood_vocab)
        if ref is None:
            self.dismissed = set()
            result = self.agent.run(text)
        else:
            shown_ids = [r.song["id"] for r in last.recommendations]
            self.dismissed |= {shown_ids[r - 1] for r in ref.drop_ranks}
            if ref.fresh:
                self.dismissed |= set(shown_ids)
            if ref.like_rank:
                self.dismissed.add(shown_ids[ref.like_rank - 1])   # "like #2" means other songs like it
            new_profile = apply_refinement(profile, ref, last, self.agent.genre_vocab)
            result = self.agent.refine(new_profile, f"{text} -> {ref.describe()}", self.dismissed)
        self.turns.append(result)
        return result

    def reset(self) -> None:
        self.turns.clear()
        self.dismissed.clear()
