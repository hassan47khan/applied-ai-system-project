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
# Cards whose ranges for a feature sit this far apart describe opposite situations:
# merging them would give a range that nearly every song "fits".
BLEND_CONFLICT_GAP = 0.15

_NORMALISE = [
    (re.compile(r"[‘’ʼ]"), "'"),   # phones and macOS type curly apostrophes
    (re.compile(r"\s+"), " "),
    (re.compile(r"\blo[- ]fi\b"), "lofi"),
]

# Words of multi-word genres that also mean something else, so alone they don't name a genre
# ("classic rock", "around the world", and "electronic" is already a feature word below).
_AMBIGUOUS_GENRE_WORDS = frozenset({"classic", "world", "electronic"})

# What starts a rejection. Everything from here to the end of the clause is rejected.
_NEGATION_CUE = re.compile(
    r"(?<!why )\b(?:"
    r"anything (?:but|except|other than)|everything (?:but|except)|all but|"
    r"(?:other|rather) than|instead of|(?:apart|aside) from|except(?: for)?|"
    r"exclud(?:e|es|ing)|without|minus|"
    r"avoid(?:s|ing)?|skip(?:s|ping)?|hat(?:e|es|ed|ing)|dislik(?:e|es|ed|ing)|"
    r"(?:can'?t|cannot) stand|(?:sick|tired) of|fed up with|stay away from|stop playing|"
    r"not in the mood for|(?:do|does|did|would|will|should) not|"
    r"(?:don|doesn|didn|wouldn|won|shouldn|isn|aren|wasn|weren|ain)'?t|"
    r"not|no|non|never|nothing(?! but\b)|neither|nor|none of|less"
    r")\b"
    # ...except idioms that reject nothing: "not sure", "no idea", "don't mind jazz".
    r"(?![ -]+(?:sure|only|just|know|care|mind|matter|idea|clue|problem|stop|doubt|worry)\b)"
    # Verbs between the cue and the thing rejected: "don't want to hear jazz".
    r"(?: (?:want|wanna|like|love|need|play|give|put|hear|listen to|listening to)"
    r"(?: (?:me|to hear|to listen to|to play))?)?"
)

# What ends a rejection: punctuation, or a word that starts a new thought or a context
# ("no pop for the gym" rejects pop, not the gym). "and"/"or" are absent on purpose, so
# "not jazz or rock" and "I hate jazz and rock" reject both. Checked after the first word
# of the scope, so "don't play rock" and "not for the gym" still count.
_SCOPE_END = re.compile(
    r"[,.;:!?()]|\b(?:but|however|though|although|yet|instead|rather|just|only|because|since|"
    r"so|then|while|when|during|for|to|at|in|on|with|i|we|give|gimme|play|make|want|need|"
    r"something|maybe|please|pick|love|like|prefer|enjoy)\b"
)
_FIRST_WORD = re.compile(r"[ -]+[\w'-]+")

# (feature, words, target when asked for, target when rejected). Rejecting a word rules out
# that end of the scale without pretending the user asked for the extreme opposite, so the
# "not loud" style entries are ranges. First matching rule per feature wins.
_NUMERIC_LEXICON: List[Tuple[str, re.Pattern, Target, Target]] = [
    ("acousticness", re.compile(r"\b(?:acoustic|unplugged)\b"), 0.85, 0.10),
    ("acousticness", re.compile(r"\b(?:electronic|synthetic|digital)\b"), 0.10, (0.40, 1.00)),
    ("danceability", re.compile(r"\bundanceable\b"), 0.15, 0.88),
    ("danceability", re.compile(r"\b(?:danceable|dance|groovy)\b"), 0.88, 0.15),
    ("energy", re.compile(r"\b(?:max(?:imum)?|highest|very high|super high|high)[ -]energy\b"
                          r"|\bloud\b|\bpumped\b"), 0.92, (0.00, 0.60)),
    ("energy", re.compile(r"\b(?:low|lowest|minimal)[ -]energy\b"
                          r"|\bcalm\b|\bmellow\b|\bquiet\b|\bgentle\b|\bsoft\b"), 0.25, (0.50, 1.00)),
]


@dataclass
class Profile:
    genres: List[str] = field(default_factory=list)
    avoid_genres: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    avoid_moods: List[str] = field(default_factory=list)
    context_genres: List[str] = field(default_factory=list)
    context_moods: List[str] = field(default_factory=list)
    targets: Dict[str, Target] = field(default_factory=dict)
    explicit_numeric: Set[str] = field(default_factory=set)
    rejected_numeric: Set[str] = field(default_factory=set)   # subset set by "not loud" etc.
    cards: List[Tuple[Card, float]] = field(default_factory=list)
    genre_scale: float = 1.0
    context_clash: Optional[str] = None   # set when two matched cards contradicted each other

    @property
    def has_explicit_signal(self) -> bool:
        # A rejected feature constrains the answer but doesn't say what the user wants.
        return bool(self.genres or self.moods or self.explicit_numeric - self.rejected_numeric)

    @property
    def has_preference(self) -> bool:
        """Do we know what the user wants, not just what they ruled out?"""
        return bool(
            self.has_explicit_signal or self.context_genres or self.context_moods
            or set(self.targets) - self.rejected_numeric
        )

    @property
    def has_signal(self) -> bool:
        return bool(self.has_preference or self.avoid_genres or self.avoid_moods
                    or self.rejected_numeric)

    def allows(self, song: Dict) -> bool:
        """Rejections are hard filters: a rejected song never competes on score."""
        return song["genre"] not in self.avoid_genres and song["mood"] not in self.avoid_moods

    def to_prefs(self) -> Dict:
        prefs: Dict = {"genre_scale": self.genre_scale}
        if self.genres:
            prefs["genres"] = list(self.genres)
        if self.avoid_genres:
            prefs["avoid_genres"] = list(self.avoid_genres)
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
        if self.avoid_genres:
            parts.append("avoid=" + "/".join(self.avoid_genres))
        if self.moods:
            parts.append("mood=" + "/".join(self.moods))
        if self.avoid_moods:
            parts.append("avoid mood=" + "/".join(self.avoid_moods))
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
            genres=list(self.genres), avoid_genres=list(self.avoid_genres), moods=list(self.moods),
            avoid_moods=list(self.avoid_moods), context_genres=list(self.context_genres), context_moods=list(self.context_moods),
            targets=dict(self.targets), explicit_numeric=set(self.explicit_numeric),
            rejected_numeric=set(self.rejected_numeric),
            cards=list(self.cards), genre_scale=self.genre_scale,
            context_clash=self.context_clash,
        )


def _normalise(text: str) -> str:
    working = text.lower()
    for pattern, repl in _NORMALISE:
        working = pattern.sub(repl, working)
    return working


def _split_negated(text: str) -> Tuple[str, str]:
    """Split normalised text into (what is wanted, what is rejected)."""
    kept: List[str] = []
    rejected: List[str] = []
    pos = 0
    while True:
        cue = _NEGATION_CUE.search(text, pos)
        if cue is None:
            break
        first = _FIRST_WORD.match(text, cue.end())
        if first is None:               # "no, I want jazz": the cue negates nothing
            kept.append(text[pos:cue.end()])
            pos = cue.end()
            continue
        end = _SCOPE_END.search(text, first.end())
        stop = end.start() if end else len(text)
        kept.append(text[pos:cue.start()])
        rejected.append(text[first.start():stop])
        pos = stop
    kept.append(text[pos:])
    return " ".join(kept), " ".join(rejected)


def strip_negated(text: str) -> str:
    """Remove rejected phrases so something rejected can't pull in its own context card."""
    return re.sub(r"\s+", " ", _split_negated(_normalise(text))[0]).strip()


def _genre_terms(genre_vocab: Sequence[str]) -> List[str]:
    """Everything that names a genre: catalog genres plus words like 'folk' or 'indie'.
    Longest first so "ambient electronic" is not also read as "ambient"."""
    genres = {g.lower() for g in genre_vocab}
    words = {w for g in genres for w in g.split()} - _AMBIGUOUS_GENRE_WORDS
    return sorted(genres | words, key=lambda t: (-len(t), t))


def _containing(term: str, genre_vocab: Sequence[str]) -> List[str]:
    """Catalog genres that include every word of `term` ('pop' -> pop, indie pop, latin pop)."""
    words = set(term.split())
    return [g for g in genre_vocab if words <= set(g.lower().split())]


def _extract_genres(text: str, genre_vocab: Sequence[str], rejecting: bool) -> Tuple[List[str], str]:
    """Return (matched catalog genres, text without them)."""
    exact = {g.lower(): g for g in genre_vocab}
    found: List[str] = []
    for term in _genre_terms(genre_vocab):
        pattern = re.compile(rf"\b{re.escape(term)}\b")
        if not pattern.search(text):
            continue
        text = pattern.sub(" ", text)
        # Asking for "pop" means pop; rejecting "pop" also rules out indie pop and latin pop.
        matched = [exact[term]] if term in exact and not rejecting else _containing(term, genre_vocab)
        found += [g for g in matched if g not in found]
    return found, text


def _apply_numeric(profile: Profile, text: str, rejected: bool) -> None:
    for attr, pattern, wanted, rejected_target in _NUMERIC_LEXICON:
        if attr not in profile.explicit_numeric and pattern.search(text):
            profile.targets[attr] = rejected_target if rejected else wanted
            profile.explicit_numeric.add(attr)
            if rejected:
                profile.rejected_numeric.add(attr)


def parse_explicit(text: str, genre_vocab: Sequence[str], mood_vocab: Sequence[str]) -> Profile:
    """Pull catalog genres/moods and numeric hints straight out of the text."""
    wanted, rejected = _split_negated(_normalise(text))

    profile = Profile()
    profile.avoid_genres, rejected = _extract_genres(rejected, genre_vocab, rejecting=True)
    genres, wanted = _extract_genres(wanted, genre_vocab, rejecting=False)
    profile.genres = [g for g in genres if g not in profile.avoid_genres]

    def moods_in(segment: str) -> List[str]:
        return [m for m in mood_vocab if re.search(rf"\b{re.escape(m.lower())}\b", segment)]

    profile.avoid_moods = moods_in(rejected)
    profile.moods = [m for m in moods_in(wanted) if m not in profile.avoid_moods]

    # What the user asked for outranks what they ruled out ("calm, not loud" is 0.25).
    _apply_numeric(profile, wanted, rejected=False)
    _apply_numeric(profile, rejected, rejected=True)
    return profile


def _range_gap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(a[0], b[0]) - min(a[1], b[1])


def _clashing_features(first: Card, second: Card, explicit: Set[str]) -> List[str]:
    return [
        attr for attr in NUMERIC_ATTRS
        if attr not in explicit and attr in first.ranges and attr in second.ranges
        and _range_gap(first.ranges[attr], second.ranges[attr]) >= BLEND_CONFLICT_GAP
    ]


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
    if len(chosen) > 1:
        first, second = chosen[0][0], chosen[1][0]
        clash = _clashing_features(first, second, profile.explicit_numeric)
        if clash:
            # Don't merge opposites into a range everything fits: follow the stronger
            # match and let the verifier tell the user about the other one.
            chosen = chosen[:1]
            profile.context_clash = (
                f"'{first.title}' and '{second.title}' call for opposite {', '.join(clash)}, "
                f"so I followed '{first.title}'. Tell me if you meant the other one"
            )
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
