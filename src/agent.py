"""
VibeFinder agent: plan -> retrieve -> act -> verify -> revise -> decide.

The agent does not just rank songs. After ranking it checks its own answer and can
change its plan:

  verify   computes a confidence score and looks for problems
             conflict       an explicit genre has no song near the requested feel
             unsatisfiable  no song in the catalog meets all numeric targets together
             mixed_context  two matched context cards contradicted each other
             exclusion_only the user said what to avoid but not what they want
             low_confidence confidence is below CONFIDENCE_MIN
  revise   for a `conflict`, relaxes the genre weight so the numeric request wins
  decide   anything the agent had to relax, or could not satisfy, is flagged for a
           human to confirm (human-in-the-loop) instead of being passed off as certain

Every step is recorded in `AgentResult.trace` and in the log file.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .guardrails import sanitize_query
from .llm import Explainer, TemplateExplainer
from .profile import Profile, build_profile, strip_negated
from .recommender import (
    NUMERIC_ATTRS, genre_affinity, load_songs, max_score, numeric_distance, rank_songs,
)
from .retrieval import Card, Retriever, load_cards

log = logging.getLogger("vibefinder.agent")

CONFIDENCE_MIN = 0.60        # below this the answer is flagged for human review
RETRIEVAL_MIN_SCORE = 0.12   # cosine floor for a context card to count (see eval notes)
EVIDENCE_REF = 0.30          # retrieval score treated as "full evidence" for a vague request
UNSAT_TOLERANCE = 0.25       # a target is unreachable if every song is farther than this
CONFLICT_TOLERANCE = 0.25
RELAXATION_PENALTY = 0.15    # confidence cost for overriding something the user asked for
RELAXED_GENRE_SCALE = 0.5
DIVERSITY_PENALTY = 0.4      # per already-picked song with the same genre or artist
MAX_ITERATIONS = 2           # the first plan, plus at most one revision


@dataclass
class Recommendation:
    rank: int
    song: Dict
    score: float
    fit: float
    reasons: List[str]
    explanation: str


@dataclass
class AgentResult:
    query: str
    cleaned_query: str = ""
    ok: bool = True
    profile: str = ""
    retrieved: List[Tuple[str, float]] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    confidence: float = 0.0
    issues: List[str] = field(default_factory=list)
    relaxations: List[str] = field(default_factory=list)
    needs_review: bool = False
    message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    trace: List[Dict] = field(default_factory=list)

    @property
    def top_titles(self) -> List[str]:
        return [r.song["title"] for r in self.recommendations]


@dataclass
class _Attempt:
    """The final pass of the act/verify/revise loop."""
    profile: Profile
    picks: List[Tuple[Dict, float, List[str]]]
    ceiling: float
    confidence: float
    issues: List[Tuple[str, str]]


def _step(result: AgentResult, action: str, detail: str) -> None:
    """Record one step in the result's trace and in the log file."""
    result.trace.append({"step": len(result.trace) + 1, "action": action, "detail": detail})
    log.info("%s: %s", action, detail)


class VibeFinderAgent:
    def __init__(
        self,
        songs: Optional[List[Dict]] = None,
        cards: Optional[List[Card]] = None,
        explainer: Optional[Explainer] = None,
        k: int = 5,
        use_retrieval: bool = True,
    ):
        self.songs = songs if songs is not None else load_songs()
        cards = cards if cards is not None else load_cards()
        self.cards = cards if use_retrieval else []
        self.retriever = Retriever(self.cards) if self.cards else None
        self.explainer = explainer or TemplateExplainer()
        self.k = k
        self.genre_vocab = sorted({s["genre"] for s in self.songs})
        self.mood_vocab = sorted({s["mood"] for s in self.songs})

    # ------------------------------------------------------------------ run
    def run(self, text) -> AgentResult:
        """Guardrail -> retrieve + plan -> act/verify/revise -> decide. Never raises."""
        result = AgentResult(query=text if isinstance(text, str) else repr(text))

        cleaned = self._guard(text, result)
        if cleaned is None:
            return result

        retrieved, profile = self._plan(cleaned, result)
        if not profile.has_signal:
            return self._stop(result, profile, "no_signal", "no usable preferences found; refusing to guess",
                              "I couldn't tell what you're after, so I won't guess. Try adding an "
                              "activity (studying, workout), a mood (chill, happy) or a genre.")

        candidates = [s for s in self.songs if profile.allows(s)]
        if len(candidates) < len(self.songs):
            _step(result, "filter", f"removed {len(self.songs) - len(candidates)} rejected songs")
        if not candidates:
            return self._stop(result, profile, "no_candidates", "no songs left to recommend",
                              "Everything in the catalog is ruled out by what you asked me to avoid."
                              if self.songs else "The song catalog is empty, so I have nothing to recommend.")

        attempt = self._act_verify_revise(profile, candidates, retrieved, result)
        self._decide(attempt, result)
        return result

    # ----------------------------------------------------------------- stages
    @staticmethod
    def _guard(text, result: AgentResult) -> Optional[str]:
        """1. Guardrail: the cleaned text, or None if the input can't be used."""
        cleaned = sanitize_query(text)
        result.warnings = list(cleaned.warnings)
        if not cleaned.ok:
            _step(result, "guardrail", f"rejected input: {'; '.join(cleaned.warnings)}")
            result.ok = False
            result.needs_review = True
            result.issues = ["invalid_input"]
            result.message = ("I couldn't read that request. Describe what you want to listen to, "
                              "e.g. 'something calm for studying' or 'upbeat pop for a workout'.")
            return None
        result.cleaned_query = cleaned.text
        _step(result, "guardrail", "input accepted"
              + (f" with warnings: {'; '.join(cleaned.warnings)}" if cleaned.warnings else ""))
        return cleaned.text

    def _plan(self, text: str, result: AgentResult) -> Tuple[List[Tuple[Card, float]], Profile]:
        """2. Retrieve context cards (rejected phrases removed first) and build the profile."""
        retrieval_text = strip_negated(text, self.genre_vocab, self.mood_vocab)
        retrieved = (
            self.retriever.retrieve(retrieval_text, k=3, min_score=RETRIEVAL_MIN_SCORE)
            if self.retriever else []
        )
        result.retrieved = [(c.id, round(s, 3)) for c, s in retrieved]
        _step(result, "retrieve", ", ".join(f"{c.id} ({s:.2f})" for c, s in retrieved)
              or "no context card matched")
        profile = build_profile(text, retrieved, self.genre_vocab, self.mood_vocab)
        _step(result, "plan", profile.describe())
        return retrieved, profile

    @staticmethod
    def _stop(result: AgentResult, profile: Profile, issue: str, why: str, message: str) -> AgentResult:
        """Give up honestly: no recommendations, flagged for a human, with a reason."""
        _step(result, "decide", why)
        result.profile = profile.describe()
        result.needs_review = True
        result.issues = [issue]
        result.message = message
        return result

    def _act_verify_revise(self, profile: Profile, candidates: List[Dict],
                           retrieved: List[Tuple[Card, float]], result: AgentResult) -> "_Attempt":
        """3. Rank, check the answer, and on a genre/feel conflict relax the genre once."""
        top_card_score = retrieved[0][1] if retrieved else 0.0
        for iteration in range(1, MAX_ITERATIONS + 1):
            prefs = profile.to_prefs()
            ranked = rank_songs(prefs, candidates)
            picks, moved = self._diversify(ranked)
            ceiling = max_score(prefs) or 1.0
            _step(result, "act", f"iteration {iteration}: ranked {len(ranked)} songs; top = "
                                 f"{picks[0][0]['title']} ({picks[0][1]:.2f}/{ceiling:.2f})")
            if moved:
                _step(result, "diversify", moved)

            fit = max(0.0, min(1.0, picks[0][1] / ceiling))
            evidence = 1.0 if profile.has_explicit_signal else min(1.0, top_card_score / EVIDENCE_REF)
            penalty = RELAXATION_PENALTY if result.relaxations else 0.0
            confidence = max(0.0, fit * evidence - penalty)
            issues = self._verify(profile, candidates)
            if confidence < CONFIDENCE_MIN:
                issues.append(("low_confidence",
                               f"confidence {_shown(confidence):.2f} is below {CONFIDENCE_MIN:.2f}"))
            _step(result, "verify", f"fit={fit:.2f} evidence={evidence:.2f} "
                                    f"confidence={_shown(confidence):.2f}; issues: "
                                    + (", ".join(code for code, _ in issues) or "none"))

            conflict = next((text for code, text in issues if code == "conflict"), None)
            if conflict and iteration < MAX_ITERATIONS:
                profile = profile.copy()
                profile.genre_scale = RELAXED_GENRE_SCALE
                result.relaxations.append(conflict)
                _step(result, "revise", f"{conflict} -> genre weight x{RELAXED_GENRE_SCALE:g}, re-ranking")
                continue
            break
        return _Attempt(profile, picks, ceiling, confidence, issues)

    def _decide(self, attempt: "_Attempt", result: AgentResult) -> None:
        """4. Fill in the answer, explain each pick, and tell the user what to double-check."""
        result.profile = attempt.profile.describe()
        result.confidence = _shown(attempt.confidence)
        result.issues = [code for code, _ in attempt.issues]
        issue_text = dict(attempt.issues)
        result.needs_review = bool(result.relaxations) or bool(attempt.issues)

        chosen_cards = [c for c, _ in attempt.profile.cards]
        for rank, (song, score, reasons) in enumerate(attempt.picks, start=1):
            result.recommendations.append(Recommendation(
                rank=rank, song=song, score=round(score, 2),
                fit=round(max(0.0, min(1.0, score / attempt.ceiling)), 2), reasons=reasons,
                explanation=self.explainer.explain(song, reasons, self._card_for(song, chosen_cards)),
            ))

        notes = [f"I relaxed a preference you gave: {relaxation}. Tell me if the original "
                 f"genre matters more than the feel you described." for relaxation in result.relaxations]
        notes += [issue_text[code] for code in ("mixed_context", "exclusion_only", "unsatisfiable")
                  if code in issue_text]
        if "low_confidence" in issue_text:
            notes.append(issue_text["low_confidence"] + "; a more specific request will help.")
        result.message = " ".join(notes) or None
        _step(result, "decide", f"confidence={result.confidence:.2f}, needs_review={result.needs_review}")

    # ------------------------------------------------------------- helpers
    def _diversify(self, ranked) -> Tuple[List, Optional[str]]:
        """Greedy re-rank so one genre/artist can't fill the whole list."""
        picks: List = []
        remaining = list(ranked)
        while remaining and len(picks) < self.k:
            def adjusted(item):
                song = item[0]
                repeats = sum(
                    (p[0]["genre"] == song["genre"]) + (p[0]["artist"] == song["artist"]) for p in picks
                )
                return item[1] - DIVERSITY_PENALTY * repeats
            best = max(remaining, key=adjusted)   # ties keep rank order (max returns first)
            picks.append(best)
            remaining.remove(best)

        plain = [r[0]["id"] for r in ranked[: self.k]]
        final = [p[0]["id"] for p in picks]
        if plain == final:
            return picks, None
        moved_in = [p[0]["title"] for p in picks if p[0]["id"] not in plain]
        return picks, ("diversity re-rank changed the top-%d" % self.k
                       + (f"; brought in {', '.join(moved_in)}" if moved_in else "; reordered only"))

    def _verify(self, profile: Profile, candidates: List[Dict]) -> List[Tuple[str, str]]:
        issues: List[Tuple[str, str]] = []
        if profile.context_clash:
            issues.append(("mixed_context", profile.context_clash + "."))
        if not profile.has_preference:
            issues.append(("exclusion_only",
                           "You told me what to avoid but not what you want, so these are "
                           "simply tracks that pass your exclusions."))
        targets = {a: t for a, t in profile.targets.items()}

        # conflict: explicit genre vs explicit numeric request
        # NUMERIC_ATTRS order (not set order) keeps messages identical across runs
        explicit_numeric = {a: targets[a] for a in NUMERIC_ATTRS
                            if a in profile.explicit_numeric and a in targets}
        if profile.genres and explicit_numeric and profile.genre_scale == 1.0:
            in_genre = [s for s in candidates if any(genre_affinity(g, s["genre"]) for g in profile.genres)]
            if in_genre:
                best = min(
                    max(numeric_distance(s[a], t) for a, t in explicit_numeric.items()) for s in in_genre
                )
                if best > CONFLICT_TOLERANCE:
                    feel = ", ".join(f"{a} {_fmt(t)}" for a, t in explicit_numeric.items())
                    issues.append(("conflict",
                                   f"no {'/'.join(profile.genres)} track is close to {feel}"))

        # unsatisfiable: no song in the whole catalog meets all numeric targets together
        if targets:
            best_all = min(
                max(numeric_distance(s[a], t) for a, t in targets.items()) for s in candidates
            )
            if best_all > UNSAT_TOLERANCE:
                issues.append(("unsatisfiable",
                               f"No song in the {len(candidates)}-track catalog meets all of those "
                               f"targets at once (closest miss: {best_all:.2f}); these are the nearest options"))
        return issues

    @staticmethod
    def _card_for(song: Dict, cards: List[Card]) -> Optional[Card]:
        """Prefer a retrieved card that actually lists this song's genre/mood."""
        for card in cards:
            if song["genre"] in card.genres or song["mood"] in card.moods:
                return card
        return cards[0] if cards else None


def _shown(confidence: float) -> float:
    """Confidence as reported: rounded to 2 decimals, but always on the same side of
    CONFIDENCE_MIN as the raw value the decision used.

    Plain rounding let a 0.5995 that failed the 0.60 check print as "confidence 0.60 is
    below 0.60". Only values in [0.595, 0.60) are affected; they are shown as 0.59."""
    shown = round(confidence, 2)
    if confidence < CONFIDENCE_MIN <= shown:
        shown = math.floor(CONFIDENCE_MIN * 100 - 1) / 100
    return shown


def _fmt(target) -> str:
    return f"{target[0]:.2f}-{target[1]:.2f}" if isinstance(target, tuple) else f"{float(target):.2f}"
