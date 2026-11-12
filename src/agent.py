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

from .gemini import GeminiUnavailable
from .guardrails import sanitize_query
from .llm import Explainer, TemplateExplainer
from .profile import Profile, apply_context, build_profile, parse_explicit, strip_negated
from .recommender import (
    NUMERIC_ATTRS, genre_affinity, load_songs, max_score, numeric_distance, rank_songs,
)
from .retrieval import Card, load_cards

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
    explained_by: str = "template"       # "gemini" or "template"
    cited_card: Optional[str] = None


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
    retrieval_mode: str = ""
    parser: str = "rules"                  # "gemini" or "rules"
    support: Optional[str] = None          # crisis-support pointer when distress is detected
    intent: Optional[Dict] = None          # the validated Gemini intent, when used
    disagreements: List[str] = field(default_factory=list)

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
        retrieval_mode: Optional[str] = None,
        retriever=None,
        understander="auto",
    ):
        self.songs = songs if songs is not None else load_songs()
        cards = cards if cards is not None else load_cards()
        self.cards = cards if use_retrieval else []
        self.retrieval_notes: List[str] = []
        if retriever is not None:
            self.retriever = retriever if use_retrieval else None
        elif self.cards:
            from .embeddings import make_retriever   # imported late: pulls in the Gemini client
            self.retriever, self.retrieval_notes = make_retriever(self.cards, mode=retrieval_mode)
        else:
            self.retriever = None
        self.explainer = explainer or TemplateExplainer()
        self.k = k
        self.genre_vocab = sorted({s["genre"] for s in self.songs})
        self.mood_vocab = sorted({s["mood"] for s in self.songs})
        if understander == "auto":
            from .gemini import shared_client
            from .understanding import GeminiUnderstander
            client = shared_client()
            understander = (GeminiUnderstander(client, self.genre_vocab, self.mood_vocab)
                            if client.available else None)
        self.understander = understander

    # ------------------------------------------------------------------ run
    def run(self, text) -> AgentResult:
        """Guardrail -> retrieve + plan -> act/verify/revise -> decide. Never raises."""
        result = AgentResult(query=text if isinstance(text, str) else repr(text))

        cleaned = self._guard(text, result)
        if cleaned is None:
            return result

        retrieved, profile = self._plan(cleaned, result)
        result._profile = profile       # not a dataclass field: kept out of --json output
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

    def refine(self, profile: Profile, label: str, exclude_ids=()) -> AgentResult:
        """Re-rank from an edited profile (multi-turn refinement, see session.py).

        Skips guardrail/understanding/retrieval: the profile already carries the earlier
        turn's context cards. Act, verify, revise and decide run exactly as in run()."""
        result = AgentResult(query=label, cleaned_query=label, parser="refine")
        result.retrieved = [(c.id, round(s, 3)) for c, s in profile.cards]
        _step(result, "refine", label)
        _step(result, "plan", profile.describe())
        result._profile = profile
        if not profile.has_signal:
            return self._stop(result, profile, "no_signal", "refinement left no preferences",
                              "That leaves me nothing to go on. Try describing what you want to hear.")
        excluded = set(exclude_ids)
        candidates = [s for s in self.songs if profile.allows(s) and s["id"] not in excluded]
        if len(candidates) < len(self.songs):
            _step(result, "filter", f"removed {len(self.songs) - len(candidates)} rejected or already-dismissed songs")
        if not candidates:
            return self._stop(result, profile, "no_candidates", "no songs left to recommend",
                              "Nothing in the catalog is left after those changes.")
        attempt = self._act_verify_revise(profile, candidates, profile.cards, result)
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
        result.support = cleaned.support
        _step(result, "guardrail", "input accepted"
              + (f" with warnings: {'; '.join(cleaned.warnings)}" if cleaned.warnings else ""))
        return cleaned.text

    def _understand(self, text: str, result: AgentResult):
        """2a. Gemini reads the request; on any failure the rule-based parser takes over."""
        if self.understander is None:
            return None
        if any("instruction-override" in w for w in result.warnings):
            # Defence in depth: text that already looks like an attack never reaches the LLM.
            _step(result, "understand", "rule-based parser (input flagged by guardrail; not sent to Gemini)")
            return None
        try:
            intent = self.understander.parse(text)
        except GeminiUnavailable as exc:
            _step(result, "understand", f"rule-based parser (fallback: {exc.reason})")
            return None
        result.parser = "gemini"
        result.intent = {k: v for k, v in vars(intent).items()}
        _step(result, "understand", "gemini: " + intent.describe()
              + (f"; validator dropped {', '.join(intent.dropped)}" if intent.dropped else ""))
        return intent

    def _plan(self, text: str, result: AgentResult) -> Tuple[List[Tuple[Card, float]], Profile]:
        """2. Understand the request, retrieve context cards, build the profile."""
        intent = self._understand(text, result)
        if intent is not None and not intent.on_topic:
            _step(result, "plan", "gemini judged the request off-topic; not retrieving")
            return [], Profile()
        if intent is not None and intent.english_paraphrase:
            retrieval_text = intent.english_paraphrase   # rejected things already left out
        else:
            retrieval_text = strip_negated(text, self.genre_vocab, self.mood_vocab)
        retrieved = (
            self.retriever.retrieve(retrieval_text, k=3, min_score=RETRIEVAL_MIN_SCORE)
            if self.retriever else []
        )
        result.retrieved = [(c.id, round(s, 3)) for c, s in retrieved]
        _step(result, "retrieve", ", ".join(f"{c.id} ({s:.2f})" for c, s in retrieved)
              or "no context card matched")
        info = dict(getattr(self.retriever, "last_info", None) or {})
        if info:
            result.retrieval_mode = info.get("mode", "")
            why = info.get("fallback") or "; ".join(self.retrieval_notes)
            _step(result, "route", f"retrieval={info.get('mode')}"
                  + (f" (fallback: {why})" if why and info.get("fallback") else ""))
        if intent is None:
            profile = build_profile(text, retrieved, self.genre_vocab, self.mood_vocab)
        else:
            from .understanding import disagreements, profile_from_intent
            explicit = profile_from_intent(intent, self.genre_vocab)
            diffs = disagreements(parse_explicit(text, self.genre_vocab, self.mood_vocab), explicit)
            if diffs:
                result.disagreements = diffs
                _step(result, "cross-check", "rule parser read it differently: " + "; ".join(diffs))
                _log_disagreement(text, diffs)
            profile = apply_context(explicit, retrieved)
        _step(result, "plan", profile.describe())
        return retrieved, profile

    @staticmethod
    def _stop(result: AgentResult, profile: Profile, issue: str, why: str, message: str) -> AgentResult:
        """Give up honestly: no recommendations, flagged for a human, with a reason."""
        _step(result, "decide", why)
        result.profile = profile.describe()
        result.needs_review = True
        if result.support:
            message = result.support + " " + message
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
        result.needs_review = bool(result.relaxations) or bool(attempt.issues) or bool(result.support)

        chosen_cards = [c for c, _ in attempt.profile.cards]
        cards = [self._card_for(song, chosen_cards) for song, _, _ in attempt.picks]
        batch = getattr(self.explainer, "explain_batch", None)
        explained = (batch([(song, reasons, card) for (song, _, reasons), card in zip(attempt.picks, cards)])
                     if batch else None)
        for rank, ((song, score, reasons), card) in enumerate(zip(attempt.picks, cards), start=1):
            if explained:
                text, source, cited = explained[rank - 1]
            else:
                text, source, cited = self.explainer.explain(song, reasons, card), "template", None
            result.recommendations.append(Recommendation(
                rank=rank, song=song, score=round(score, 2),
                fit=round(max(0.0, min(1.0, score / attempt.ceiling)), 2), reasons=reasons,
                explanation=text, explained_by=source, cited_card=cited,
            ))
        if explained:
            n = sum(src == "gemini" for _, src, _ in explained)
            _step(result, "explain", f"gemini wrote {n}/{len(explained)} explanations"
                  + (f"; {len(explained) - n} fell back to the template" if n < len(explained) else ""))

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


def _log_disagreement(text: str, diffs: List[str]) -> None:
    """Append to logs/parser_disagreements.jsonl: a running record of where Gemini and the
    rules differ, for spotting regressions in either."""
    import json
    from .log_setup import LOG_DIR
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "parser_disagreements.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"query": text, "diffs": diffs}, ensure_ascii=False) + "\n")
    except OSError:
        pass


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
