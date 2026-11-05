# AI Interactions Log

## Agentic Workflow Enhancement (stretch): reasoning traces

VibeFinder's agent records every intermediate step (guardrail → retrieve → plan → act → diversify → verify → revise → decide). The complete set of traces for all evaluation cases is committed at **[`logs/agent_traces.md`](logs/agent_traces.md)**. Regenerate it with:

```bash
python -m src.evaluate --traces logs/agent_traces.md
```

### Trace 1: multi-step revision on a contradictory request

The agent's first plan produces a conflict. It verifies, revises its own plan, re-acts, and re-verifies before deciding to hand the result to a human.

```
Request: "lofi at maximum energy, not acoustic"
1. guardrail: input accepted
2. retrieve: no context card matched
3. plan: genre=lofi, energy=0.92, acousticness=0.10
4. act: iteration 1: ranked 18 songs; top = Storm Runner (2.97/4.50)
5. verify: fit=0.66 evidence=1.00 confidence=0.66; issues: conflict
6. revise: no lofi track is close to energy 0.92, acousticness 0.10 -> genre weight x0.5, re-ranking
7. act: iteration 2: ranked 18 songs; top = Storm Runner (2.97/3.75)
8. verify: fit=0.79 evidence=1.00 confidence=0.64; issues: none
9. decide: confidence=0.64, needs_review=True
```

### Trace 2: retrieval-driven plan, no revision needed

```
Request: "I need something calm for studying late at night"
1. guardrail: input accepted
2. retrieve: late-night-study (0.49), night-drive (0.12)
3. plan: context genres~lofi/ambient/ambient electronic, context moods~focused/chill, energy=0.25, danceability=0.30-0.62, acousticness=0.65-0.95
4. act: iteration 1: ranked 18 songs; top = Quantum Lullaby (5.72/5.75)
5. diversify: diversity re-rank changed the top-5; reordered only
6. verify: fit=0.99 evidence=1.00 confidence=0.99; issues: none
7. decide: confidence=0.99, needs_review=False
```

### Trace 3: decision to refuse

```
Request: "asdfghjkl qwerty"
1. guardrail: input accepted
2. retrieve: no context card matched
3. plan: (no preferences detected)
4. decide: no usable preferences found; refusing to guess
```

## RAG Enhancement (stretch): before/after

Retrieval uses a custom knowledge base (`data/knowledge/`, 13 context cards) alongside the song catalog. Here is the same scorer with retrieval switched off and on (`python -m src.evaluate`):

```
Retrieval lift (requests that describe a situation, not a genre)
  without retrieval (baseline): 2/7 correct top result
  with retrieval (RAG)        : 7/7 correct top result
```

For example, "pump me up for the gym" gives the baseline nothing to score (`no_signal`, no recommendations). With retrieval, the `gym-workout` card supplies energy 0.85–1.00 and low acousticness, and *Gym Hero* ranks first (see `tests/test_agent.py::test_retrieval_actually_changes_the_ranking`).

## AI-assisted development (how the build itself used an agent)

**Task given to the agent:** extend the Module 3 Music Recommender into a Project 4 applied AI system meeting the course rubric (RAG + agentic loop + reliability testing + docs), keeping the system runnable offline.

**What the agent generated or changed:** the base-project history merge; scorer fixes (`src/recommender.py`); the retrieval layer, knowledge cards, planner, guardrails, agent loop, explainer and CLI; the tests (53 at first, 132 now) and the evaluation harness; this documentation. Each milestone is a separate commit.

**What was verified or fixed by review:** the agent's first test suite passing 17/17 was not accepted as proof. Probing with untuned inputs found the negation bug ("not jazz" → jazz at confidence 1.00) and the "work out" retrieval bug, and both were fixed with regression tests. A cross-process nondeterminism in an output message was caught by comparing two terminal runs, then locked down with a `PYTHONHASHSEED` test. An AI code review of the finished branch then found ten more defects, including negation phrasings the first fix missed; all ten were fixed with regression tests. All README outputs were copied from real runs.
