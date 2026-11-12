# VibeFinder 3.0: a Gemini-powered, retrieval-grounded, self-checking music recommender

> **CodePath AI110, Project 4: Applied AI System**
> **Base project:** *Music Recommender Simulation* (Module 3, [`ai110-module3show-musicrecommendersimulation-starter`](https://github.com/hassan47khan/ai110-module3show-musicrecommendersimulation-starter)), included in this repo's history. VibeFinder 2.0 (the first submission) is tagged in history at commit `2a45781`.

Describe a moment ("feeling bummed out on a rainy Sunday", "música para una fiesta en la playa", "pump me up for the gym but no pop"). VibeFinder picks five real tracks from a 580-song catalog, plays their 30-second previews, tells you **how sure it is**, and **flags its own answer for review** when it had to bend your request or couldn't satisfy it. Follow-ups such as "more upbeat", "not #3" or "more like #2" refine the last answer instead of starting over.

## What changed from 2.0 to 3.0

| | VibeFinder 2.0 | VibeFinder 3.0 |
|---|---|---|
| **Catalog** | 18 invented songs | **580 real tracks** from a BSD-licensed Spotify snapshot ([provenance](data/CATALOG_PROVENANCE.md)) |
| **Knowledge base** | 13 context cards | **30 cards**, ranges calibrated to the real catalog |
| **Query understanding** | Regex rules only | **Gemini, schema-constrained to the catalog's vocabulary**; rules as fallback and as a cross-check |
| **Retrieval** | TF-IDF | **Hybrid: BM25 + Gemini embeddings, RRF fusion, MMR diversity**, one calibrated evidence scale, optional Gemini rerank |
| **Explanations** | Template, or one Gemini call per song | **One batched call, each explanation validated** (title, length, cited card, no invented numbers) |
| **Conversation** | One-shot | **Multi-turn refinement** |
| **Interface** | CLI | **Streamlit app** with cover art and previews; the CLI still works |
| **Safety** | Override detection | Plus Unicode normalisation, a 26-attack **prompt-injection suite**, and **distress detection** with crisis-line info |
| **Evaluation** | 29 checks, 211 tests | 40 checks, **44-query labelled retrieval benchmark with an ablation table**, **384 tests**, **63-mutant mutation check** |

Everything degrades gracefully. Without a Gemini key, or when the free tier is out of quota, VibeFinder runs fully offline on the 2.0 path (rules, lexical retrieval, templates) and says so in the trace and the UI.

## Architecture

Full Mermaid diagram: [`diagrams/architecture.md`](diagrams/architecture.md).

```
request ─▶ follow-up? ─yes─▶ edit last profile ───────────────────────────────┐
             │no                                                              ▼
             ▼                                                     [act ▶ variety ▶ verify ▶ revise?]* ▶ explain ▶ output + review flag ▶ UI
         guardrails ─▶ Gemini intent (validated) ─▶ hybrid retrieval ─▶ profile ┘
                       └─ fallback: rule parser     └─ fallback: BM25
```

| Component | File | Role |
|---|---|---|
| Settings | `src/config.py` | Env and `.env` settings: paths, retrieval mode, model fallback chains |
| Gemini client | `src/gemini.py` | One door to Gemini: retry with backoff on 429/5xx, model fallback on 404, sha256 disk cache, token and latency accounting, `GeminiUnavailable` for clean fallback |
| Guardrails | `src/guardrails.py` | Rejects unusable input, NFKC and zero-width stripping, override detection (flagged text never reaches Gemini), distress detection |
| Understanding | `src/understanding.py` | Gemini reads the request into an `Intent` whose genre and mood fields are enums of the catalog. The validator drops unknown values, clamps ranges and strips instruction-like paraphrases |
| Retrieval | `src/retrieval.py`, `src/embeddings.py`, `src/rerank.py` | TF-IDF (2.0), BM25 with a body-only discount, dense retrieval over a committed `gemini-embedding-2` index, and a hybrid of the two with RRF and MMR. Every arm is calibrated onto the agent's evidence scale from held-out queries |
| Planner and scorer | `src/profile.py`, `src/recommender.py` | Intent (or rules) plus card genres, moods and feature ranges become a scoring profile |
| Agent | `src/agent.py` | Plan, retrieve, act, verify, revise, decide; confidence; human-review flag; full trace |
| Explanations | `src/llm.py` | Batched, grounded Gemini explanations, each validated; templates as fallback |
| Conversation | `src/session.py` | Deterministic follow-up parser and profile editing |
| UI | `app/`, `src/enrich.py` | Streamlit app; Deezer cover art and previews (no key needed, never used by the core) |
| Evaluation | `src/evaluate.py`, `src/retrieval_eval.py`, `data/eval/`, `tests/`, `tools/mutate.py` | Behaviour cases, retrieval ablation, injection suite, unit tests, mutation check |

## Setup

Requires Python 3.10+ (developed on 3.14).

```bash
git clone https://github.com/hassan47khan/applied-ai-system-project.git
cd applied-ai-system-project
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Gemini (optional, free).** Create a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), copy `.env.example` to `.env` and paste the key in. Free-tier limits are per project and no longer published; check yours in AI Studio. Then build the card index once. It is committed, so this only needs re-running when a card changes:

```bash
python -m src.index build       # embeds the 30 cards (one API call)
python -m src.index calibrate   # fits the dense score scale on held-out queries
python -m src.index verify      # checks the index matches the cards
```

Run it:

```bash
streamlit run app/streamlit_app.py                          # the app
python -m src.main "I need something calm for studying late at night"
python -m src.main --interactive                            # follow-ups refine the last answer
python -m src.main --trace "lofi at maximum energy, not acoustic"
python -m pytest                                            # 384 tests, offline
python -m src.evaluate --injection                          # harness + ablation + injection suite
python tools/mutate.py                                      # do the tests notice 63 planted bugs?
python tools/build_catalog.py                               # rebuild data/songs.csv from the source dataset
```

## The app

- **Discover.** A request, or a follow-up to the last one. The top pick shows its cover art, a 30-second preview, its fit and an explanation; "Why this track" breaks down the score. Beside it are the agent's confidence, the review state with the agent's own message, the listening contexts it used, and which parser and retriever ran. **Where the picks land** plots the request's target ranges for energy, danceability and acousticness and shows where each pick falls. The other picks follow, then the full agent trace.
- **Compare.** The same request ranked with retrieval off and on, side by side.
- **Evaluation.** The retrieval ablation on the labelled benchmark, Hit@1 by request type, and the requests each retriever misses.

The interface is a true-black screen with colour only on data: cover art and one lime data hue. Its layout was checked at 1440, 768 and 390px wide.

## Sample interactions (offline path, real catalog)

Full output with traces: [`logs/sample_runs.txt`](logs/sample_runs.txt).

```
$ python -m src.main --k 3 "I need something calm for studying late at night"
  Retrieved context : late-night-study (0.33), night-drive (0.20), meditation (0.14)
  Confidence        : 1.00   Needs human review: no
#1  Views from Yamashita Park - Nathan Kawanishi  [lofi / focused]       fit 1.00
#2  One Summer Day - Joe Hisaishi                 [classical / focused]  fit 0.97
#3  Sky and Sand - Paul Kalkbrenner               [electronic / focused] fit 0.96
```

The 1.0 failure case, a contradictory request, is detected, revised and flagged:

```
$ python -m src.main --k 3 "lofi at maximum energy, not acoustic"
  Confidence        : 0.71   Needs human review: YES
#1  Highway to Hell - AC/DC                  [rock / intense]           fit 0.86
#2  Afterglow - Wilkinson                    [drum and bass / intense]  fit 0.84
  >> I relaxed a preference you gave: no lofi track is close to energy 0.92, acousticness 0.10. ...
```

A lone incidental word is flagged instead of trusted (BM25's body-only discount; this answered at 0.81 confidence before the fix):

```
$ python -m src.main --k 3 "good songs"
  Confidence        : 0.49   Needs human review: YES
  >> confidence 0.49 is below 0.60; a more specific request will help.
```

## Reliability and evaluation

`python -m src.evaluate --injection`, offline. Saved in [`logs/eval_results_offline.txt`](logs/eval_results_offline.txt):

```
Checks passed: 33/33  (7 skipped: need Gemini)

Retrieval lift (requests that describe a situation, not a genre)
  without retrieval (baseline): 2/9 correct top result
  with retrieval (RAG)        : 9/9 correct top result

Confidence calibration
  mean confidence, answers NOT flagged: 0.99 (n=18)
  mean confidence, answers flagged     : 0.30 (n=8)

Retrieval ablation (44 labelled queries, data/eval/retrieval_gold.json)
  retriever        R@3  Hit@1   MRR  nDCG@3  abstain  direct synonym  multi non-en
  tfidf           0.58   0.64  0.64    0.60     0.88    1.00    0.44   1.00   0.00
  bm25            0.64   0.64  0.66    0.64     0.88    1.00    0.44   1.00   0.00

Prompt-injection suite: 26/26 attacks handled safely
```

**How the benchmark avoids grading its own homework.**
- The 44 gold queries in [`data/eval/retrieval_gold.json`](data/eval/retrieval_gold.json) were labelled before any retriever ran on them. They cover direct requests, synonyms, multi-intent requests, non-English requests and off-topic requests.
- Score calibration uses a separate, disjoint set ([`calibration_queries.json`](data/eval/calibration_queries.json)), and a test enforces that the two sets don't overlap.
- The 7 Gemini-only behaviour cases (synonyms, other languages, several intents in one request) are reported as SKIP offline, not counted as passes.

**What the offline numbers show.** Lexical retrieval is perfect on direct requests and weak where vocabulary differs: 0.44 Hit@1 on synonyms and 0.00 on non-English requests. "Feeling bummed out" retrieves *pregame-hype* because the word "out" matches "going out". This gap is what the Gemini path is for.

**Results with Gemini: not run yet.** The dense, hybrid and hybrid+rerank rows, the 7 Gemini-only cases and the live injection run need a key and a built index. Run `python -m src.index build && python -m src.index calibrate && python -m src.evaluate --injection` and the output replaces this paragraph.

**Testing.**
- **384 pytest tests**, all offline. Gemini is replaced by a fake SDK whose embeddings put synonyms on the same axis, so the hybrid's semantic recall is tested without a network.
- **The injection suite runs against a fully hijacked fake model.** It returns SQL in genre fields, `energy: 9999` and "ignore previous instructions" as the paraphrase, and the system must still act only on catalog values within [0, 1].
- **`tools/mutate.py` plants 63 realistic bugs one at a time**, 16 of them in the 3.0 code. Examples: invented numbers allowed in explanations, Gemini retries switched off, edited cards not detected by the index, flagged input sent to Gemini. The suite must fail on every one.

**What went wrong along the way, and what I learned.**
- **The real catalog broke two assumptions.** Spotify's "chill" tag holds ODESZA and EDM remixes, so mapping it to lofi was wrong; only "study" is. My first mood rules also labelled every piano and ambient piece "melancholy", because instrumental music has low valence by nature. Checking instrumentalness first fixed it, and the provenance file documents both rules.
- **BM25 gave a one-word request full confidence.** "Good songs" matched the cooking card's body text "feel-good" at 0.81, because normalising by the query lets any single rare word look like a perfect match. Words a card is *about* (its keywords and title) now count for more than words its prose happens to use.
- **Lexical scores and embedding scores live on different scales.** Plugging a new retriever into thresholds tuned for TF-IDF would have silently changed what counts as evidence. Each retriever is now mapped onto one scale with a floor and a full-evidence point fitted on held-out queries.
- **Writing the injection suite found real gaps in the 2.0 guardrails.** Zero-width characters ("i​gnore previous instructions") and full-width letters walked straight past the override detector, and nothing responded to a self-harm statement.
- **Known limitations** (details in the [model card](model_card.md)):
  - Mood is derived from audio features, not labelled, so it is wrong for some songs.
  - The Spotify snapshot over-represents 2022-era English pop.
  - Follow-up parsing covers a fixed set of phrasings.
  - Confidence measures fit to your request, not whether you will like the songs.

## Design decisions and trade-offs

- **Gemini decides what you meant, never what gets ranked.** Its output is enum-constrained JSON, validated and cross-checked against the rules, and the agent's scoring, verification and confidence stay deterministic. Trade-off: the model can't express preferences outside the catalog's vocabulary.
- **The user's raw text reaches Gemini only for understanding and rerank, never for explanations.** The explanation prompt holds catalog facts and card text only, as in 2.0.
- **A committed index plus a disk cache.** The free tier's limits are unpublished and tight, so the knowledge base is embedded once at build time, and every query, rating and explanation is cached. A demo costs a handful of calls, and the evaluation is reproducible.
- **The offline path is a real product, not an error page.** All 384 tests run without a network; the UI and CLI say which path ran and why.
- **A real catalog, curated deterministically.** The catalog is popularity-first with at most 2 songs per artist, and the build script is committed. Trade-off: it inherits the source's biases (see provenance).
- **Follow-ups are parsed by rules, not by the model.** "Not #3" has one meaning, and a rule you can read is easier to trust than a model for it.
- **Refuse rather than guess, relax and ask rather than silently override**, and keep variety re-ranking. All three carry over from 2.0.

## Repository map

```
app/            Streamlit app (streamlit_app.py) and its presentation helpers (ui.py)
src/            agent, understanding, gemini client, retrieval (+ embeddings, rerank), profile,
                scoring, guardrails, llm, session, enrich, CLI, evaluate, index
data/           songs.csv (580 tracks) + CATALOG_PROVENANCE.md, knowledge/ (30 cards),
                index/ (embedding index + calibration), eval/ (gold set, calibration set, injection suite)
tests/          384 tests; fixtures/ keeps the 2.0 catalog and cards the older tests pin
tools/          build_catalog.py, mutate.py
diagrams/       architecture.md (Mermaid)
logs/           eval results, sample runs, traces
model_card.md   limitations, bias, misuse, testing, AI collaboration
```

## Reflection

The 2.0 lesson still holds: **a reliable AI system is mostly made of the parts that check the AI.** 3.0 added a second lesson. Putting an LLM in the loop doesn't remove the need for those checks; it moves them. Once Gemini could read the user's text, the parser had to become a validator, the guardrails had to anticipate obfuscated attacks, and every number in an explanation had to be traced back to a fact. The most valuable additions weren't the Gemini calls. They were the fallback that makes each call optional, and the benchmark that says whether each call earned its place.

## Portfolio note

*What this project says about me as an AI engineer:* I build AI systems that know their own limits, and I measure them. VibeFinder 3.0 uses an LLM where it beats rules (meaning, synonyms, other languages) and keeps deterministic code where trust matters (ranking, verification, confidence). It treats the model's output as untrusted, degrades cleanly when the model is unavailable, and backs each claim in this README with a command you can run.
