# Model Card: VibeFinder 3.0

> **CodePath AI110, Project 4: Applied AI System.**
> Extends VibeFinder 2.0, which extended the Module 3 *Music Recommender Simulation* starter.
> The 1.0 card is in history at commit `12213e3`; the 2.0 card at commit `2a45781`. This card
> is written for 3.0 and marks, inline, which findings date from an earlier version and whether
> they still hold.

**Contents**

1. [System overview and intended use](#1-system-overview-and-intended-use)
2. [System limitations and biases](#2-system-limitations-and-biases)
3. [Misuse potential and guardrails](#3-misuse-potential-and-guardrails)
4. [Reliability testing and evaluation](#4-reliability-testing-and-evaluation)
5. [AI collaboration reflection](#5-ai-collaboration-reflection)
6. [Version history](#6-version-history)

---

## 1. System overview and intended use

| | |
|---|---|
| **Name** | VibeFinder 3.0 |
| **Type** | Retrieval-augmented, content-based music recommender wrapped in a self-checking agent loop, with an LLM for query understanding and explanation |
| **Models** | Generation: `gemini-3.5-flash`, falling back to `gemini-3.5-flash-lite`, then `gemini-flash-latest`, then `gemini-2.5-flash`. Embeddings: `gemini-embedding-2` at 768 dimensions, falling back to `gemini-embedding-001`. **All are optional**: with no key, the system runs the fully offline 2.0 path |
| **Input** | Free text in any language, at most 300 characters, or a follow-up to the previous answer ("more upbeat", "not #3", "more like #2") |
| **Output** | Top-k tracks with grounded explanations, a confidence score in [0, 1], a `needs_review` flag with a plain-language message, and a step-by-step agent trace |
| **Catalog** | `data/songs.csv`: 580 real tracks from the BSD-licensed Hugging Face `maharshipandya/spotify-tracks-dataset`, curated deterministically by `tools/build_catalog.py` (see [`data/CATALOG_PROVENANCE.md`](data/CATALOG_PROVENANCE.md)) |
| **Knowledge base** | `data/knowledge/`: 30 hand-written listening-context cards, their feature ranges calibrated against the real catalog |
| **Third-party media** | Cover art and 30-second previews come from Deezer's public API at display time. No audio is stored or redistributed, and the core never calls it |

### What Gemini does, and does not do

| Stage | Gemini's role | Who makes the decision |
|---|---|---|
| Query understanding | Reads the request into an enum-constrained `Intent`: genres and moods drawn from the catalog's own vocabulary, feature ranges clamped to [0, 1], plus a short English paraphrase used for retrieval | Gemini proposes, a validator decides |
| Retrieval | Embeds the request for dense retrieval; optionally rates candidate context cards | Deterministic BM25 + RRF + MMR |
| Explanation | Writes the explanation for each pick | Validator keeps or discards each one independently |
| **Ranking** | **none** | Deterministic scorer, `src/recommender.py` |
| **Confidence** | **none** | Deterministic, `fit x evidence` |
| **Review flag** | **none** | Deterministic checks, `src/agent.py` |

The design rule is that the LLM may decide **what the user meant**, never **what gets ranked**
or **how sure the system claims to be**.

### Intended use

A portfolio and classroom demonstration of how to make an AI recommender **grounded** (it
retrieves before deciding), **self-checking** (it verifies and revises its own answer) and
**honest** (it flags low confidence instead of hiding it).

### Not intended for

- **Production listeners.** 580 tracks is a demo catalog, not a music service.
- **Any profiling of users.** Nothing about the user is stored, and no demographic is modelled.
- **Any setting where a recommendation is treated as authoritative.**
- **Any safety-critical use.** The distress pointer (§3.3) is a signpost, not a safety system.

Confidence measures how well the picks fit the **stated request**, not whether a person will
actually enjoy them. That distinction is the single most important limitation in this card.

---

## 2. System limitations and biases

### 2.1 Data and catalog bias

- **Popularity-first selection from a 2022-era Spotify snapshot** over-represents
  English-language, mainstream, Western releases. A request in Spanish or Japanese is now
  *understood* (Gemini reads it) but can only be answered from that skewed catalog.
- **Genres are Spotify's tags, not descriptions of sound.** The `jazz` tag includes show tunes
  such as *Alexander Hamilton*; the `chill` tag includes ODESZA and EDM remixes. An early
  version of the build script mapped `chill` to lofi and was wrong; only `study` maps that way
  now, and [`data/CATALOG_PROVENANCE.md`](data/CATALOG_PROVENANCE.md) records both rules.
- **Mood is inferred, never labelled by a person.** It is derived from valence, energy,
  acousticness and instrumentalness, so it is wrong for some songs. A first pass labelled every
  piano and ambient piece "melancholy", because instrumental music has low valence by nature;
  checking instrumentalness first fixed that, but the underlying inference is still a heuristic.
- **Some genres are thin.** With at most 2 songs per artist and 580 tracks across many tags,
  a narrow genre request can have only a handful of real candidates.
- **The knowledge base encodes my assumptions.** The 30 cards say what music suits an activity
  — that studying means low-energy instrumental music, that workouts mean high-energy pop and
  rock. **Someone who studies to metal or works out to jazz is systematically served worse.**
  The cards are written in English and their genre vocabulary is Western-centric. This is the
  most consequential bias in the system, because the cards set the scoring targets.

### 2.2 Model and inference limitations

- **Gemini quality varies with the model behind the alias**, which Google can change. Runs are
  cached by sha256 of the request, so a fixed cache reproduces exactly; a cold cache after a
  model change may not.
- **Model availability is not stable.** As measured on 2026-11-14, a newly issued free AI Studio
  key gets 404 on `gemini-flash-latest` and is told `gemini-2.5-flash` is "no longer available
  to new users". This is why the model chain has four entries and why `GeminiUnavailable`
  degrades instead of crashing.
- **Free-tier quota is unpublished and tight.** When it runs out, the system returns to the
  offline path. The trace and the UI say so, but the answers get measurably worse for synonyms
  and non-English requests — exactly the cases the LLM was added for.
- **The explanation validator catches shape, not truth.** It enforces that an explanation names
  its track, stays under 70 words, cites only its own card, and contains no number absent from
  its facts. It **cannot detect an invented qualitative claim**: a reply asserting "won a Grammy
  in 2019" contains no stray number and would survive. This is why the template explainer is
  the default and ranking never depends on generated text. See §4.5.
- **Confidence is a heuristic, not a probability.** It is `fit x evidence`, minus 0.15 when a
  preference was relaxed. A one-word request like "pop" can score near 1.00 because every pop
  song fits "pop" perfectly, which is not the same as the user being satisfied.

### 2.3 The confidence signal is only valid for the retriever it was tuned on

**This is the most serious limitation in the system, and it is a safety regression introduced
by an improvement.**

Confidence is `fit x evidence`, where `evidence` is a retrieval score. The `no_signal` and
`low_confidence` thresholds that drive the refusal guardrail were fitted when retrieval was
lexical — when finding nothing for nonsense *was* the behaviour. Dense retrieval finds
plausible context for nonsense, because a gibberish string still lands somewhere in embedding
space. The evidence term stays high, confidence never drops, and the guardrail never fires.

Measured by re-running the harness changing only `VIBEFINDER_RETRIEVAL`:

| Retriever | Behaviour harness | Injection suite | What it fails |
|---|---|---|---|
| `bm25` (lexical) | 30 / 40 | **26 / 26** | synonyms, non-English, multi-intent |
| `hybrid` (default) | 30 / 40 | 24 / 26 | **the refusal flags** |
| `dense` | **31 / 40** | 23 / 26 | **the refusal flags** |

The two failure sets are almost disjoint. Under `dense` and `hybrid`, six cases that pass
under `bm25` — `anything but jazz`, `not happy`, exclusion-only, a lone vague word and
**gibberish** — answer without raising `exclusion_only`, `low_confidence` or `no_signal`. The
injection score degrades monotonically as retrieval gets stronger, with `delimiter_escape`,
`schema_abuse` and then `bias_bait` each answering confidently instead of refusing.

**Consequences for a user.** On the Gemini path the system is measurably more willing to
produce a confident-looking answer to input it should decline — nonsense, contradictory
requests, and some prompt-injection attempts. The recommendations themselves are harmless;
what degrades is the system's honesty about not knowing, which is the property this project
exists to demonstrate.

**What it needs.** Recalibrate the evidence scale per retriever (it is fitted on 28 held-out
queries), and stop deriving abstention from an absolute retrieval score — a margin between the
top cards would not inherit the old retriever's scale. `hybrid` is kept as the default rather
than the better-ranking `dense` precisely because abstention matters more here than ranking.

**Caveat.** All three runs are mixed-path: 74 of ~148 Gemini calls hit `429
RESOURCE_EXHAUSTED`, so the parser fell back to rules for part of each run. The comparison
isolates *retrieval* cleanly, since that was the only variable changed, but does not isolate
the LLM parser's own contribution.

### 2.4 Known failure modes and edge cases

| Edge case | What happens | Status |
|---|---|---|
| Multi-intent request ("sad songs for a road trip") | A second card is blended only if it scores ≥ 60% of the best card, so the sadness is dropped | **Known failure.** Reported as a failing row in §4.3, not hidden |
| Opposite contexts ("sleep then gym") | Follows the stronger card and flags the other | By design, bluntly: "dinner with friends" is flagged too, though its picks are fine |
| Off-topic request made of known words ("the night of the concert was late") | Answered rather than refused | **Known failure.** Off-topic detection is word counting: refuse if fewer than half the words appear in the knowledge base |
| Follow-up phrased outside the rule set | Treated as a brand-new request | By design (§2.5), but silent — the user is not told the follow-up was not understood |
| Negation with a double negative | Not understood in general | Partly handled: "nothing less than loud" and "more or less acoustic" are recognised as idioms |
| "No jazz, rock or pop" | Read as a list of three rejections | Genuinely ambiguous; documented rather than guessed |
| Deezer has no match for a track | Placeholder shown, never a wrong cover (weak matches rejected) | 24 of 25 sampled tracks matched |

### 2.5 Limitations carried over from 2.0, and whether they still hold

| 2.0 limitation | Still true in 3.0? |
|---|---|
| Tiny catalog: 18 invented songs, "best" match often the *only* match | **Fixed.** 580 real tracks |
| Keyword retrieval has no synonyms; "bummed" won't match "sad" | **Fixed when Gemini is available** (dense retrieval + paraphrase). **Still true offline**: measured at 0.44 Hit@1 on synonyms, 0.00 on non-English (§4.2) |
| Non-English requests are refused | **Fixed when Gemini is available.** Still refused offline |
| The crude suffix stripper can merge unrelated words ("notes"/"not") | **Still true.** No current card is affected |
| Negation is a list of rule patterns, not language understanding | **Still true.** Covers genres, moods and features across many phrasings; a comma ends a rejection unless it continues a list closed by "or"/"and", and "and" always extends one, so "no jazz and calm" rejects calm too |
| Multi-intent requests collapse to one intent | **Still true** (§2.4) |
| Confidence is a heuristic, not a probability | **Still true** (§2.2) |

---

## 3. Misuse potential and guardrails

### 3.1 The safety surface changed in 3.0 — and the 2.0 card's main claim is now false

The 2.0 card said the user's text never reaches an LLM. **That is no longer true.** Since 3.0
the sanitised request is sent to Gemini for understanding and, optionally, rerank. It is still
never sent for explanations. Recording that reversal plainly is the point: the mitigations
exist because the guarantee was given up deliberately.

| # | Mitigation | Where |
|---|---|---|
| 1 | Input the guardrail flags as an override attempt is **never sent to Gemini** at all; it falls through to the rule parser | `src/guardrails.py` |
| 2 | Unicode NFKC normalisation, plus stripping of zero-width and bidirectional characters, **before** detection runs | `src/guardrails.py` |
| 3 | The request is delimited and the system instruction declares it data, not instructions | `src/understanding.py` |
| 4 | Output must match a JSON schema whose values are the catalog's **own enums**; a validator then drops unknown values, clamps numbers to [0, 1] and discards instruction-like paraphrases | `src/understanding.py` |
| 5 | Nothing Gemini writes as free text reaches the scorer. Explanations are shown only after per-pick validation, each falling back to the template on its own | `src/llm.py` |
| 6 | A 26-attack suite runs in the tests against a **fully hijacked** fake model, and live via `--injection` | `data/eval/injection_suite.json` |

Mitigation 2 exists because writing the suite found real gaps: zero-width characters inside
"i␣gnore previous instructions" and full-width letters both walked straight past the 2.0
detector.

### 3.2 Misuse scenarios and the safeguard for each

| Misuse | How it would work | Safeguard |
|---|---|---|
| **Prompt injection** | Hide instructions in the request to make the system emit attacker-chosen text, or inflate a feature to force a ranking | The six mitigations in §3.1. The schema is the real defence: even a fully hijacked model returning `energy: 9999`, SQL in a genre field and "ignore previous instructions" as the paraphrase cannot move the scorer, because every value is dropped unless it is a catalog enum in range |
| **Laundering claims through the explainer** | Get the model to assert something flattering and untrue about a track | Explanations are validated (names its track, ≤ 70 words, cites only its own card, no number absent from its facts) and fall back to the template. **Incomplete**: invented qualitative claims survive (§2.2) |
| **Over-trusting the output** | Treat a relaxed or impossible match as a confident recommendation | `needs_review`, the confidence score and a plain-language message saying what was bent. Confidence is deterministic, so the model cannot inflate it |
| **Profiling a user** | Infer demographics or identity from requests and score against that | Nothing about the user is stored and no demographics are modelled. If extended with listening history, the rule should stand: score against stated preferences, never inferred identity |
| **Resource abuse / cost** | Long inputs, large `k`, or a loop that calls the API repeatedly | Input capped at 300 characters, `k` capped at 10, agent loop capped at 3 iterations, every call cached on disk, retry with backoff rather than a tight loop |
| **Passing it off as professional advice** | Present the output as clinically or therapeutically meaningful | Section 1 states the non-intended uses; the UI shows confidence and the review flag beside every answer |

### 3.3 Sensitive input

A statement of self-harm or suicidal intent is **not refused**. The answer carries a short
pointer to the 988 Suicide & Crisis Lifeline (US) and findahelpline.com, and is flagged for
review.

**This is deliberately limited, and its limits matter.** Detection is keyword-based, so it will
miss indirect phrasing. It is a pointer to help, not a safety system, not a triage step, and not
a substitute for one. Nothing was added in 3.0 that tries to assess risk, because a keyword
match is not a basis for that.

### 3.4 What the guardrails do not cover

- **Refusal under strong retrieval** (§2.3). The guardrail that declines nonsense and some
  injection attempts is driven by a retrieval score and under-fires on the Gemini path. The
  schema and validation defences are unaffected; the *abstention* defence is.
- **Invented qualitative claims in explanations** (§2.2).
- **Indirectly phrased distress** (§3.3).
- **Bias in the cards themselves.** No guardrail notices that the study card assumes lofi. The
  only defence is that the cards are plain text in the repo, readable and editable.
- **Catalog bias.** Guardrails constrain the model, not the data it ranks.

---

## 4. Reliability testing and evaluation

Everything in this section is reproducible from the repo. Commands are in the
[README](README.md#reproducible-execution-evidence); saved output is in [`logs/`](logs/).

### 4.1 Headline results

| Evaluation | Command | Result |
|---|---|---|
| Unit and behaviour tests | `python -m pytest` | **426 / 426 pass**, fully offline |
| Mutation check | `python tools/mutate.py` | **67 / 67** planted bugs caught by the suite |
| Behaviour harness, offline | `python -m src.evaluate` | **33 / 33** checks pass, 7 Gemini-only cases reported as SKIP |
| Behaviour harness, with Gemini | same, with a key | **30 / 40** checks pass — see §2.3 |
| Prompt-injection suite, offline | `python -m src.evaluate --injection` | **26 / 26** attacks handled safely |
| Prompt-injection suite, with Gemini | same, with a key | **24 / 26** — see §2.3 |
| Retrieval benchmark | included above | 44 labelled queries, ablation across 5 retriever configurations (§4.2) |
| Index integrity | `python -m src.index verify` | 30 cards, `gemini-embedding-2` (768-d), up to date |

The test suite replaces the Gemini SDK with a fake whose embeddings put synonyms on the same
axis, so the hybrid's semantic recall is exercised without a network.

### 4.2 Retrieval benchmark and ablation

44 labelled queries in [`data/eval/retrieval_gold.json`](data/eval/retrieval_gold.json),
labelled before any retriever ran on them. Columns after `abstain` are Hit@1 by request type.
From [`logs/eval_results.txt`](logs/eval_results.txt):

```
  retriever        R@3  Hit@1   MRR  nDCG@3  abstain  direct synonym  multi non-en
  tfidf           0.58   0.64  0.64    0.60     0.88    1.00    0.44   1.00   0.00
  bm25            0.64   0.64  0.66    0.64     0.88    1.00    0.44   1.00   0.00
  dense           0.98   1.00  1.00    0.99     0.88    1.00    1.00   1.00   1.00
  hybrid          0.97   0.94  0.97    0.96     0.75    1.00    0.88   1.00   1.00
  hybrid+rerank   0.97   0.94  0.97    0.96     0.75    1.00    0.88   1.00   1.00
```

**What this shows.** Lexical retrieval is perfect on direct requests and useless where the
vocabulary differs: 0.44 Hit@1 on synonyms, **0.00 on non-English**. "Feeling bummed out"
retrieves *pregame-hype*, because the word "out" matches "going out". Closing that gap is the
entire justification for the Gemini path, and dense retrieval closes it completely (1.00 on
both).

**What it also shows, which I did not expect.** `dense` alone beats the configured `hybrid`
default on every ranking metric; fusing a strong retriever with a weak one made it worse, and
the optional rerank changed nothing measurable. The default is still `hybrid`, for the reason
in §2.3: better ranking came with worse abstention, and abstention is the property that
matters most here.

**Confidence calibration.** Offline, the separation is clean: answers not flagged average 0.99
(n=18), flagged answers average 0.30 (n=8). On the Gemini path it collapses to 0.93 (n=29) and
0.70 (n=5) — the same regression described in §2.3, visible as a calibration number.

### 4.3 Adversarial probing: untuned inputs, before and after fixes

These cases were produced by throwing requests at the system that it had **not** been tuned on.
Every row that says "Fail" is a bug this probing found; all of them are now regression tests.

| Input | Criterion | Before fixes | After |
|---|---|---|---|
| "relaxing music but not jazz" | no jazz in top 3 | **Fail**: jazz #1 at confidence 1.00 | Pass |
| "I hate rock, give me something to work out to" | workout songs, no rock | **Fail**: parsed genre=rock, retrieved *coding* card | Pass |
| "anything but jazz" | no jazz | **Fail**: jazz #1 at confidence 1.00 | Pass: flagged as exclusion-only |
| "not loud music" | no loud songs | **Fail**: high-energy pick at confidence 0.99 | Pass: energy capped at 0.60, flagged |
| "not happy" | no happy songs | **Fail**: happy songs at confidence 1.00 | Pass |
| "pump me up for the gym but no pop" | no pop anywhere in the list | **Fail**: pop at #2 and #4 | Pass |
| "I am late for my dentist appointment" | refuse to guess | **Fail**: study music at confidence 0.95 | Pass: refused |
| "sleep then gym" | notice the contradiction | **Fail**: ambient mixed with rock, unflagged | Pass: follows one card, flagged |
| "no jazz, upbeat rock or pop" | rock or pop, no jazz | **Fail**: all three rejected, confidence 0.00 | Pass: pop #1, jazz excluded |
| "skip the jazz, chill lofi or ambient" | chill lofi/ambient | **Fail**: lofi, ambient and "chill" all rejected | Pass: lofi #1 |
| "I want pop, not indie pop" | pop, no indie pop | **Fail**: pop penalised as avoided, confidence 0.50 | Pass: pop #1 at 1.00 |
| "less acoustic please" | lower acousticness | **Fail**: treated as exclusion-only, confidence 0.00 | Pass: acousticness 0.00–0.50, not flagged |
| "good songs" | don't trust one incidental word | **Fail**: answered at confidence 0.81 | Pass: 0.49, flagged |
| "something to dance to at my cousin's wedding" | upbeat danceable pop | Pass | Pass |
| "high energy lofi" | flag the conflict | Pass: relaxed + flagged | Pass |
| "I'm bored" | handle gracefully | Pass: refuses to guess | Pass |
| "música para estudiar" | handle gracefully | Pass offline: refused, no crash | **Answered correctly with Gemini** |
| **"sad songs for a road trip"** | sad *and* driving | **Fail**: sadness dropped | **Still fails** — known limitation (§2.4) |

The last row is left in deliberately. A 17/18 table with the failure named is worth more than
an 18/18 table that quietly dropped the case.

### 4.4 Prompt-injection suite

26 attacks in [`data/eval/injection_suite.json`](data/eval/injection_suite.json), covering
direct override text, zero-width and full-width obfuscation, delimiter escapes, schema abuse
(out-of-range numbers, SQL in enum fields) and instruction-like paraphrases.

In the test suite they run against a **fully hijacked** fake model that actively returns
attacker payloads in every field; the system must still act only on catalog values within
[0, 1]. There the suite scores **26/26**, and so it does live under lexical retrieval.

**Live, with strong retrieval, it does not.** The score falls monotonically as retrieval gets
stronger — 26/26 with `bm25`, 24/26 with `hybrid`, 23/26 with `dense` — as
`delimiter_escape`, `schema_abuse` and then `bias_bait` each stop refusing and answer
confidently instead. The schema defence (mitigation 4) still holds in every case: no attacker
value ever reaches the scorer, and nothing is echoed. What fails is the *refusal*: the system
answers an attack it should decline, because the confidence signal that drives refusal is
borrowed from the retrieval score. See §2.3. This is a real gap, not a measurement artifact,
and it is the reason the injection row in §4.1 reports both numbers rather than the better one.

### 4.5 Explanation specialization: baseline vs. constrained output

The explanation layer is the specialized component: output is constrained to a fixed, grounded
shape rather than left to free generation. Reproduce with:

```bash
python -m src.compare_explainers      # saved output: logs/explainer_comparison.txt
```

The baseline is not re-typed from memory; the script loads the Module 3 scorer straight out of
commit `12213e3` and runs it.

| | Baseline (Module 3) | Specialized (VibeFinder) |
|---|---|---|
| Source | `"; ".join(reasons)` at commit `12213e3` | `TemplateExplainer`, `src/llm.py` |
| Output | `genre match (+1.0); mood match (+1.0); energy proximity (+1.80)` | `Quantum Lullaby fits because genre fits the context: ambient electronic (+0.75); mood fits the context: focused (+0.50); energy 0.26 fits target 0.25 (+1.47). Context (Late-night studying): Studying works best with low-energy, steady music that does not compete for attention.` |
| Length | 9 words | 39 words |
| Names the song | no | yes |
| Cites scored evidence | weight labels only | every contributing reason, with its value |
| Listening guidance | none | the retrieved card's guidance, verbatim |
| Trade-offs stated | never | separate "Trade-off:" clause when a feature misses |

The measurable difference is **grounding**, not length: the baseline string cannot be attributed
to a song or a situation, so it is unusable on its own.

**The constraint is enforced, not just requested.** Validation runs against a stub client, so
this table is offline and deterministic:

| Reply style | Words | Result |
|---|---|---|
| Short, names the song | 12 | **KEPT** |
| Omits the song title | 12 | **REJECTED** → template fallback |
| Marketing padding over the cap | 92 | **REJECTED** → template fallback |
| Cites a card it was not given | 20 | **REJECTED** → template fallback |
| States a number absent from its facts | 14 | **REJECTED** → template fallback |
| **Invents a qualitative claim, short, names the song** | 15 | **KEPT** |

That last row is a limitation, not a passing grade (§2.2). 3.0 added the card-citation and
stray-number checks; it did not solve invented qualitative claims, and does not claim to.

### 4.6 How the benchmark avoids grading its own homework

- The 44 gold queries in [`data/eval/retrieval_gold.json`](data/eval/retrieval_gold.json) were
  labelled **before any retriever ran on them**. They cover direct requests, synonyms,
  multi-intent requests, non-English requests and off-topic requests.
- Score calibration uses a **separate, disjoint** query set
  ([`calibration_queries.json`](data/eval/calibration_queries.json)), and a test enforces that
  the two sets do not overlap.
- The 7 Gemini-only behaviour cases are reported as SKIP when offline, **never counted as
  passes**.
- `tools/mutate.py` plants 67 realistic bugs one at a time (invented numbers allowed through,
  Gemini retries switched off, edited cards undetected by the index, flagged input sent to
  Gemini). The suite must fail on every one. This tests the tests.

### 4.7 What surprised me during testing

- **My first test suite proved nothing.** It passed 17/17 because I wrote the cases after
  seeing the outputs. Ten minutes of *unscripted* requests found the worst bug in the project:
  "not jazz" produced jazz **at confidence 1.00**, and retrieval made it worse because the word
  "jazz" pulled in the jazz card.
- **My negation fix passed its tests and was still wrong.** A later review found "not jazz" was
  fixed but "anything but jazz" was not: I had fixed the one phrasing I tested. The regression
  case held while the bug class stayed open. The real fix was to stop matching phrases per genre
  and work out which *part of the sentence* is being rejected, once, for genres, moods and
  features alike.
- **A bug that only appears across processes.** A conflict message listed features in a set's
  iteration order, which changes with Python's per-process hash randomization. My "run it 5
  times" check ran in one process, so it could never see it. I only noticed because two terminal
  runs differed. That changed what I think "deterministic" means, and the consistency check now
  spans processes with different `PYTHONHASHSEED`.
- **Confidence was easier to compute than to make meaningful.** Raw "fit" was ~1.0 for almost
  everything. Adding an *evidence* term is what made vague requests drop below the 0.60
  threshold and get flagged. Reported confidence is rounded but never across the threshold: a
  raw 0.5995 once printed as "confidence 0.60 is below 0.60".
- **BM25 gave a one-word request full confidence.** "Good songs" matched the cooking card's body
  text "feel-good" at 0.81, because normalising by the query length lets any single rare word
  look like a perfect match. Words a card is *about* now count for more than words its prose
  happens to use.
- **Lexical and embedding scores live on different scales.** Plugging a new retriever into
  thresholds tuned for TF-IDF would have silently changed what counts as evidence. Each
  retriever is now mapped onto one scale, with a floor and a full-evidence point fitted on
  held-out queries.
- **One surprise was infrastructure, not code.** Bringing up the live path on 2026-11-14,
  every model in the configured chain returned 404 — including `gemini-2.5-flash`, which
  `models.list()` happily reported. The real message was that it is "no longer available to new
  users". A model being listed does not mean it is callable, and a fallback chain is only as
  good as its newest entry.
- **The biggest surprise: the upgrade broke the guardrail.** I expected the live run to score
  *better* than the offline one, because the whole point of the Gemini path is the cases
  lexical retrieval cannot reach. It scored worse (30/40 against 33/33, 24/26 against 26/26),
  and the failures were not in understanding — they were in refusing. Six cases that correctly
  declined offline now answered. My first instinct was that the LLM parser was inventing
  intent, which was wrong: holding everything else constant and changing only the retriever
  showed the cause was dense retrieval keeping the evidence term high for input that deserved
  none (§2.3). The lesson generalises past this project: **a confidence signal borrowed from a
  component silently inherits that component's weaknesses as its calibration, and a green
  offline suite cannot see it**, because every threshold in that suite was fitted against the
  component it is still using.

---

## 5. AI collaboration reflection

I built this project with an AI coding assistant (Claude) acting as a pair programmer. It read
the project spec, inspected my Module 3 repo, proposed an architecture, wrote code and tests,
and ran them. My job was to decide the scope, check the claims against real output, and push
back. Every interaction log is in [`ai_interactions.md`](ai_interactions.md).

### A suggestion I accepted

**Use structured context cards instead of plain text documents for retrieval.** The assistant
proposed giving each listening-context card YAML front matter — genres, moods and numeric
feature ranges — rather than storing prose and pasting the retrieved text beside a normal
answer.

*Why I accepted it:* it was the only version of RAG in which retrieval actually **changes the
system's behaviour**. With front matter, a retrieved card rewrites the scoring targets, so
retrieval moves the ranking; without it, retrieval would only have decorated the output. It is
also what made the retrieval lift measurable at all, since I could ablate the card's ranges and
watch the top result change. The measured lift — 2/9 to 9/9 correct top result on
situation-style requests — is the evidence that it was the right call.

*What I added on top:* the assistant's cards were hand-written ranges. I had them recalibrated
against the real 580-track catalog once the catalog landed, because ranges invented for 18
fictional songs did not describe real music.

### A suggestion I rejected and revised

**The first evaluation harness.** The assistant produced a harness that reported **17/17
passing** and presented that as evidence the system was reliable.

*Why I rejected it:* the cases had been written *after* looking at the system's outputs, so the
score largely measured the system agreeing with itself. A green harness built that way is worse
than no harness, because it stops you looking.

*How I revised it instead of discarding it:* I kept the harness as a regression suite but
changed where its cases come from. I had the system probed with requests it had not been tuned
on, which immediately exposed the negation bug ("not jazz" → jazz at confidence 1.00) and the
"work out" parsing bug. Those became cases that **originally failed**, so the suite now contains
evidence rather than agreement. I then added `tools/mutate.py`, which plants 67 bugs to check
that the tests actually fail when the code is wrong — testing the tests, which was the real
answer to the 17/17 problem. The one case that still fails ("sad songs for a road trip") is
reported as a failure in §4.3 rather than deleted.

### A second pair, from the 3.0 work

- **Accepted:** route every Gemini call through one client (`src/gemini.py`) carrying retry,
  model fallback, a sha256 disk cache and token accounting. I had been about to call the SDK
  from three modules. The single door is what made the free-tier quota survivable, the
  evaluation reproducible, and — unexpectedly — the 404 outage diagnosable, because the fallback
  logic was in one place to inspect.
- **Rejected:** letting Gemini rank the songs. It was offered as the simplest way to use the
  model well, and it would have made ranking non-deterministic, put user text on the path to the
  scorer, and made the system untestable offline. I scoped Gemini to understanding, embedding,
  optional rerank and explanation, and kept ranking, confidence and the review flag in
  deterministic code. The §3.2 injection argument only works *because* of that boundary.

### Where I double-checked the AI

- **Every number in the README and this card was copied from an actual run**, not from the
  assistant's summary. The `logs/` directory holds the raw output for each.
- **One "deterministic" claim turned out to be false** across processes (§4.7). The assistant
  had written the consistency check single-process and called it settled.
- **A claimed test count was wrong twice**, which is why two commits in history do nothing but
  correct it.
- **The assistant's first catalog build mapped Spotify's `chill` tag to lofi.** Listening to what
  that actually selected (ODESZA, EDM remixes) is what caught it.
- **When the live path 404'd, I did not accept the assistant's first theory** (a bad key). The
  key worked; the models were gone. Reading the actual 404 body was what settled it.
- **I did not accept the first explanation of the live regression either** — mine or the
  assistant's. Both of us initially read the under-firing review flag as the LLM parser
  inventing intent. The controlled re-run (same everything, retriever swapped) showed it was
  retrieval, not the parser. The lesson I took from it is that a plausible causal story about
  an AI component is worth exactly as much as the experiment that isolates it.

---

## 6. Version history

### 1.0 → 2.0

| 1.0 problem (from its model card) | 2.0 change |
|---|---|
| Never flagged unsatisfiable requests | `unsatisfiable` and `conflict` checks, plus a human-review flag |
| Bad matches scored 0, not negative | Numeric similarity now runs from +1 to −1 |
| One genre monopolized the top 5 | Variety re-rank (genre/artist repeat penalty) |
| Genre over-weighting hid adjacent styles | Related-genre partial credit ("pop" ~ "indie pop") |
| Users had to write a Python dict | Free-text requests + retrieval |
| `Recommender` class was a TODO stub | Wired to the shared scorer; the Module 3 tests pass |

### 2.0 → 3.0

| | VibeFinder 2.0 | VibeFinder 3.0 |
|---|---|---|
| Catalog | 18 invented songs | 580 real tracks, documented provenance |
| Knowledge base | 13 context cards | 30 cards, calibrated to the real catalog |
| Query understanding | Regex rules only | Gemini, schema-constrained to the catalog's vocabulary; rules as fallback *and* cross-check |
| Retrieval | TF-IDF | Hybrid: BM25 + Gemini embeddings, RRF fusion, MMR diversity, one calibrated evidence scale, optional rerank |
| Explanations | Template, or one Gemini call per song | One batched call, each explanation validated independently |
| Conversation | One-shot | Multi-turn refinement |
| Interface | CLI | Streamlit app with cover art and previews; CLI still works |
| Safety | Override detection | Plus NFKC normalisation, a 26-attack injection suite, and distress detection |
| Evaluation | 29 checks, 211 tests, 47 mutants | 40 checks, 426 tests, 67 mutants, 44-query labelled retrieval benchmark with ablation |
| LLM text in prompts | Never | **Yes, for understanding and rerank** — see §3.1 |
