# Model Card: VibeFinder 2.0

*Extends VibeFinder 1.0 (Module 3 Music Recommender Simulation). The original model card is preserved in this repo's git history, at commit `12213e3`.*

## 1. Model overview

| | |
|---|---|
| **Name** | VibeFinder 2.0 |
| **Type** | Retrieval-augmented, content-based recommender wrapped in a self-checking agent loop |
| **Input** | A free-text request in English (≤ 300 characters) |
| **Output** | Top-k songs with explanations, a confidence score (0–1), a `needs_review` flag with a clarifying message, and a step-by-step trace |
| **AI features** | RAG (context-card retrieval that changes scoring targets); agentic plan → act → verify → revise loop; confidence scoring; guardrails; optional Gemini explanations with validation and fallback |
| **Data** | `data/songs.csv` (18 fictional tracks, unchanged from Module 3); `data/knowledge/` (13 hand-written context cards) |

## 2. Intended use and non-intended use

**Intended:** a portfolio and classroom demonstration of how to make an AI recommender *grounded* (it retrieves before deciding), *self-checking* (it verifies and revises its own answer) and *honest* (it flags low confidence instead of hiding it).

**Not intended for:** real listeners (18 songs is a toy catalog), any profiling of users, or any setting where a recommendation is treated as authoritative. Confidence measures how well songs fit the *stated* request, not whether a person will actually like them.

## 3. Testing results

| Evaluation | Result |
|---|---|
| Evaluation harness (`python -m src.evaluate`) | **29/29 checks pass**: 23 behaviour, 5 robustness, 1 consistency |
| Unit and behaviour tests (`python -m pytest`) | **129/129 pass** |
| Retrieval lift on situation-style requests | baseline without retrieval **2/7** correct top result → with RAG **7/7** |
| Confidence calibration | answers not flagged average **0.96**; flagged answers average **0.29** |
| Consistency | identical output across 5 repeated runs, and across 3 processes with different `PYTHONHASHSEED` |

Human evaluation of untuned inputs (probing done after the harness first passed):

| Input | Criterion | Before fixes | After |
|---|---|---|---|
| "relaxing music but not jazz" | no jazz in top 3 | **Fail**: jazz #1 at confidence 1.00 | Pass |
| "I hate rock, give me something to work out to" | workout songs, no rock | **Fail**: parsed genre=rock, retrieved *coding* card | Pass |
| "anything but jazz" | no jazz | **Fail**: jazz #1 at confidence 1.00 | Pass: no jazz, flagged as exclusion-only |
| "not loud music" | no loud songs | **Fail**: Storm Runner at confidence 0.99 | Pass: energy capped at 0.60, flagged |
| "not happy" | no happy songs | **Fail**: happy songs at confidence 1.00 | Pass |
| "pump me up for the gym but no pop" | no pop anywhere in the list | **Fail**: Gym Hero #2, latin pop #4 | Pass |
| "I am late for my dentist appointment" | refuse to guess | **Fail**: study music at confidence 0.95 | Pass: refused |
| "sleep then gym" | notice the contradiction | **Fail**: ambient mixed with rock, unflagged | Pass: follows one card, flagged |
| "something to dance to at my cousin's wedding" | upbeat danceable pop | Pass | Pass |
| "sad songs for a road trip" | sad *and* driving | **Fail**: sadness dropped (the night-drive card dominates) | Still fails (known limitation) |
| "high energy lofi" | flag the conflict | Pass: relaxed + flagged | Pass |
| "música para estudiar" | handle gracefully | Pass: refused, no crash | Pass (but not *useful*) |
| "I'm bored" | handle gracefully | Pass: refuses to guess | Pass |

## 4. Limitations and biases

- **Tiny, skewed catalog.** 18 songs, 12 genres, most genres with 1–2 tracks, leaning toward chill/acoustic music. Whatever genre you name, the "best" match is often the *only* match. Retrieval cannot recommend what isn't there.
- **Knowledge-base bias.** The 13 context cards encode *my* assumptions about what music suits an activity (for example, that studying means low-energy lofi, or that workouts mean pop and rock). Someone who studies to metal or works out to jazz is systematically served worse. The cards are also English-only and Western-centric in their genre vocabulary.
- **Keyword retrieval has no synonyms.** "Bummed" won't match "sad", and non-English requests are refused. It fails safe, but it isn't helpful.
- **Multi-intent requests collapse to one intent.** A second card is only blended in if it scores at least 60% of the best card, so "sad songs for a road trip" loses the sadness.
- **Confidence is a heuristic, not a probability.** It is `fit × evidence − 0.15 if relaxed`. A one-word request like "pop" scores 1.00 because every pop song fits "pop" perfectly, which is not the same as the user being satisfied.
- **Negation is rule-based.** It covers genres, moods and features across many phrasings ("anything but jazz", "not too loud", "I don't want anything acoustic"), but it is a list of patterns, not language understanding. A comma ends a rejection, so "no jazz, rock, or pop" rejects only jazz, and "and" extends one, so "no jazz and calm" also rejects calm.
- **Opposite contexts follow one card.** "Sleep then gym" follows the stronger match and flags the other instead of trying to serve both. The check is also blunt: "dinner with friends" is flagged because its two cards differ, even though the picks are reasonable.
- **Off-topic detection is word counting.** A request is refused if fewer than half its words appear in the knowledge base. An off-topic request made of known words ("the night of the concert was late") still gets an answer.

## 5. Misuse and prevention

The realistic misuse surface for a music recommender is small, but the same patterns would matter in a higher-stakes system:

- **Prompt injection or manipulation.** User text is used only for keyword matching and retrieval. It is **never** placed in an LLM prompt, and Gemini only sees song fields, score reasons and card guidance. Override-style text is flagged in the logs and output. Gemini replies are validated (non-empty, ≤ 70 words, must name the song) and fall back to templates on failure.
- **Over-trusting the output.** The `needs_review` flag, the confidence score and the plain-language message are designed so the system never presents a relaxed or impossible match as a confident answer.
- **Profiling.** VibeFinder stores nothing about the user and models no demographics. If it were extended with listening history, it should keep the same rule: score songs against stated preferences, not inferred identity.
- **Resource abuse.** Input is capped at 300 characters and `k` at 10. The agent loop is capped at 3 iterations.

## 6. What surprised me while testing reliability

- **My first test suite proved nothing.** It passed 17/17 because I wrote the cases after seeing the outputs. Ten minutes of throwing *unscripted* requests at the system found the worst bug in the project: "not jazz" produced jazz **at confidence 1.00**. Retrieval made it worse, because the word "jazz" pulled in the jazz card. The fix needed two layers: excluding rejected songs from ranking, and stripping rejected phrases from the retrieval query.
- **My negation fix passed its tests and was still wrong.** A later code review found that "not jazz" was fixed but "anything but jazz" was not, because I had fixed the one phrasing I tested. The regression case held while the bug class stayed open. The real fix was to stop matching phrases per genre and work out which part of a sentence is being rejected, once, for genres, moods and features alike.
- **A bug that only appears across processes.** The conflict message listed features in a set's iteration order, which changes with Python's per-process hash randomization. My "run it 5 times" consistency check ran in one process, so it could never see it. I only noticed because two terminal runs differed. That changed how I think about "deterministic".
- **Confidence was easier to compute than to make meaningful.** Raw "fit" was ~1.0 for almost everything. Adding an *evidence* term (how strongly retrieval matched) is what made vague requests like "something upbeat" drop to 0.60 and get flagged.

## 7. Changes from VibeFinder 1.0 (and why)

| 1.0 problem (from its model card) | 2.0 change |
|---|---|
| Never flagged unsatisfiable requests | `unsatisfiable` and `conflict` checks, plus a human-review flag |
| Bad matches scored 0, not negative | Numeric similarity now runs from +1 to −1 |
| One genre monopolized the top 5 | Variety re-rank (genre/artist repeat penalty) |
| Genre over-weighting hid adjacent styles | Related-genre partial credit ("pop" ~ "indie pop") |
| Users had to write a Python dict | Free-text requests + retrieval |
| `Recommender` class was a TODO stub | Wired to the shared scorer; the Module 3 tests pass |

## 8. AI collaboration

I built this project with an AI coding assistant (Claude) acting as a pair programmer. It read the project spec, inspected my Module 3 repo, proposed an architecture, wrote code and tests, and ran them. My job was to decide the scope, check the claims against real output, and push back.

- **A suggestion I accepted:** use *structured* context cards (genres, moods and numeric ranges in front matter) instead of plain text documents for retrieval. It was the clearest way to meet the requirement that RAG must *change* the system's behaviour, not just print retrieved text beside a normal answer. The retrieval-lift numbers (2/7 → 7/7) show that it does.
- **A suggestion I did not accept as written:** the first evaluation harness. It reported 17/17 passing, but the cases had been written after looking at the system's outputs, so the score mostly measured agreement with itself. I didn't treat that as evidence of reliability. Instead I had the system probed with requests it hadn't been tuned on, which exposed the negation and "work out" bugs. Those cases were added as regression tests, so the final 19/19 includes cases that originally **failed**.
- **Scoped down:** the LLM's role. Having Gemini parse the whole request was an option, but it would have made ranking non-deterministic, put user text into prompts (an injection surface), and made the system untestable offline. Gemini was limited to an optional explanation layer with output validation and a template fallback.
- **Where I double-checked the AI:** every number in the README was copied from an actual run, not from the assistant's summary. One "deterministic" claim turned out to be false across processes (see section 6).
