# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

This recommender uses a **content-based** approach: instead of learning from what other listeners enjoy, it compares the measurable attributes of each song in the catalog against a stored taste profile for the user. Every track is scored independently, then ranked, and the top results are returned. The simulation prioritizes **transparency and interpretability** — each recommendation can be traced back to the specific features that earned it points, so the behavior of the system is easy to reason about and tune.

### Features used

Each `Song` object exposes:

- `title`, `artist` — identity
- `genre` — categorical (e.g. `indie folk`, `synthwave`, `lofi`)
- `mood` — categorical (e.g. `chill`, `melancholy`, `intense`)
- `energy` — numeric, 0.0–1.0
- `tempo_bpm` — numeric, beats per minute
- `valence` — numeric, 0.0–1.0 (musical positivity)
- `danceability` — numeric, 0.0–1.0
- `acousticness` — numeric, 0.0–1.0

Each `UserProfile` stores:

- `favorite_genre`, `favorite_mood` — the categorical targets
- `target_energy`, `target_danceability`, `target_acousticness` — numeric targets on the 0.0–1.0 scale
- (Optional) `target_tempo_bpm`, `target_valence`, and per-feature `weights` for tuning

### Algorithm recipe

For each song, the recommender computes:

| Component | Rule | Points |
|---|---|---|
| Genre match | `song.genre == profile.favorite_genre` | **+2.0** |
| Mood match | `song.mood == profile.favorite_mood` | **+1.0** |
| Numerical similarity | `1.0 - abs(song.feature - profile.target)` per feature, scaled by weight | **0.0 – weight** |

Numerical similarity is applied to `energy`, `danceability`, and `acousticness` (each already on 0.0–1.0). A song exactly on target earns the full feature weight; a song at the opposite end earns zero. The final score is the sum of all components. Songs are then sorted by score (descending), ties broken by closer energy match, and the top **K** are returned alongside a short list of reasons such as `"genre match"` or `"near target energy"`.

### Pseudocode

```
for song in catalog:
    score = 0
    if song.genre == profile.favorite_genre: score += 2.0
    if song.mood  == profile.favorite_mood:  score += 1.0
    for feat in [energy, danceability, acousticness]:
        score += weight[feat] * (1 - abs(song[feat] - profile[feat]))
return top_K(sorted(scored, by=score, desc))
```

### Known biases and limitations

- **Genre over-weighting.** At +2.0, an exact genre match can outweigh several near-perfect numeric matches combined. A user who likes `indie folk` may rarely see adjacent styles like `indie pop` or `acoustic` surface, even when they'd fit.
- **Filter-bubble effect.** Because the system rewards tracks that look like the stored profile, it reinforces the user's current taste and offers little opportunity for discovery. There is no exploration term, no randomness, and no collaborative signal from other listeners.
- **Equal weighting across features ignores salience.** Not every numeric feature matters equally for every user, but the recipe treats them uniformly unless weights are tuned by hand.
- **Categorical sparsity.** Rare moods (e.g. `melancholy`) match very few songs; a profile pinned to one can collapse to a tiny result set.
- **Cold-start and catalog bias.** The system cannot recommend what isn't in the CSV, and underrepresented genres in the dataset are structurally underrecommended regardless of a user's taste.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
# e.g.:
# User profile: genre=indie, mood=chill, energy=low
# Recommendations:
#   1. ...
#   2. ...
#   3. ...
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



