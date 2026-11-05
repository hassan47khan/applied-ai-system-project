# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

I called it VibeFinder because the whole point of the system is to match a listener's "vibe" — a mix of genre, mood, and a few numeric feelings — against songs in a small catalog.

---

## 2. Goal / Task

I built a simple recommender that takes a listener's stated preferences (favorite genre, favorite mood, and target levels for energy, danceability, and acousticness) and returns the five songs from my catalog that best fit those preferences, along with a short explanation for each pick.

I wasn't trying to predict what someone would actually listen to in the real world. I was trying to simulate how a very basic recommender *reasons* about taste, so I could see where that kind of reasoning breaks down.

---

## 3. Data Used

I used the `data/songs.csv` file that came with the starter project. It has **18 songs** and the following fields per song: id, title, artist, genre, mood, energy, tempo, valence, danceability, and acousticness. The numeric fields are all between 0 and 1 (except tempo, which is in BPM).

I didn't add or remove any songs. The main thing I noticed about the dataset is how uneven it is:

- There are 12 different genres across only 18 songs, so most genres have **only one or two tracks**.
- The catalog leans heavily toward chill, acoustic, lower-energy music. The median energy is 0.43 and the median acousticness is 0.70. About 40% of the songs are tagged `chill` or `relaxed`.
- Some combinations a listener might reasonably ask for (like upbeat jazz, or melancholy pop) simply don't exist in the data.

This matters a lot once the recommender starts running — the shape of the catalog quietly shapes the output.

---

## 4. Algorithm Summary

My scoring works like a judge handing out points. Every song starts at zero, and I walk through a short checklist for each one, adding points when the song matches what the listener asked for.

- If the song's **genre** matches the listener's pick exactly, it earns +2.0 points. Genre is the strongest signal.
- If the **mood** matches, it earns +1.0 more.
- For each **numeric vibe** the listener specified (energy, danceability, acousticness), I measure how close the song's value is to the listener's target. A perfect match is worth close to a full point; the reward shrinks the farther apart they are. If a song is way off, it gets zero for that category — I don't take points away.

Once every song has a score, I sort from highest to lowest and show the top five, with the specific reasons it won: "genre match", "mood match", "energy proximity (+0.98)". The explanation is as important as the ranking to me — I wanted to see *why* each song was picked.

The main change from the starter code: the starter just returned the first five songs regardless of input. Now the ranking actually responds to the listener, and every pick comes with a plain-language reason.

---

## 5. Observed Behavior / Biases

The biggest pattern I noticed is that **the genre match dominates almost everything else.** Because genre is worth +2.0 and each numeric feature can only earn up to +1.0, picking any genre essentially locks in a +2.0 head-start for every song in that genre. Combined with the thin catalog, this means a listener asking for `rock` always gets the one rock song, a listener asking for `jazz` always gets the one jazz song — regardless of what else they ask for.

A second pattern: **the system quietly favors chill, acoustic music.** The catalog is biased that way, so when a listener's numeric preferences are vague or lean neutral, mid-energy acoustic songs win because they're never *too far* from anyone's target. The data is doing some of the recommending whether I meant it to or not.

A third issue: **the system never penalizes a bad match.** If a song misses every single preference, it scores zero — the same as a song that just wasn't scored at all. That means the recommender can look confident about a terrible answer, because the math only adds and never subtracts.

---

## 6. Evaluation Process

I tested VibeFinder against several made-up listeners to see how the top-5 shifted:

- **The upbeat pop fan** — happy pop, high energy. The top pick was *Sunrise City*, which is exactly that. ✓
- **The chill lofi listener** — relaxed vibe, acoustic, low energy. Top picks were lofi and ambient tracks. ✓
- **The contradiction** — I asked for lofi (normally slow and acoustic) but at *high* energy and *not* acoustic — an impossible combination. The system served three slow mellow lofi songs and completely ignored the energy request.
- **The extremist** — ambient music that was also intense, maximally energetic, maximally acoustic, and totally undanceable. The system returned a chill ambient song whose actual numbers were far from the targets, because the genre match outweighed everything.

I also ran a **weight-shift experiment**: I halved the genre weight (2.0 → 1.0) and doubled the numeric-feature weight (1.0 → 2.0). The straightforward profiles still returned the same top picks, which told me those rankings weren't fragile. But the adversarial profiles swung in the opposite direction — now the numeric targets dominated, and listeners asking for "lofi" got gym songs because the gym songs matched the numbers. That was a useful lesson: rebalancing weights doesn't fix the deeper problem of the two kinds of signals (categorical vs. numeric) competing instead of cooperating.

What surprised me most was how **politely the system fails** on impossible requests. It never flags that a combination of preferences is unsatisfiable — it just hands over the closest fit and pretends that's the answer. For a real user, that's worse than returning nothing.

---

## 7. Intended Use and Non-Intended Use

**Intended use.** This is a classroom simulation. I built it to practice writing scoring logic, to see how small design choices (weights, penalty functions, catalog shape) ripple into output, and to produce a system I could reason about and critique honestly.

**Not intended for.**

- Real music recommendations for real listeners. The catalog is 18 songs — too small to be useful.
- Any production or decision-making context. The scoring is naïve, the data is unbalanced, and the system cannot tell a user when their request is unsatisfiable.
- Any setting where the recommendations are presented as authoritative. The system is confident on inputs it cannot actually handle.
- Any demographic or behavioral profiling. I only score songs against stated preferences; I don't model listeners.

---

## 8. Ideas for Improvement

If I kept developing VibeFinder, I'd try three things:

1. **Penalize mismatches, not just reward matches.** Right now, being wrong about genre or mood is "free" — it just doesn't add points. Subtracting a small amount for mismatches would push the system away from confidently wrong answers.
2. **Flag unsatisfiable requests.** Before scoring, I'd check whether any song in the catalog is even close to the listener's targets. If none is, say so instead of returning a bad fit.
3. **Add diversity to the top-k.** Right now the top-3 for any lofi-leaning profile is literally all three lofi songs. I'd introduce a rule that prevents the same genre or artist from monopolizing the list, so the listener sees a slightly wider range of good matches.

A bigger-picture change I'd try: let listeners weight their own preferences. Someone who cares deeply about mood but doesn't care about genre should be able to say so, and have the scoring reflect that.

---

## 9. Personal Reflection

My biggest learning moment was watching the weight-shift experiment in Step 3 of Phase 4. I expected that halving the genre weight would just produce "slightly better" results across the board. Instead, it fixed one failure mode and created the opposite one — now the numeric targets ran the show and the genre label became decorative. That drove home that **you can't just tune your way out of a bad scoring design**. The weights aren't a dial you turn until things feel right; they're a statement about which signals you trust, and if those signals disagree with each other, no single weighting works for everyone.

AI tools were genuinely useful at a few specific moments: when I needed to translate my scoring recipe into clean Python, when I wanted a second opinion on the difference between `list.sort()` and `sorted()`, and when I needed help spotting biases I wasn't going to find on my own (I would not have noticed on my own that genre and acousticness are correlated in the data, meaning the scorer rewards the same underlying trait twice). The place I had to double-check the AI's work the most was in the actual numbers — when it told me a song scored X, I ran the code myself and read the output rather than trusting the explanation. The reasoning was almost always right; the arithmetic sometimes wasn't. I also had to push back on suggestions that would have made the code more "general" or "flexible" than the project actually needed.

What surprised me about simple algorithms is how much they *feel* like recommendations even when they're doing something very basic. +2 for genre, +1 for mood, a bit of proximity math — and the output looks like a real "Top 5 For You" list, complete with explanations. The polish of the output tricks you (and the user) into thinking the system knows more than it does. That was probably the most important thing I took away from this project: a well-formatted result is not the same as a trustworthy result, and the responsibility to tell those apart lives with the person building the system.

If I extended this project, I'd want to try replacing the hand-tuned weights with something learned from actual listening data — give the system a bunch of "this user liked these songs" pairs and see if it can figure out its own weights. I'd also want to build a tiny web UI so I could watch the top-5 change in real time as I move sliders around; a lot of what I learned in Phase 4 came from rerunning the CLI, and a more interactive view would make the system's behavior easier to probe. Finally, I'd want to grow the catalog to a few hundred songs. Many of the "biases" I found really come from the catalog having 18 songs; it would be interesting to see which problems survive at a bigger scale and which ones go away.
