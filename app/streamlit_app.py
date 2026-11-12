"""
VibeFinder 3.0 - Streamlit app.

    streamlit run app/streamlit_app.py

Three tabs:
  Discover    describe a moment, get five tracks with cover art, a 30-second preview,
              the agent's confidence, the context it used and its full trace. Follow-ups
              ("more upbeat", "not #3", "more like #2") refine the last answer.
  Compare     the same request with retrieval switched off and on, side by side.
  Evaluation  the retrieval ablation on the labelled gold set, and the behaviour harness.

Everything degrades: with no GEMINI_API_KEY the app runs the offline path and says so;
if Deezer is unreachable the cards show a placeholder instead of art and no player.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import ui  # noqa: E402
from src.agent import RETRIEVAL_MIN_SCORE, VibeFinderAgent  # noqa: E402
from src.config import RETRIEVAL_MODES, get_settings  # noqa: E402
from src.embeddings import make_retriever  # noqa: E402
from src.enrich import lookup  # noqa: E402
from src.gemini import GeminiClient, shared_client  # noqa: E402
from src.llm import GroundedBatchExplainer, TemplateExplainer  # noqa: E402
from src.log_setup import setup_logging  # noqa: E402
from src.recommender import load_songs  # noqa: E402
from src.retrieval import load_cards  # noqa: E402
from src.session import Conversation  # noqa: E402
from src.understanding import GeminiUnderstander  # noqa: E402

st.set_page_config(page_title="VibeFinder", page_icon=":material/graphic_eq:", layout="wide",
                   initial_sidebar_state="collapsed")
setup_logging()

EXAMPLES = [
    "Something calm for studying late at night",
    "Pump me up for the gym, but no pop",
    "Feeling bummed out on a rainy Sunday",
    "Música para una fiesta en la playa",
    "Romantic dinner with some jazz",
]
KIND_NAMES = {"direct": "Direct", "synonym": "Synonym", "multi_intent": "Multi-intent",
              "non_english": "Non-English"}
MODE_LABELS = {"hybrid": "Hybrid (BM25 + embeddings)", "dense": "Embeddings only",
               "bm25": "BM25", "tfidf": "TF-IDF (2.0 baseline)"}


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner=False)
def catalog():
    settings = get_settings()
    return load_songs(settings.songs_path), load_cards(settings.knowledge_dir)


@st.cache_resource(show_spinner=False)
def build_agent(use_gemini: bool, mode: str, rerank: bool, k: int):
    songs, cards = catalog()
    base = get_settings()
    settings = base.__class__(use_gemini=use_gemini and base.gemini_enabled, rerank=rerank)
    client = shared_client() if settings.gemini_enabled else GeminiClient(settings=settings)
    retriever, notes = make_retriever(cards, mode=mode, settings=settings, client=client)
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})
    understander = GeminiUnderstander(client, genres, moods) if client.available else None
    explainer = GroundedBatchExplainer(client) if client.available else TemplateExplainer()
    agent = VibeFinderAgent(songs=songs, cards=cards, retriever=retriever, understander=understander,
                            explainer=explainer, k=k)
    agent.retrieval_notes = notes
    return agent


@st.cache_data(ttl=3600, show_spinner=False)
def media(title: str, artist: str):
    m = lookup(title, artist)
    return None if m is None else {"cover": m.cover, "preview": m.preview, "link": m.link}


def settings_state():
    s = st.session_state
    s.setdefault("use_gemini", get_settings().gemini_enabled)
    s.setdefault("mode", "hybrid")
    s.setdefault("rerank", False)
    s.setdefault("k", 5)
    return s


def status_line(agent: VibeFinderAgent, use_gemini: bool) -> str:
    client = shared_client()
    if not get_settings().gemini_enabled:
        gem = "Offline (no Gemini key)"
    else:
        gem = "Gemini on" if use_gemini else "Gemini off"
    return f"{gem} · {MODE_LABELS[st.session_state.mode].split(' (')[0]} · {len(agent.songs)} tracks"


# ------------------------------------------------------------------- shell
st.html(ui.CSS)
state = settings_state()
agent = build_agent(state.use_gemini, state.mode, state.rerank, state.k)
if "conversation" not in state or state.get("agent_key") != (state.use_gemini, state.mode, state.rerank, state.k):
    state.conversation = Conversation(agent)
    state.agent_key = (state.use_gemini, state.mode, state.rerank, state.k)
conversation: Conversation = state.conversation

bar_left, bar_right = st.columns([5, 1], vertical_alignment="center")
with bar_left:
    st.html(ui.top_bar(status_line(agent, state.use_gemini)))
with bar_right:
    with st.popover("Settings", use_container_width=True):
        key_ok = get_settings().gemini_enabled
        st.toggle("Use Gemini", key="use_gemini", disabled=not key_ok,
                  help=None if key_ok else "Add GEMINI_API_KEY to .env to enable Gemini.")
        if not key_ok:
            st.caption("No GEMINI_API_KEY found, so VibeFinder runs fully offline: rule-based "
                       "parsing, lexical retrieval and template explanations.")
        st.selectbox("Retrieval", RETRIEVAL_MODES[::-1], key="mode", format_func=MODE_LABELS.get)
        st.toggle("Gemini rerank of context cards", key="rerank", disabled=not (key_ok and state.use_gemini))
        st.slider("Tracks per answer", 1, 10, key="k")
        stats = shared_client().stats
        st.html(ui.kv([("Gemini calls", f"{stats.calls:,}"), ("Cache hits", f"{stats.cache_hits:,}"),
                       ("Fallbacks", f"{stats.failures:,}"),
                       ("Tokens", f"{stats.prompt_tokens + stats.output_tokens:,}")]))
        if agent.retrieval_notes:
            st.caption("Retrieval: " + "; ".join(agent.retrieval_notes))

discover, compare, evaluation = st.tabs(["Discover", "Compare", "Evaluation"])


# ------------------------------------------------------------------ discover
def run_request(text: str) -> None:
    if text and text.strip():
        with st.spinner("Finding tracks"):
            conversation.send(text.strip())


def render_lead(result, titles) -> None:
    rec = result.recommendations[0]
    song = rec.song
    m = media(song["title"], song["artist"])
    lead, side = st.columns([2, 1])
    with lead:
        with st.container(key="card_lead"):
            art, info = st.columns([1, 2])
            with art:
                st.html(ui.cover(m and m["cover"], f"Cover of {song['title']}"))
            with info:
                st.html(f'<div class="vf-label">Top pick for you</div>'
                        f'<h2 class="vf-track-title" style="margin:2px 0 0">{ui.e(song["title"])}</h2>'
                        f'<div class="vf-muted">{ui.e(song["artist"])}</div>'
                        + ui.pills([(song["genre"], ""), (song["mood"], ""),
                                    (f"fit {rec.fit:.2f}", "pos")])
                        + f'<div class="vf-p">{ui.e(rec.explanation)}</div>')
                if m and m["preview"]:
                    st.audio(m["preview"], format="audio/mp3")
                else:
                    st.caption("No preview available for this track.")
            with st.expander("Why this track"):
                st.html(ui.kv([(r.rsplit(" (", 1)[0], r.rsplit(" (", 1)[1].rstrip(")") if " (" in r else "")
                               for r in rec.reasons]))
                src = "Gemini, checked against the facts" if rec.explained_by == "gemini" else "the offline template"
                st.caption(f"Explanation written by {src}"
                           + (f"; grounded in the '{titles.get(rec.cited_card, rec.cited_card)}' card."
                              if rec.cited_card else "."))
    with side:
        with st.container(key="card_confidence"):
            st.html('<div class="vf-label">Confidence</div>'
                    f'<div class="vf-figure">{result.confidence:.2f}</div>'
                    + (ui.pills([("Needs your review", "warn")]) if result.needs_review
                       else ui.pills([("No review needed", "pos")])))
            if result.support:
                st.html(f'<div class="vf-p" style="color:#e5a93b">{ui.e(result.support)}</div>')
            if result.message:
                st.html(f'<div class="vf-p vf-muted">{ui.e(result.message)}</div>')
            st.html('<div class="vf-section" style="margin-top:14px">Listening context</div>'
                    + ui.context_list(result.retrieved, titles)
                    + ui.kv([("Understood by", "Gemini" if result.parser == "gemini" else
                              "follow-up rules" if result.parser == "refine" else "rule parser"),
                             ("Retrieval", result.retrieval_mode or "-")]))


def render_rest(result) -> None:
    profile = getattr(result, "_profile", None)
    with st.container(key="card_landing"):
        st.html('<div class="vf-section">Where the picks land</div>'
                '<div class="vf-label" style="margin-bottom:10px">Your request as target ranges; '
                'the top pick in lime, the others in white.</div>'
                + ui.landing_chart(profile.targets if profile else {}, [r.song for r in result.recommendations]))
    if len(result.recommendations) > 1:
        with st.container(key="card_more"):
            st.html(f'<div class="vf-section">Also fits</div>'
                    f'<div class="vf-label">{len(result.recommendations) - 1} more, ranked with a variety '
                    'check so one artist or genre cannot fill the list</div>')
            for rec in result.recommendations[1:]:
                song = rec.song
                m = media(song["title"], song["artist"])
                c_art, c_text, c_fit, c_play = st.columns([0.55, 4, 1, 3], vertical_alignment="center")
                with c_art:
                    st.html(ui.cover(m and m["cover"], f"Cover of {song['title']}", small=True))
                with c_text:
                    st.html(f'<div class="vf-row-title">{ui.e(song["title"])}</div>'
                            f'<div class="vf-row-sub">{ui.e(song["artist"])} · {ui.e(song["genre"])} · '
                            f'{ui.e(song["mood"])}</div>')
                with c_fit:
                    st.html(f'<div class="vf-num" style="text-align:right">{rec.fit:.2f}</div>'
                            '<div class="vf-label" style="text-align:right">fit</div>')
                with c_play:
                    if m and m["preview"]:
                        st.audio(m["preview"], format="audio/mp3")
                    else:
                        st.html('<div class="vf-label">No preview</div>')
    with st.expander("How VibeFinder decided"):
        st.html(ui.trace(result.trace))


with discover:
    result = conversation.last
    head, actions = st.columns([5, 1], vertical_alignment="bottom")
    with head:
        title = f"“{result.cleaned_query or result.query}”" if result else "Discover"
        st.html(f'<h1 class="vf-title">{ui.e(title)}</h1>')
        if result is None:
            sub = ("Describe a moment: what you are doing, how you feel, where you are. VibeFinder picks "
                   f"from {len(agent.songs)} real tracks and tells you how sure it is.")
        elif not result.recommendations:
            sub = "No tracks this time; the reason is below."
        else:
            top = result.recommendations[0].song
            ctx = titles_ = {c.id: c.title for c in agent.cards}
            card = ctx.get(result.retrieved[0][0]) if result.retrieved else None
            sub = (f"{len(result.recommendations)} tracks" + (f" for ‘{card}’" if card else "")
                   + f", led by {top['genre']}. " + ("Flagged for your review." if result.needs_review
                                                     else "Confident in this answer."))
        st.html(f'<div class="vf-sub">{ui.e(sub)}</div>')
    with actions:
        if result is not None and st.button("New request", use_container_width=True):
            conversation.reset()
            st.rerun()

    with st.form("ask", clear_on_submit=True, border=False):
        f_in, f_go = st.columns([5, 1.3], vertical_alignment="bottom")
        with f_in:
            text = st.text_input(
                "What do you want to hear?",
                placeholder=("Try a follow-up: more upbeat, less jazz, not #3, more like #2"
                             if result and result.recommendations else EXAMPLES[0]),
                label_visibility="collapsed")
        with f_go:
            sent = st.form_submit_button("Find tracks", use_container_width=True)
    if sent:
        run_request(text)
        st.rerun()

    if result is None:
        songs_, cards_ = agent.songs, agent.cards
        ex_col, know_col = st.columns([1, 2])
        with ex_col:
            with st.container(key="card_examples"):
                st.html('<div class="vf-section">Start from one of these</div>')
                for example in EXAMPLES:
                    if st.button(example, key=f"ex_{example}", use_container_width=True):
                        run_request(example)
                        st.rerun()
            with st.container(key="card_followups"):
                st.html('<div class="vf-section">Then refine it</div>'
                        + ui.follow_ups([("more upbeat", "raise the energy, same context"),
                                         ("less jazz", "rule a genre out"),
                                         ("not #3", "drop one pick, keep the rest"),
                                         ("more like #2", "neighbours of a pick"),
                                         ("something else", "five new tracks")]))
        with know_col:
            with st.container(key="card_knows"):
                genres = sorted({s["genre"] for s in songs_})
                moods = sorted({s["mood"] for s in songs_})
                st.html(f'<div class="vf-section">What VibeFinder knows</div>'
                        f'<div class="vf-label">{len(songs_)} tracks from a public Spotify snapshot · '
                        f'{len(genres)} genres · {len(moods)} moods · {len(cards_)} listening contexts</div>'
                        + ui.context_grid([(c.title, ", ".join(c.genres[:3])) for c in sorted(cards_, key=lambda c: c.title)]))
    elif not result.recommendations:
        with st.container(key="card_empty"):
            if result.support:
                st.html(f'<div class="vf-p" style="color:#e5a93b">{ui.e(result.support)}</div>')
            st.html(f'<div class="vf-section">No tracks</div><div class="vf-p vf-muted">{ui.e(result.message or "")}</div>')
            for w in result.warnings:
                st.caption(f"Guardrail: {w}")
        with st.expander("How VibeFinder decided"):
            st.html(ui.trace(result.trace))
    else:
        titles = {c.id: c.title for c in agent.cards}
        for w in result.warnings:
            st.caption(f"Guardrail: {w}")
        render_lead(result, titles)
        render_rest(result)


# ------------------------------------------------------------------- compare
with compare:
    st.html('<h1 class="vf-title">Compare</h1><div class="vf-sub">The same request ranked twice: once on '
            'your words alone, once with retrieved listening context.</div>')
    last = conversation.last
    default = (last.cleaned_query if last and last.parser != "refine" else EXAMPLES[2])
    with st.form("compare_form", border=False):
        c_in, c_go = st.columns([5, 1.3], vertical_alignment="bottom")
        with c_in:
            q = st.text_input("Request to compare", value=default, label_visibility="collapsed")
        with c_go:
            go = st.form_submit_button("Compare", use_container_width=True)
    if go or "compare_result" not in state or (state.get("compare_default") != default and not go):
        state.compare_default = default
        songs, cards = catalog()
        baseline = VibeFinderAgent(songs=songs, cards=cards, use_retrieval=False, understander=None,
                                   explainer=TemplateExplainer(), k=state.k)
        state.compare_result = (q, baseline.run(q), agent.run(q))
    q, without, with_rag = state.compare_result
    same = len({r.song["id"] for r in without.recommendations} & {r.song["id"] for r in with_rag.recommendations})
    st.html(f'<div class="vf-sub">“{ui.e(q)}”: retrieval changed '
            f'{len(with_rag.recommendations) - same} of {len(with_rag.recommendations)} picks.</div>')
    left, right = st.columns(2)
    for col, label, res in ((left, "Without retrieval", without),
                            (right, f"With retrieval · {MODE_LABELS[state.mode].split(' (')[0]}", with_rag)):
        with col:
            with st.container(key=f"card_cmp_{label[:7]}"):
                st.html(f'<div class="vf-section">{ui.e(label)}</div>'
                        f'<div class="vf-label">Confidence</div><div class="vf-figure">{res.confidence:.2f}</div>'
                        + (ui.pills([("Needs review", "warn")]) if res.needs_review else ui.pills([("No review needed", "pos")])))
                if not res.recommendations:
                    st.html(f'<div class="vf-p vf-muted">{ui.e(res.message or "No tracks.")}</div>')
                rows = [(r.song["title"], f'{r.song["genre"]} · {r.song["mood"]}', f"{r.fit:.2f}")
                        for r in res.recommendations]
                if rows:
                    st.html(ui.table(["Track", "Genre · mood", "Fit"], rows, numeric_from=2))
                st.html(ui.kv([("Context", ", ".join(c for c, _ in res.retrieved) or "-")]))


# ---------------------------------------------------------------- evaluation
@st.cache_data(show_spinner=False)
def ablation(gemini_available: bool, _key: str):
    from src.retrieval_eval import evaluate_retriever, load_queries
    songs, cards = catalog()
    queries = load_queries()
    settings = get_settings()
    out = []
    for label, mode in (("TF-IDF (2.0)", "tfidf"), ("BM25", "bm25"), ("Embeddings", "dense"), ("Hybrid", "hybrid")):
        if mode in ("dense",) and not gemini_available:
            continue
        retriever, _ = make_retriever(cards, mode=mode, settings=settings)
        rep = evaluate_retriever(label, lambda s: [c.id for c, _ in retriever.retrieve(s, k=3, min_score=RETRIEVAL_MIN_SCORE)],
                                 queries, fallback_of=lambda: bool((getattr(retriever, "last_info", {}) or {}).get("fallback")))
        out.append(rep)
    return out, len(queries)


with evaluation:
    gem_ok = shared_client().available
    st.html('<h1 class="vf-title">Evaluation</h1><div class="vf-sub">Retrieval measured on 44 labelled requests '
            'written before any retriever saw them: direct, synonym, multi-intent, non-English and off-topic.</div>')
    reports, n = ablation(gem_ok, "v1")
    def unique_best(metric):
        """Index of the single best report on a metric, or None on a tie (a tie has no winner)."""
        values = [round(getattr(r, metric), 2) for r in reports]
        top = max(values, default=None)
        return values.index(top) if top is not None and values.count(top) == 1 else None
    best_hit, best_r3 = unique_best("hit1"), unique_best("recall3")
    left, right = st.columns([3, 2])
    with left:
        with st.container(key="card_ablation"):
            st.html('<div class="vf-section">Retrieval ablation</div>'
                    + ui.table(["Retriever", "Recall@3", "Hit@1", "MRR", "nDCG@3", "Abstain", "Fell back"],
                               [(r.name, f"{r.recall3:.2f}", f"{r.hit1:.2f}", f"{r.mrr:.2f}", f"{r.ndcg3:.2f}",
                                 f"{r.abstain:.2f}", f"{r.fallbacks}") for r in reports],
                               best={1: best_r3, 2: best_hit}))
            if not gem_ok:
                st.caption("Embeddings need GEMINI_API_KEY and a built index; Hybrid runs on its lexical arm "
                           "until then, which the 'Fell back' column counts.")
    with right:
        with st.container(key="card_kinds"):
            names = [r.name for r in reports]
            pick = st.segmented_control("Retriever", names, default=names[-1] if names else None,
                                        label_visibility="collapsed")
            rep = next((r for r in reports if r.name == pick), reports[-1] if reports else None)
            if rep:
                st.html(f'<div class="vf-section">Hit@1 by request type · {ui.e(rep.name)}</div>'
                        + ui.bar_list([(KIND_NAMES.get(k, k), v) for k, v in
                                       sorted(rep.by_kind.items(), key=lambda kv_: (-kv_[1], kv_[0]))]))
                st.html(ui.kv([("On-topic requests", f"{rep.n_on_topic}"), ("Off-topic requests", f"{rep.n_off_topic}")]))
    with st.container(key="card_misses"):
        rep = reports[-1] if reports else None
        st.html(f'<div class="vf-section">Where {ui.e(rep.name if rep else "-")} misses</div>')
        if rep and rep.misses:
            st.html(ui.table(["Request", "Retrieved", "Expected"],
                             [(q_, ", ".join(got) or "nothing", ", ".join(want) or "nothing")
                              for q_, got, want in rep.misses[:8]], numeric_from=99))
            if len(rep.misses) > 8:
                st.caption(f"Showing 8 of {len(rep.misses)} misses. Full list: python -m src.evaluate --misses")
        else:
            st.html('<div class="vf-p vf-muted">No misses.</div>')
