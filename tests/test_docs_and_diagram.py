"""Checks on the submitted documentation itself.

These are cheap, offline and catch the two failure modes that bite hardest at submission
time: a diagram whose embedded copy has drifted from its source, and a relative link in the
docs that points at a file that is not in the repo.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DIAGRAM_MMD = ROOT / "diagrams" / "architecture.mmd"
DIAGRAM_MD = ROOT / "diagrams" / "architecture.md"
DOCS = ("README.md", "model_card.md", "diagrams/architecture.md", "data/CATALOG_PROVENANCE.md")


def _graph_from_mmd() -> str:
    lines = DIAGRAM_MMD.read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if not l.startswith("%%")).strip()


def _graph_from_md() -> str:
    body = DIAGRAM_MD.read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(.*?)```", body, flags=re.DOTALL)
    assert len(blocks) == 1, f"expected exactly one mermaid block, found {len(blocks)}"
    return blocks[0].strip()


def test_embedded_diagram_matches_canonical_source() -> None:
    """diagrams/architecture.md must embed exactly the graph in architecture.mmd."""
    assert _graph_from_md() == _graph_from_mmd(), (
        "diagrams/architecture.md has drifted from diagrams/architecture.mmd. "
        "The .mmd file is canonical; re-embed it rather than editing the .md block."
    )


def test_diagram_is_a_flowchart_with_both_checkpoint_kinds() -> None:
    graph = _graph_from_mmd()
    assert graph.startswith("flowchart"), "the diagram should be a Mermaid flowchart"
    # The grading checklist asks for human and testing checkpoints to be visible.
    for node in ("H1", "H2", "H3", "H4"):
        assert f"{node}[[" in graph, f"human checkpoint {node} is missing"
    for node in ("T1", "T2", "T3", "T4"):
        assert f"{node}[[" in graph, f"testing checkpoint {node} is missing"
    assert "class H1,H2,H3,H4 human" in graph
    assert "class T1,T2,T3,T4 testing" in graph


def test_diagram_names_the_real_modules() -> None:
    """Every src/ module named in the diagram must exist, so the diagram can't go stale."""
    graph = _graph_from_mmd()
    for ref in sorted(set(re.findall(r"\b(?:src|app|tools|data)/[\w./]+", graph))):
        target = ROOT / ref.rstrip(".")
        assert target.exists(), f"the diagram names {ref}, which does not exist"


@pytest.mark.parametrize("doc", DOCS)
def test_relative_links_in_docs_resolve(doc: str) -> None:
    """No broken relative links in the submitted markdown."""
    path = ROOT / doc
    text = path.read_text(encoding="utf-8")
    broken = []
    for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        resolved = (path.parent / target.split("#", 1)[0]).resolve()
        if not resolved.exists():
            broken.append(f"[{label}]({target})")
    assert not broken, f"{doc} has broken relative links: {broken}"


KEY_PATTERNS = {
    "Gemini": r"AIza[0-9A-Za-z_\-]{30,}",        # AI Studio keys
    "NVIDIA": r"nvapi-[0-9A-Za-z_\-]{20,}",      # build.nvidia.com keys
}


def test_no_api_key_shaped_strings_in_tracked_docs() -> None:
    """Real provider keys must never appear in a tracked file."""
    for doc in DOCS + ("requirements.txt", ".env.example", ".gitignore"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        for provider, pattern in KEY_PATTERNS.items():
            assert not re.search(pattern, text), f"{doc} looks like it holds a {provider} key"


@pytest.mark.parametrize("provider, sample", [("Gemini", "AIza" + "x" * 35), ("NVIDIA", "nvapi-" + "y" * 40)])
def test_the_key_patterns_would_catch_a_real_key(provider, sample) -> None:
    assert re.search(KEY_PATTERNS[provider], sample)


def test_env_is_ignored_and_example_is_not() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [l.strip() for l in ignored], ".env must stay gitignored"
    assert (ROOT / ".env.example").exists(), ".env.example should be committed as the template"


# --- committed-index integrity -------------------------------------------------------
#
# The README tells a grader the embedding index is committed, so that `dense` and `hybrid`
# retrieval work straight after a clone without an API key or a rebuild. That claim was
# silently false once: data/index/cards.json existed locally but had never been added, so a
# fresh clone quietly fell back to lexical retrieval. These tests run offline and fail if
# that happens again.

INDEX_DIR = ROOT / "data" / "index"


def test_committed_index_files_are_present() -> None:
    for name in ("manifest.json", "cards.json", "calibration.json"):
        target = INDEX_DIR / name
        assert target.exists(), (
            f"data/index/{name} is missing. The README promises a committed index; "
            "run `python -m src.index build` and `calibrate`, then commit data/index/."
        )


def test_committed_index_matches_the_cards_on_disk() -> None:
    """Equivalent to `python -m src.index verify`, but offline and in the test suite.

    Uses the real data/knowledge and data/index paths explicitly. conftest.py points the
    rest of the suite at the small 2.0 fixture world, and going through get_settings() here
    would compare the real index against the 13 fixture cards and always report it stale.
    """
    from src.embeddings import load_index
    from src.retrieval import load_cards

    cards = load_cards(ROOT / "data" / "knowledge")
    index = load_index(cards, INDEX_DIR)

    assert index is not None, "no index could be loaded from data/index"
    assert not index.stale, f"the index is stale for: {index.stale} (card text changed)"
    assert not index.missing, f"these cards have no vector: {index.missing}"
    assert index.usable, "the committed index is not usable"


def test_calibration_covers_every_retriever_that_needs_it() -> None:
    import json

    calibration = json.loads((INDEX_DIR / "calibration.json").read_text(encoding="utf-8"))
    for retriever in ("bm25", "dense"):
        assert retriever in calibration, f"no fitted score scale for {retriever}"
        entry = calibration[retriever]
        assert entry["floor"] < entry["full"], f"{retriever} scale is inverted or flat"


# --- the documented test count must match reality ------------------------------------
#
# The suite total is quoted in several places across README.md, model_card.md and the
# diagram. Rather than anchor on one sentence (which breaks the moment the prose is
# reworded), find every number that is presented as a test count and require them all to
# agree with each other and with what pytest actually collected.

COUNT_DOCS = ("README.md", "model_card.md", "diagrams/architecture.mmd")

# Each pattern captures a number that is claimed to be the size of the test suite.
COUNT_PATTERNS = (
    r"\*\*(\d{3,}) tests\*\*",          # **402 tests**
    r"\*\*(\d{3,}) pytest tests\*\*",   # **402 pytest tests**
    r"\*\*(\d{3,}) / \d{3,} pass\*\*",  # **402 / 402 pass**
    r"(\d{3,}) tests, offline",          # 402 tests, offline
    r"All (\d{3,}) tests run",            # All 402 tests run
    r"^tests/\s+(\d{3,}) tests",          # repo map line
    r"pytest: (\d{3,}) tests",            # diagram node
    r"(\d{3,}) tests; fixtures/",         # repo map, inline form
)

# Numbers that describe an *earlier* version on purpose, and must not be rewritten.
HISTORICAL = {211}


def _documented_counts() -> dict:
    found: dict = {}
    for doc in COUNT_DOCS:
        text = (ROOT / doc).read_text(encoding="utf-8")
        hits = set()
        for pattern in COUNT_PATTERNS:
            for raw in re.findall(pattern, text, flags=re.MULTILINE):
                value = int(raw)
                if value not in HISTORICAL:
                    hits.add(value)
        if hits:
            found[doc] = hits
    return found


def test_every_doc_quotes_the_same_test_count() -> None:
    found = _documented_counts()
    assert found, f"no test count found in any of {COUNT_DOCS}"
    everything = set().union(*found.values())
    assert len(everything) == 1, f"docs disagree on the test count: {found}"


def test_documented_test_count_matches_the_suite(pytestconfig) -> None:
    collected = getattr(pytestconfig, "vibefinder_collected", None)
    assert collected is not None, "the collection hook in tests/conftest.py did not run"
    if not collected["full_run"]:
        pytest.skip("partial run: the collected count is not the suite total")

    documented = set().union(*_documented_counts().values())
    assert documented == {collected["count"]}, (
        f"the docs say {sorted(documented)} tests, the suite collected "
        f"{collected['count']}. Update the count in README.md, model_card.md and "
        "diagrams/architecture.mmd (then re-embed the diagram into architecture.md)."
    )


def test_committed_dense_calibration_is_in_the_units_retrieval_uses() -> None:
    """A calibration fitted on raw cosine is refused at runtime with only a log warning, so
    committing a stale one would quietly fall back to the default. Catch it here instead."""
    import json

    from src.retrieval import DenseRetriever

    dense = json.loads((INDEX_DIR / "calibration.json").read_text(encoding="utf-8"))["dense"]
    assert dense.get("signal") == DenseRetriever.SIGNAL, (
        f"data/index/calibration.json has a dense calibration in {dense.get('signal', 'native')!r} "
        f"units; retrieval emits {DenseRetriever.SIGNAL!r}. Run: python -m src.index calibrate"
    )
    assert dense.get("n_cards") == 30, "the dense calibration was fitted on a different card count"
