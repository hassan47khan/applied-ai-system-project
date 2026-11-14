# VibeFinder 3.0: System Architecture

The canonical source is [`architecture.mmd`](architecture.mmd); the block below is the same
graph, embedded so GitHub renders it. `tests/test_diagram.py` fails if the two drift apart.
A vector export sits at [`../assets/architecture.svg`](../assets/architecture.svg).

**Checkpoints.** `H1`-`H4` (green) are the **human checkpoints**: the points where a person is
asked to look, decide, or is warned. `T1`-`T4` (blue) are the **testing checkpoints**: the
automated checks that gate behaviour before it reaches the user.

```mermaid
flowchart TD
    U([Request in any language<br/>CLI, Streamlit UI or a follow-up]) --> CONV{Follow-up?<br/>src/session.py}
    CONV -- "more upbeat / not #3 / more like #2" --> REF[Edit last profile<br/>agent.refine]
    CONV -- "new request" --> G{Guardrails<br/>src/guardrails.py<br/>NFKC, invisible chars,<br/>override + distress detection}
    G -- "empty / non-text" --> RJ[Reject with guidance]
    G -- "flagged as override" --> RULES
    G -- "distress detected" --> H1
    G -- "clean text" --> UND

    H1[["H1 human checkpoint<br/>crisis-line info surfaced,<br/>answer flagged"]] --> UND

    subgraph GEM["Gemini (src/gemini.py: retry, model fallback, disk cache, accounting)"]
        UND[Query understanding<br/>src/understanding.py<br/>JSON schema with catalog enums]
        QE[Query embedding<br/>gemini-embedding-2, 768-d]
        RR[Optional rerank<br/>src/rerank.py]
        EXP[Batched grounded explanations<br/>src/llm.py]
    end

    UND --> T1[["T1 testing checkpoint<br/>validate intent: drop values<br/>outside catalog enums,<br/>clamp ranges to 0-1,<br/>strip instruction-like text"]]
    T1 -- "validated Intent" --> PLAN
    UND -. "GeminiUnavailable" .-> RULES[Rule-based parser<br/>src/profile.py]
    RULES --> PLAN
    T1 -. "cross-check vs rules<br/>logs/parser_disagreements.jsonl" .-> H2

    H2[["H2 human checkpoint<br/>parser disagreements logged<br/>for later review"]]

    subgraph RAG["Retrieval: src/retrieval.py, src/embeddings.py"]
        KB[(30 context cards<br/>data/knowledge)]
        IDX[(Prebuilt card index<br/>data/index, committed)]
        BM[BM25<br/>body-only discount]
        DN[Dense cosine]
        HY[Hybrid: RRF + MMR<br/>calibrated to one evidence scale]
        KB --> BM --> HY
        IDX --> DN --> HY
    end
    T1 -- "English paraphrase" --> HY
    QE --> DN
    HY -. "dense unavailable: lexical,<br/>reason in the trace" .-> BM
    HY --> RR

    CAT[(580 real tracks<br/>data/songs.csv<br/>HF Spotify snapshot)] --> A
    RR -- "context cards" --> PLAN[Profile: explicit intent<br/>+ card genres, moods, ranges]
    PLAN -- "off-topic / no signal" --> NS[Refuse to guess]
    REF --> A

    subgraph AGENT["Agent loop: src/agent.py"]
        A[Act: score and rank]
        D[Variety re-rank]
        V{Verify: confidence = fit x evidence<br/>conflict, unsatisfiable,<br/>mixed context, exclusion only}
        RV[Revise: relax genre weight]
        A --> D --> V
        V -- "conflict" --> RV --> A
    end
    PLAN --> A

    V --> EXP
    EXP --> T2[["T2 testing checkpoint<br/>validate each explanation:<br/>names its track, under 70 words,<br/>cites only its own card,<br/>no number absent from its facts"]]
    T2 -. "a pick fails validation" .-> TPL[Template explanation]
    V -- "relaxed / unsatisfiable /<br/>low confidence / distress" --> H3
    H3[["H3 human checkpoint<br/>needs_review flag + plain-language<br/>message explaining what was bent"]]
    T2 --> OUT([Top-k tracks, explanations,<br/>confidence, trace])
    TPL --> OUT
    H3 --> OUT
    OUT --> UI[Streamlit UI<br/>Deezer cover art + 30 s previews<br/>src/enrich.py]
    UI --> H4[["H4 human checkpoint<br/>full agent trace, confidence,<br/>which parser and retriever ran"]]
    H4 -. "follow-up" .-> U

    subgraph TESTING["T3/T4 offline reliability gates (CI-style, no network)"]
        T3[["T3 pytest: 426 tests<br/>against a fake Gemini SDK,<br/>incl. a fully hijacked model"]]
        T4[["T4 src/evaluate.py: behaviour cases,<br/>retrieval ablation on 44 labelled queries,<br/>26-attack injection suite;<br/>tools/mutate.py: 67 planted bugs"]]
    end
    T3 -.->|gates| AGENT
    T4 -.->|gates| RAG

    classDef human fill:#1f3a1f,stroke:#7ddc1f,stroke-width:2px,color:#eaffea
    classDef testing fill:#1f2a3a,stroke:#4fa3ff,stroke-width:2px,color:#eaf3ff
    class H1,H2,H3,H4 human
    class T1,T2,T3,T4 testing
```

**How to read it.** A new request passes the guardrails. When Gemini is available it reads the request into a schema-constrained intent; the validator throws away anything outside the catalog's own vocabulary, and if Gemini is down, rate-limited or the input was flagged, the original rule-based parser takes over. Hybrid retrieval finds the listening contexts that match (BM25 and dense embeddings, fused with Reciprocal Rank Fusion and diversified with MMR), on one calibrated evidence scale, so the agent's thresholds mean the same thing in every mode. The agent ranks the 580-track catalog, re-ranks for variety and verifies its own answer; anything it had to relax or could not satisfy is flagged for a human. Gemini then explains every pick in one call, each explanation validated against its own facts. Follow-ups such as "more upbeat" edit the previous profile rather than starting over. Every Gemini call is cached on disk, so a demo works on a tight free-tier quota and the evaluation is reproducible.
