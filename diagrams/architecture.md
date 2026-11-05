# VibeFinder 2.0: System Architecture

```mermaid
flowchart TD
    U([User request<br/>free text]) --> G{Guardrails<br/>src/guardrails.py}
    G -- "empty / non-text" --> RJ[Reject with guidance<br/>issue: invalid_input]
    G -- "cleaned text<br/>(truncated, override text flagged)" --> NEG[Strip negated genres<br/>'not jazz' -> avoid list]

    subgraph RAG["Retrieval (RAG) - src/retrieval.py"]
        KB[(Knowledge base<br/>data/knowledge/*.md<br/>13 context cards)]
        R[TF-IDF retriever<br/>cosine >= 0.12]
        KB --> R
    end
    NEG --> R

    CAT[(Song catalog<br/>data/songs.csv)] --> P
    R -- "top context cards<br/>genres, moods, feature ranges" --> P[Planner / profile builder<br/>src/profile.py<br/>explicit words + card ranges]
    G --> P

    P -- "no usable signal" --> NS[Refuse to guess<br/>issue: no_signal]

    subgraph AGENT["Agent loop - src/agent.py (max 3 iterations)"]
        A[Act: score + rank all songs<br/>src/recommender.py]
        D[Variety re-rank<br/>genre/artist repeats penalised]
        V{Verify<br/>confidence = fit x evidence<br/>conflict? unsatisfiable?}
        RV[Revise: relax genre weight x0.5]
        A --> D --> V
        V -- "conflict" --> RV --> A
    end
    P --> A
    CAT --> A

    V -- "no issues" --> E[Explainer<br/>src/llm.py<br/>template, or Gemini with validation + fallback]
    V -- "relaxed / unsatisfiable / low confidence" --> HR[Flag: needs human review<br/>+ clarifying message]
    HR --> E
    R -. "card guidance grounds each explanation" .-> E
    E --> OUT([Top-k songs + explanations<br/>+ confidence + trace])
    OUT --> H{{Human-in-the-loop:<br/>user confirms or refines request}}
    H -. "refined request" .-> U

    subgraph TESTING["Reliability testing"]
        T1[pytest: 52 unit/behaviour tests]
        T2[src/evaluate.py: 19 checks,<br/>RAG vs no-RAG baseline,<br/>confidence calibration]
        T3[(logs/: traces, eval results,<br/>vibefinder.log)]
    end
    AGENT -. "every step logged" .-> T3
    T2 -. "runs predefined inputs through" .-> U
```

**How to read it.** A request passes the guardrails, then the retriever finds the context cards that best match it. The planner merges those cards (genres, moods, feature ranges) with anything the user typed explicitly into a scoring profile. The agent ranks the catalog, re-ranks for variety, and **verifies its own answer**: it computes a confidence score and checks for conflicts and impossible requests. On a conflict it revises the plan (relaxes the genre weight) and re-ranks. Anything it had to relax, or could not satisfy, is flagged for a human instead of being presented as certain. Testing sits outside the loop: pytest and the evaluation harness drive predefined inputs through the same entry point, and every agent step is written to `logs/`.
