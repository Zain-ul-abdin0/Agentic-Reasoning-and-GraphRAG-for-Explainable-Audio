# Shared-criteria retrieval evaluation

Provisional AI-assisted draft relevance labels; not independently reviewed or held out

15 questions; 27 chunks; k=3.

| Metric | Graph | TF-IDF |
|---|---:|---:|
| precision_at_k | 0.5556 | 0.6000 |
| recall_at_k | 0.7333 | 0.7389 |
| mrr_at_k | 0.8667 | 0.8889 |
| ndcg_at_k | 0.7587 | 0.7624 |

## Limits
- This is a pilot on graph-derived text chunks, not original paper passages or clinical truth.
- Questions and relevance labels were authored during development; independent blinded review is still required.
- Labels express intended relevance within the current graph, not scientific validity of its relationships. Unlisted chunks are provisionally judged nonrelevant.
- The shared corpus comes from the graph, so this comparison is conditional on that representation and may favour graph retrieval.
- Both methods receive known seed-node metadata; results do not measure entity linking or open-ended document search.
- The baseline is lexical TF-IDF, not dense-vector or hybrid RAG. Report generation and clinical usefulness are not evaluated.
- Small, related questions limit uncertainty estimates. The bootstrap intervals are descriptive, not independent validation.

Historical path-structure precision is a separate diagnostic, not a relevance score.
