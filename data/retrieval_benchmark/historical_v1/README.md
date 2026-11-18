# Historical thesis retrieval pilot

This directory freezes the graph, questions and results used by the original retrieval pilot. It contains no participant recordings or participant-level feature tables.

The graph deliberately retains the original descriptions and source strings, including annotations corrected in the release graph. It is an experimental snapshot, not the recommended graph for new reports. Its citations must not be treated as verified clinical support.

Reproduce without overwriting recorded results:

```powershell
python -m src.evaluation.retrieval_benchmark --benchmark data/retrieval_benchmark/historical_v1/questions.json --graph data/retrieval_benchmark/historical_v1/graph.jsonld --top-k 3 --output-dir data/local_runs/historical_retrieval
```

The stored summary's graph-path metadata was made repository-relative during release preparation. The corpus, questions, labels and recorded metric values were not edited. Latency will vary between runs.
