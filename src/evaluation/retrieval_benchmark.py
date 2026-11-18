"""Paired retrieval evaluation; relevance labels never enter the retrievers."""

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from src.evaluation.vector_baseline import TfidfIndex, build_edge_corpus
from src.graph.knowledge_graph import build_graph, terminal_risk_nodes
from src.rag.graph_rag import get_all_explanations

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = BASE_DIR / "data/retrieval_benchmark/questions.json"
DEFAULT_OUTPUT = BASE_DIR / "data/retrieval_benchmark/results"
METRICS = ("precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k")


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serializable_corpus(graph):
    return sorted([
        {**item, "related_nodes": sorted(item["related_nodes"])}
        for item in build_edge_corpus(graph)
    ], key=lambda item: item["chunk_id"])


def validate_benchmark(benchmark, graph, corpus):
    ids = {item["chunk_id"] for item in corpus}
    if not benchmark.get("questions"):
        raise ValueError("The benchmark must contain questions with explicit relevance labels.")
    if benchmark.get("corpus_sha256") != fingerprint(corpus):
        raise ValueError("Corpus differs from the frozen benchmark. Review labels before updating its hash.")
    seen = set()
    for q in benchmark["questions"]:
        qid = q.get("id")
        if not qid or qid in seen:
            raise ValueError("Question IDs must be nonempty and unique.")
        seen.add(qid)
        if not q.get("question", "").strip() or not q.get("rationale", "").strip():
            raise ValueError(f"{qid}: question and annotation rationale are required.")
        seeds = q.get("seed_nodes", [])
        if not seeds or len(set(seeds)) != len(seeds) or any(s not in graph for s in seeds):
            raise ValueError(f"{qid}: seed nodes must be unique existing nodes.")
        relevant = q.get("relevant_chunk_ids", [])
        if not relevant or len(set(relevant)) != len(relevant) or not set(relevant) <= ids:
            raise ValueError(f"{qid}: relevance labels must be unique known corpus IDs.")


def query_text(request, graph):
    # Both systems receive the same question and upstream entity identifiers.
    return " ".join([
        request["question"],
        *[f"{node} {graph.nodes[node].get('name', '')}" for node in request["seed_nodes"]],
    ])


def retrieve_graph_chunks(request, graph, corpus, top_k):
    """Project shortest paths adapted to unique edge chunks, nearest hop first.

    The graph route is intentionally the existing seed-driven retriever, not a
    question-aware oracle. Labels and intended target answers are never supplied.
    """
    by_id = {item["chunk_id"]: item for item in corpus}
    ranks = {}
    for seed in request["seed_nodes"]:
        for explanation in get_all_explanations(graph, seed, sorted(terminal_risk_nodes(graph))):
            path = explanation["path"]
            for hop, (source, target) in enumerate(zip(path, path[1:])):
                chunk_id = f"{source}->{target}"
                ranks[chunk_id] = min(hop, ranks.get(chunk_id, hop))
    selected = sorted(ranks, key=lambda chunk_id: (ranks[chunk_id], chunk_id))[:top_k]
    return [by_id[chunk_id] for chunk_id in selected]


def retrieval_metrics(retrieved_ids, relevant_ids, top_k):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Answerable benchmark questions require at least one relevant item")
    unique = list(dict.fromkeys(retrieved_ids))[:top_k]
    hits = [int(item in relevant) for item in unique]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(top_k, len(relevant))))
    return {
        "precision_at_k": sum(hits) / top_k,
        "recall_at_k": sum(hits) / len(relevant),
        "mrr_at_k": next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0),
        "ndcg_at_k": dcg / ideal,
        "relevant_retrieved": sum(hits),
        "retrieved_count": len(unique),
    }


def paired_interval(differences, samples=2000, seed=20261107):
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def evaluate_retrieval(benchmark_path=DEFAULT_BENCHMARK, top_k=3, graph=None):
    graph = build_graph() if graph is None else graph
    corpus = serializable_corpus(graph)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= len(corpus):
        raise ValueError(f"top_k must be an integer between 1 and {len(corpus)}")
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    validate_benchmark(benchmark, graph, corpus)
    index = TfidfIndex(corpus)
    rows = []
    for i, question in enumerate(benchmark["questions"]):
        request = {key: question[key] for key in ("question", "seed_nodes")}
        text = query_text(request, graph)
        retrievals = {}
        # Alternate order to avoid always giving one method the first execution.
        methods = ("graph", "tfidf") if i % 2 == 0 else ("tfidf", "graph")
        for method in methods:
            started = perf_counter()
            chunks = (retrieve_graph_chunks(request, graph, corpus, top_k)
                      if method == "graph" else index.search(text, top_k))
            retrievals[method] = (chunks, (perf_counter() - started) * 1000)
        for method in ("graph", "tfidf"):
            chunks, latency = retrievals[method]
            retrieved = [item["chunk_id"] for item in chunks]
            rows.append({
                "query_id": question["id"], "question": question["question"],
                "method": method, "top_k": top_k,
                **retrieval_metrics(retrieved, question["relevant_chunk_ids"], top_k),
                "latency_ms": latency, "retrieved_ids": retrieved,
                "relevant_ids": question["relevant_chunk_ids"],
                "seed_nodes": request["seed_nodes"], "query_text": text,
            })
    scores = {}
    for method in ("graph", "tfidf"):
        own = [row for row in rows if row["method"] == method]
        scores[method] = {metric: mean(row[metric] for row in own)
                          for metric in (*METRICS, "latency_ms")}
    differences = {}
    for metric in METRICS:
        paired = [rows[i][metric] - rows[i + 1][metric] for i in range(0, len(rows), 2)]
        differences[metric] = {"graph_minus_tfidf": mean(paired),
                               "bootstrap_95_interval": paired_interval(paired)}
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_name": benchmark["name"],
        "annotation_status": benchmark["annotation_status"],
        "query_count": len(benchmark["questions"]), "corpus_size": len(corpus),
        "top_k": top_k, "unit": "unique graph-derived edge text chunk",
        "protocol": {
            "same_queries_corpus_qrels_budget": True,
            "query_inputs": "Question text and identical known seed IDs/names; no answer labels",
            "graph_ranking": "Existing shortest paths; unique edges ranked by hop then ID",
            "tfidf_ranking": "TF-IDF cosine similarity, then chunk ID; zero-score hits omitted",
            "precision_denominator": "k, including unfilled retrieval slots",
            "aggregation": "Macro average over distinct questions, not participants",
            "latency": "Single warm-process query time; TF-IDF fitting excluded; descriptive only",
            "interval": "Paired query bootstrap, 2000 replicates, seed 20261107; not external validity",
            "generation": "Not evaluated; no LLM involved",
        },
        "limitations": benchmark["limitations"],
        "provenance": {
            "corpus_sha256": fingerprint(corpus), "benchmark_sha256": fingerprint(benchmark),
            "graph_path": graph.graph.get("source_path"),
        },
        "methods": scores, "paired_differences": differences,
        "rows": rows, "benchmark_snapshot": benchmark, "corpus_snapshot": corpus,
    }


def write_results(result, output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "query_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value
                         for key, value in row.items()} for row in result["rows"])
    lines = ["# Shared-criteria retrieval evaluation", "", result["annotation_status"], "",
             f"{result['query_count']} questions; {result['corpus_size']} chunks; k={result['top_k']}.",
             "", "| Metric | Graph | TF-IDF |", "|---|---:|---:|"]
    for metric in METRICS:
        lines.append(f"| {metric} | {result['methods']['graph'][metric]:.4f} | {result['methods']['tfidf'][metric]:.4f} |")
    lines += ["", "## Limits", *[f"- {item}" for item in result["limitations"]], "",
              "Historical path-structure precision is a separate diagnostic, not a relevance score."]
    (output_dir / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    result = evaluate_retrieval(args.benchmark, args.top_k, build_graph(args.graph))
    write_results(result, args.output_dir)
    print(json.dumps({key: result[key] for key in ("query_count", "top_k", "annotation_status", "methods")}, indent=2))
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
