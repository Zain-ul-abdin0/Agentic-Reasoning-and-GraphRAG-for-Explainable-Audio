import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def build_edge_corpus(graph):
    corpus = []

    for source, target, edge in graph.edges(data=True):
        source_node = graph.nodes[source]
        target_node = graph.nodes[target]
        sources = edge.get("sources") or []
        text = " ".join([
            source,
            source_node.get("name", ""),
            source_node.get("clinical_summary", ""),
            edge.get("relation", ""),
            target,
            target_node.get("name", ""),
            target_node.get("description", ""),
            target_node.get("clinical_summary", ""),
            edge.get("evidence_note", ""),
            " ".join(sources),
        ])
        corpus.append({
            "chunk_id": f"{source}->{target}",
            "text": text,
            "source": source,
            "target": target,
            "related_nodes": {source, target},
            "sources": sources,
        })

    return corpus


def _idf(corpus_tokens):
    document_count = len(corpus_tokens)
    frequencies = Counter()

    for tokens in corpus_tokens:
        frequencies.update(set(tokens))

    return {
        token: math.log((1 + document_count) / (1 + count)) + 1
        for token, count in frequencies.items()
    }


def _tfidf(tokens, idf):
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {
        token: (count / total) * idf.get(token, 1.0)
        for token, count in counts.items()
    }


def _cosine(left, right):
    numerator = sum(left.get(token, 0.0) * right.get(token, 0.0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if not left_norm or not right_norm:
        return 0.0

    return numerator / (left_norm * right_norm)


def retrieve_chunks(query, corpus, top_k=3):
    corpus_tokens = [tokenize(item["text"]) for item in corpus]
    idf = _idf(corpus_tokens)
    query_vector = _tfidf(tokenize(query), idf)
    scored = []

    for item, tokens in zip(corpus, corpus_tokens):
        score = _cosine(query_vector, _tfidf(tokens, idf))
        scored.append({**item, "score": score})

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def graph_path_precision(explanations, matched_node_ids):
    if not explanations:
        return None

    relevant = 0
    source_backed = 0

    for explanation in explanations:
        starts_from_matched = explanation.get("start") in matched_node_ids
        has_reasoning_path = len(explanation.get("path") or []) > 1
        if starts_from_matched and has_reasoning_path:
            relevant += 1

        steps = explanation.get("steps") or []
        if any(step.get("sources_to_next") for step in steps):
            source_backed += 1

    return {
        "retrieved": len(explanations),
        "relevant": relevant,
        "precision": relevant / len(explanations),
        "source_backed": source_backed,
        "source_backed_ratio": source_backed / len(explanations),
    }


def compare_vector_baseline(graph, findings, explanations, top_k=3):
    corpus = build_edge_corpus(graph)
    matched_findings = [finding for finding in findings if finding.get("matched")]
    matched_node_ids = {finding["node"] for finding in matched_findings}
    graph_precision = graph_path_precision(explanations, matched_node_ids)
    rows = []

    for finding in matched_findings:
        query = " ".join([
            finding.get("label", ""),
            finding.get("summary", ""),
            str(finding.get("feature_key", "")),
            str(finding.get("value", "")),
        ])
        chunks = retrieve_chunks(query, corpus, top_k=top_k)
        relevant_count = sum(
            1
            for chunk in chunks
            if finding["node"] in chunk["related_nodes"]
        )
        rows.append({
            "matched_biomarker": finding["label"],
            "matched_node": finding["node"],
            "top_k": top_k,
            "retrieved": len(chunks),
            "relevant": relevant_count,
            "precision_at_k": relevant_count / len(chunks) if chunks else 0.0,
            "top_chunks": "; ".join(chunk["chunk_id"] for chunk in chunks),
        })

    return {
        "graph_rag": graph_precision,
        "vector_rows": rows,
        "vector_mean_precision_at_k": (
            sum(row["precision_at_k"] for row in rows) / len(rows)
            if rows else None
        ),
    }

