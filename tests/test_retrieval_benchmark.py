import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation.retrieval_benchmark import (
    DEFAULT_BENCHMARK, METRICS, evaluate_retrieval, fingerprint, paired_interval,
    query_text, retrieval_metrics, serializable_corpus, validate_benchmark, write_results,
)
from src.evaluation.vector_baseline import TfidfIndex
from src.graph.knowledge_graph import build_graph


class RetrievalMetricsTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        result = retrieval_metrics(["a", "x", "b"], ["a", "b", "c", "d"], 3)
        self.assertAlmostEqual(result["precision_at_k"], 2 / 3)
        self.assertEqual(result["recall_at_k"], 0.5)
        self.assertEqual(result["mrr_at_k"], 1)
        self.assertAlmostEqual(result["ndcg_at_k"], 1.5 / (1 + 1 / math.log2(3) + 0.5))

    def test_duplicates_and_short_results_do_not_inflate_precision(self):
        result = retrieval_metrics(["a", "a"], ["a"], 3)
        self.assertEqual(result["retrieved_count"], 1)
        self.assertAlmostEqual(result["precision_at_k"], 1 / 3)
        self.assertEqual(result["recall_at_k"], 1)

    def test_empty_retrieval(self):
        result = retrieval_metrics([], ["a"], 3)
        self.assertTrue(all(result[key] == 0 for key in METRICS))

    def test_bad_metric_inputs(self):
        for ids, k in [([], 3), (["a"], 0)]:
            with self.assertRaises(ValueError):
                retrieval_metrics([], ids, k)

    def test_bootstrap_is_reproducible(self):
        self.assertEqual(paired_interval([0.1, -0.2, 0.3]), paired_interval([0.1, -0.2, 0.3]))
        self.assertEqual(paired_interval([0, 0]), [0, 0])

    def test_tfidf_zero_overlap_abstains(self):
        index = TfidfIndex([{"chunk_id": "a", "text": "pitch voice"}])
        self.assertEqual(index.search("unrelatedxyz", 3), [])


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()
        cls.corpus = serializable_corpus(cls.graph)
        cls.benchmark = json.loads(DEFAULT_BENCHMARK.read_text())

    def test_frozen_corpus_and_labels_are_valid(self):
        validate_benchmark(self.benchmark, self.graph, self.corpus)
        self.assertEqual(len(self.benchmark["questions"]), 15)

    def test_changed_corpus_fails(self):
        changed = copy.deepcopy(self.corpus)
        changed[0]["text"] += " changed"
        with self.assertRaisesRegex(ValueError, "Corpus differs"):
            validate_benchmark(self.benchmark, self.graph, changed)

    def test_invalid_annotation_rejected(self):
        for mode in ("unknown", "empty", "duplicate", "seed"):
            altered = copy.deepcopy(self.benchmark)
            q = altered["questions"][0]
            if mode == "unknown": q["relevant_chunk_ids"] = ["unknown"]
            if mode == "empty": q["relevant_chunk_ids"] = []
            if mode == "duplicate": altered["questions"].append(q)
            if mode == "seed": q["seed_nodes"] = ["unknown"]
            with self.assertRaises(ValueError):
                validate_benchmark(altered, self.graph, self.corpus)

    def test_same_inputs_units_budget_and_scoring(self):
        result = evaluate_retrieval(graph=self.graph)
        corpus_ids = {item["chunk_id"] for item in self.corpus}
        self.assertEqual(len(result["rows"]), 30)
        for i in range(0, 30, 2):
            a, b = result["rows"][i:i+2]
            for key in ("query_id", "question", "seed_nodes", "query_text", "relevant_ids", "top_k"):
                self.assertEqual(a[key], b[key])
            for row in (a, b):
                self.assertLessEqual(len(row["retrieved_ids"]), 3)
                self.assertEqual(len(row["retrieved_ids"]), len(set(row["retrieved_ids"])))
                self.assertTrue(set(row["retrieved_ids"]) <= corpus_ids)
                recalculated = retrieval_metrics(row["retrieved_ids"], row["relevant_ids"], 3)
                for metric in METRICS:
                    self.assertEqual(row[metric], recalculated[metric])

    def test_relevance_labels_never_reach_retriever(self):
        def fake(request, graph, corpus, top_k):
            self.assertEqual(set(request), {"question", "seed_nodes"})
            return []
        with patch("src.evaluation.retrieval_benchmark.retrieve_graph_chunks", side_effect=fake):
            result = evaluate_retrieval(graph=self.graph)
        self.assertEqual(result["methods"]["graph"]["precision_at_k"], 0)

    def test_invalid_budget(self):
        for k in (0, -1, 28, 1.5, True):
            with self.assertRaises(ValueError):
                evaluate_retrieval(top_k=k, graph=self.graph)

    def test_outputs_include_reproducible_snapshots(self):
        result = evaluate_retrieval(graph=self.graph)
        with tempfile.TemporaryDirectory() as folder:
            write_results(result, folder)
            saved = json.loads((Path(folder) / "retrieval_summary.json").read_text())
            self.assertEqual(fingerprint(saved["corpus_snapshot"]), saved["provenance"]["corpus_sha256"])
            self.assertEqual(fingerprint(saved["benchmark_snapshot"]), saved["provenance"]["benchmark_sha256"])
            self.assertTrue((Path(folder) / "query_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
