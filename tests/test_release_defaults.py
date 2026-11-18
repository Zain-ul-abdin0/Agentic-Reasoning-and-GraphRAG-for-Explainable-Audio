import os
from pathlib import Path
import unittest
from unittest.mock import patch

from src.calibration.label_paths import default_label_files
from src.evaluation.biomarker_ablation import parse_args as ablation_args
from src.evaluation.evaluate_month_05 import parse_args as evaluation_args
from src.evaluation.mock_review import score_report_quality
from src.evaluation.retrieval_benchmark import evaluate_retrieval, portable_graph_path
from src.graph.knowledge_graph import build_graph
from src.rag.evidence_scoring import score_evidence

ROOT=Path(__file__).resolve().parents[1]


class ReleaseDefaultsTests(unittest.TestCase):
    def test_public_graph_path_does_not_leak_local_root(self):
        self.assertEqual(portable_graph_path(build_graph()), 'data/knowledge_graph/siegen_audio_depression_kg.calibrated.jsonld')

    def test_default_evaluation_has_no_simulated_history(self):
        with patch('sys.argv',['evaluate']):
            self.assertEqual(evaluation_args().simulate_history_weeks,0)

    def test_ablation_default_matches_current_policy(self):
        with patch('sys.argv',['ablation']):
            self.assertEqual(ablation_args().deployed_threshold,5)

    def test_private_label_paths_can_be_configured(self):
        with patch.dict(os.environ,{'DAIC_WOZ_LABEL_FILES':os.pathsep.join(['labels/a.csv','labels/b.csv'])}):
            self.assertEqual(default_label_files(),[Path('labels/a.csv'),Path('labels/b.csv')])

    def test_missing_faithfulness_is_not_assumed_supported(self):
        self.assertEqual(score_report_quality({},'patient')['clinical_caution'],3)

    def test_simulation_does_not_improve_checklist(self):
        blank=score_report_quality({},'patient')
        synthetic=score_report_quality({'recursive_context':{'enabled':True,'simulated':True}},'patient')
        self.assertEqual(blank['recursive_context_use'],synthetic['recursive_context_use'])

    def test_historical_metadata_is_not_current_prediction(self):
        report=score_evidence([],threshold=6)
        self.assertFalse(report['screen_positive'])
        self.assertTrue(report['calibration']['not_recomputed_for_this_recording'])
        self.assertFalse(report['calibration']['historical_cutoff_matches_current_cutoff'])

    def test_live_rules_and_topology_unchanged(self):
        old=build_graph(ROOT/'data/retrieval_benchmark/historical_v1/graph.jsonld')
        new=build_graph()
        self.assertEqual(set(old.edges),set(new.edges))
        self.assertEqual(set(old.nodes),set(new.nodes))
        for node in old.nodes:
            for key in ('threshold','direction','feature_key','feature_index','active_for_threshold_detection'):
                self.assertEqual(old.nodes[node].get(key),new.nodes[node].get(key))

    def test_historical_benchmark_still_reproduces_metrics(self):
        folder=ROOT/'data/retrieval_benchmark/historical_v1'
        result=evaluate_retrieval(folder/'questions.json',3,build_graph(folder/'graph.jsonld'))
        self.assertAlmostEqual(result['methods']['tfidf']['precision_at_k'],.6)

    def test_policy_edges_are_not_presented_as_literature_evidence(self):
        graph=build_graph()
        edges=[e for _,_,e in graph.edges(data=True) if e.get('evidence_type')=='design/policy']
        self.assertEqual(len(edges),4)
        self.assertTrue(all(not e['sources'] for e in edges))
