# Month 5 Comparative Analysis

## Evaluation Scope

- Dataset: D:\After Coming to Germany\Study\Thesis\Agentic Reasoning and GraphRAG for Explainable Audio\Dataset
- PHQ-8 clinical reference label: PHQ-8 >= 10
- Participants evaluated: 113
- LLM usage during evaluation: disabled for deterministic batch evaluation

## Depression Screening Metrics

- Accuracy: 0.796
- Sensitivity: 0.929
- Specificity: 0.667
- Balanced accuracy: 0.798
- Precision: 0.732
- F1 score: 0.819
- Confusion matrix: TP=52, TN=38, FP=19, FN=4

## Retrieval Precision

- Graph-RAG mean path precision: 1.000
- Graph-RAG source-backed path ratio: 1.000
- Vector baseline mean precision@k: 0.395

Graph-RAG is evaluated as path retrieval: a retrieved path is relevant when it starts from a matched biomarker node and reaches a clinical target node. The vector baseline retrieves edge chunks using lexical TF-IDF cosine similarity, then checks whether the retrieved chunks are directly connected to the matched biomarker.

## Mock Qualitative Review

- Review type: mock qualitative rubric
- Mean psychologist score: 4.332
- Mean patient score: 4.332

The qualitative review is a deterministic proxy rubric for thesis development. It should be replaced or complemented by human review in the final study.

## Output Files

- summary: `D:\After Coming to Germany\Study\Thesis\Agentic Reasoning and GraphRAG for Explainable Audio\data\month_05_evaluation\evaluation_summary.json`
- participant_results: `D:\After Coming to Germany\Study\Thesis\Agentic Reasoning and GraphRAG for Explainable Audio\data\month_05_evaluation\participant_results.csv`
- retrieval_comparison: `D:\After Coming to Germany\Study\Thesis\Agentic Reasoning and GraphRAG for Explainable Audio\data\month_05_evaluation\retrieval_comparison.csv`
- mock_review_scores: `D:\After Coming to Germany\Study\Thesis\Agentic Reasoning and GraphRAG for Explainable Audio\data\month_05_evaluation\mock_review_scores.csv`
- comparative_analysis: `D:\After Coming to Germany\Study\Thesis\Agentic Reasoning and GraphRAG for Explainable Audio\data\month_05_evaluation\comparative_analysis.md`
