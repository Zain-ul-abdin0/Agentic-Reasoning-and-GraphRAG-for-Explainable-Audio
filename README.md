# Explainable Audio Biomarkers for Depression Screening

This repository contains the implementation for my master's thesis, *Agentic Reasoning and GraphRAG for Explainable Audio Biomarkers for Depression Detection*, at the University of Siegen.

The project starts with an interview recording and produces a report showing the extracted measurements, the rules they matched and the graph paths used to interpret those matches. Reports can be written for a psychologist or a patient. Gemma, running locally through Ollama, can add a narrative, but it does not make the screening decision.

This is a research prototype, not a diagnostic tool. Its reference is a PHQ-8 questionnaire label, not an independent clinical diagnosis. Despite the broader depression/anxiety wording in the original proposal, the implemented calibration and evaluation concern **depression screening**.

## How it works

```text
WAV + optional transcript and COVAREP files
    -> feature extraction
    -> calibrated threshold rules
    -> count-based screening decision
    -> directed paths through the knowledge graph
    -> previous-report context, when supplied
    -> deterministic report + optional local LLM narrative
```

The orchestrator records these stages in an execution trace. It is a coded workflow, not a set of independently trained agents.

The graph is stored as JSON-LD and loaded into NetworkX. It contains 24 nodes and 27 directed edges. Retrieval follows unweighted shortest paths from matched biomarker nodes to screening or follow-up nodes. A shortest path means the fewest edges, not the strongest clinical evidence.

There are 10 active threshold rules. The original graph also retains **High MFCC 1 Mean**, which was disabled after exploratory ablation analysis. A recording is flagged when at least **5 active rules match**. The matched-rule count is not a probability of depression.

### Measurements and their sources

| Input | Measurements used by the rules |
|---|---|
| WAV waveform | RMS amplitude, MFCC means and variability |
| Transcript timestamps and tokens | Participant-turn speech rate and pause ratio |
| COVAREP pitch stream | Mean pitch, pitch variability and a pitch-instability proxy labelled `jitter` in the code |

The active rules cover pause ratio, speech rate, energy, jitter, high and low mean pitch, pitch variability, MFCC 2 variability, MFCC 3 mean and MFCC 4 variability. These are ten rules, not ten independent sources of information.

The transcript is used for timing and word counts, not sentiment analysis. COVAREP is a set of acoustic measurements supplied alongside the recording. This pipeline does not generate a transcript or run the COVAREP toolkit itself.

## Setup

Use Python 3.11 for the documented setup. Run commands from the repository root.

```powershell
git clone https://github.com/Zain-ul-abdin0/Agentic-Reasoning-and-GraphRAG-for-Explainable-Audio.git
cd Agentic-Reasoning-and-GraphRAG-for-Explainable-Audio
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate` instead. The dependency list is not version-pinned, so it does not reproduce an exact historical environment.

The application uses FastAPI, NetworkX, NumPy, SciPy and Praat through `praat-parselmouth`, with a plain HTML/CSS/JavaScript frontend. The current orchestration is implemented directly in Python; it does not require JAX or LangGraph.

### Data

Obtain DAIC-WOZ through its authorised access process. The recordings are not supplied by this repository. Keep recordings and their sidecar files together in a flat directory:

```text
Dataset/
    300_AUDIO.wav
    300_TRANSCRIPT.csv
    300_COVAREP.csv
    ...
```

These filenames are examples, not files guaranteed to be available after cloning. Batch processing searches for `*_AUDIO.wav` directly inside the selected directory, not recursively.

Matching transcript and COVAREP files are discovered automatically. For files elsewhere, pass `--transcript` and `--covarep` explicitly. `--no-auto-sidecars` disables automatic discovery.

Without sidecars, the extractor can use waveform-based fallbacks where available. **A WAV-only run is not equivalent to a sidecar-assisted run.** The fallback measurements are not necessarily compatible with thresholds calibrated on transcript or COVAREP features.

## Run a report

Start without an LLM to check the extraction, rules and graph retrieval:

```powershell
python -m src.pipeline.run_pipeline Dataset/300_AUDIO.wav --llm-provider none --persona psychologist --quiet --output data/local_runs/300_psychologist.json
```

For the patient version:

```powershell
python -m src.pipeline.run_pipeline Dataset/300_AUDIO.wav --llm-provider none --persona patient --quiet --output data/local_runs/300_patient.json
```

The JSON includes measurements, rule findings, aggregate evidence, explanation paths, report text and execution status. `--quiet --output ...` prints the saved location instead of the whole report. Without `--output`, a single-file run prints JSON but does not save it.

Omitting `--max-duration-seconds` processes the full available audio. A duration limit is useful for demonstrations, but it does not currently align all input streams: the COVAREP reader is not truncated to that limit, and a clipped transcript turn retains its full word count. Do not treat a short demo as a validated short-recording screening result.

### Add a local Gemma report

Install Ollama, start its local service and download the model:

```powershell
ollama pull gemma3:1b
```

If the service is not already running, start `ollama serve` in another terminal. Then run:

```powershell
python -m src.pipeline.run_pipeline Dataset/300_AUDIO.wav --llm-provider ollama --llm-model gemma3:1b --llm-timeout 180 --persona psychologist --quiet --output data/local_runs/300_gemma_psychologist.json
```

The default endpoint is `http://localhost:11434`. No API key is used for this local configuration. Model downloads need network access; report generation uses the configured Ollama endpoint. Generation time depends on the machine and model. If it fails or times out, inspect `llm_status`; the deterministic report remains available.

The CLI defaults to Ollama with model `gemma3`, so specify `--llm-provider none` when you do not want generation, or explicitly choose the model you downloaded.

### Previous reports and batch runs

Use `--history path/to/history.json` to provide previous report data. `--simulate-history-weeks 3` creates labelled synthetic history for demonstrations. It is not evidence of a participant's real progress.

```powershell
python -m src.pipeline.run_pipeline --dataset-dir Dataset --llm-provider none --output-dir data/local_runs/batch
```

Batch mode saves `reports.json` and `features.csv` in the selected output directory.

## Frontend

```powershell
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8001
```

Open **http://127.0.0.1:8001/**. If that port is occupied, choose another one. API documentation is available at `/docs`.

| Tab | What it shows |
|---|---|
| Overview | Measurements, matched rules and the aggregate screening result |
| Agents | Processing stages, persona configuration and execution trace |
| Graph | Retrieved paths and their stored source annotations |
| Report | Report text and generation status |
| Evaluation | Participant-level screening results, retrieval comparisons and the automated rubric |

Web reports are saved under `data/month_04_agentic_reporting_engine/web_reports/`. Uploaded files are handled under the corresponding `uploads/` directory. Treat these files as potentially sensitive. Keep the app on localhost; it is a research interface, not a production clinical service.

## Evaluation

### Screening against PHQ-8

Supply your own label CSV. It must contain `Participant_ID` and either `PHQ8_Score` or `PHQ_Score`. Replace the example label path below with your local file; repeat `--label-file` when necessary.

```powershell
python -m src.evaluation.evaluate_month_05 --dataset-dir Dataset --label-file path/to/phq8_labels.csv --simulate-history-weeks 0 --output-dir data/local_runs/evaluation
```

This evaluates recordings with matching labels, with LLM generation disabled. Leave `--limit` and `--max-duration-seconds` unset for all available labelled recordings and full audio. The output includes `evaluation_summary.json`, `participant_results.csv`, retrieval diagnostics and mock-review scores. The shared retrieval results are also written to `data/retrieval_benchmark/results/`.

The saved thesis run used 113 participants:

| Metric | Selected ten-rule policy |
|---|---:|
| Accuracy | 79.6% |
| Sensitivity | 92.9% |
| Specificity | 66.7% |
| Balanced accuracy | 79.8% |
| F1 | 81.9% |
| TP / TN / FP / FN | 52 / 38 / 19 / 4 |

These are **development-sample results**, not held-out performance. Calibration participants overlap with the evaluation sample, and the rule removal and aggregate cutoff were selected using that data. See the [public aggregate summary](data/published_results/historical_screening_summary.json). Participant-level records are retained locally and excluded from the release.

### Biomarker ablation

Ablation reuses saved participant matches; it does not extract the audio again. For results produced by the current five-match policy:

```powershell
python -m src.evaluation.biomarker_ablation --participant-results data/local_runs/evaluation/participant_results.csv --deployed-threshold 5 --output-dir data/local_runs/ablation
```

The script reports single-rule performance, leave-one-rule-out changes and feature-group ablations. It evaluates removal both with a fixed aggregate cutoff and after retuning that cutoff. These answer different questions. The default now follows the current policy (`5`). Use `--deployed-threshold 6` only when analysing saved results from the historical eleven-rule policy.

### Shared retrieval comparison

This experiment can run without recordings or Ollama:

```powershell
python -m src.evaluation.retrieval_benchmark --top-k 3 --output-dir data/local_runs/retrieval
```

The release-v2 pilot uses the same 15 questions, 27 graph-derived chunks, relevance labels and top-three budget for both methods:

| Metric | Graph paths | Lexical TF-IDF |
|---|---:|---:|
| Precision@3 | 0.556 | 0.622 |
| Recall@3 | 0.733 | 0.772 |
| MRR@3 | 0.867 | 0.922 |
| nDCG@3 | 0.759 | 0.794 |

Release cleanup corrected graph descriptions and removed unrelated citations from policy edges, which changed the lexical corpus. The original thesis benchmark and its results are preserved in [historical_v1](data/retrieval_benchmark/historical_v1/README.md). Its TF-IDF scores were 0.600 / 0.739 / 0.889 / 0.762. These are versioned retrieval experiments, not a change to the screening results. The relevance sets remain provisional, not independently reviewed.

TF-IDF has slightly higher point estimates in this pilot. The questions and relevance labels are provisional, AI-assisted development material, not independently reviewed ground truth. Both methods receive seed-node metadata. This tests retrieval within the curated graph representation, not depression prediction, generated-report quality or modern dense/hybrid RAG. See the [comparison and limitations](data/retrieval_benchmark/results/comparison.md).

Some older outputs call TF-IDF a "vector baseline" and show graph path precision of 1.000 beside TF-IDF precision. Those legacy scores use different relevance criteria and are not a valid head-to-head comparison. The frontend now labels **Mock Review** as **Automated Checklist**, not a rating from a psychologist or patient. Release-v2 checklist scoring no longer rewards simulated context or assumes a passed faithfulness check when no result is present; historical checklist scores must not be compared directly with this version.

## Calibration and graph maintenance

Normal inference loads saved thresholds; it does not recalibrate them for each recording. The loader prefers `data/knowledge_graph/siegen_audio_depression_kg.calibrated.jsonld` and falls back to the seed graph if that file is absent.

Calibration code is in `src/calibration/`. It searches thresholds in predefined high/low directions using Youden's J against PHQ-8 >= 10. These are dataset-specific operating thresholds, not universal clinical cutoffs. Calibration changes numerical node attributes; it does not learn the graph's interpretive edges.

```powershell
python -m src.calibration.build_acoustic_feature_matrix --help
python -m src.calibration.calibrate_acoustic_thresholds --help
python -m src.graph.inspect_knowledge_graph --help
```

Use separate output paths when experimenting: the calibration command's defaults can overwrite the calibrated graph and saved calibration files. For an independent experiment, separate participants before fitting thresholds or selecting rules, then freeze those choices before evaluating the test set.

## Limitations worth keeping in mind

- The transcript pause ratio includes time outside participant turns, potentially including interviewer speech. It is not a direct within-speaker silence measurement.
- The COVAREP-derived `jitter` is a frame-level pitch-instability proxy, not conventional cycle-to-cycle jitter. The live graph now states this explicitly; the historical benchmark snapshot retains its original wording solely for reproducibility.
- RMS and MFCC values depend on recording conditions and preprocessing. Calibration and live extraction still need a full numerical compatibility check.
- Missing features reduce the available matches while the aggregate cutoff remains fixed. There is not yet a comprehensive quality gate that returns an explicit insufficient-evidence outcome.
- Source labels on graph edges make assumptions inspectable; they do not establish that every directed relationship is scientifically validated.
- Report checks are heuristic. A generated narrative can still contain unsupported interpretations, and a local model does not make an output clinically reliable.

## Code layout

```text
src/
  feature_extraction.py   Waveform, transcript and COVAREP measurements
  calibration/            Feature matrices and threshold fitting
  graph/                  JSON-LD loading, validation and rule matching
  rag/                    Path retrieval, evidence scoring and report generation
  agentic/                Orchestration and previous-report context
  pipeline/               Single-file and batch command-line entry points
  evaluation/             Screening metrics, ablation and retrieval comparison
  web/                    FastAPI backend and browser interface
data/
  knowledge_graph/        Seed and calibrated graph files
  retrieval_benchmark/    Questions, relevance labels and saved pilot results
  month_*/                Historical experiment and report artefacts
tests/                    Retrieval benchmark tests
```

The `month_*` directory names reflect the thesis development schedule; they are retained so existing scripts and result references still resolve.

For frontend evaluation, place authorised label CSVs in `data/private/labels/`, or set `DAIC_WOZ_LABEL_FILES` to your local paths (separated by `;` on Windows, `:` on Linux/macOS) before starting the server. CLI commands also accept `--label-file`. Old local label directories are supported only when the files actually exist.

## Release and private files

Recordings, uploads, individual reports, feature matrices, temporary outputs and unfinished formative-review material are excluded from Git. Public results contain aggregate statistics and the non-participant retrieval benchmark. Simulated history defaults to zero and must be explicitly requested.

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the cleanup and historical-result boundaries. Excluding files from the current tree does not remove them from earlier commits. Review repository history and dataset permissions before publishing; history cleanup, if needed, is a separate operation.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

The current tests cover retrieval metrics, benchmark validation, reproducibility and the shared evaluation setup. They do not establish clinical validity or full end-to-end extraction correctness.

## Author

Zain ul Abdin, University of Siegen.

Before sharing a fork or experiment output, review it for participant information, local file paths and dataset access restrictions. Do not publish recordings, transcripts or participant-level reports without the required permission.
