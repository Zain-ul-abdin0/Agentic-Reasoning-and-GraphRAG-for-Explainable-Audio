# Agentic Reasoning and Graph-RAG for Explainable Audio Biomarkers for Depression Detection

## Current Thesis Report Draft Covering Months 1 to 3

Student: To be assigned  
Programme: Master's Thesis / Block Coursework  
Duration: 1 Semester  
Current stage: Month 3 completed  

## Abstract

Speech-based mental-health analysis has become an active research area because clinical interviews contain more than verbal content. A speaker's pauses, pitch movement, vocal intensity, spectral structure, rhythm, and hesitation patterns can provide indirect evidence about affective state, cognitive effort, and psychomotor activity. Prior work on datasets such as DAIC-WOZ has shown that acoustic and textual features can support depression screening, especially when temporal patterns are modelled. However, many successful systems still behave as black boxes: they may output a depression or anxiety risk score, but they do not clearly explain why a particular acoustic pattern is clinically meaningful. This limits trust and practical clinical usefulness.

This thesis explores a lightweight agentic and Graph-RAG framework for explainable audio biomarker reporting. The work completed so far covers three foundations. First, numerical speech features are mapped into clinical concepts such as vocal instability, speech disruption, reduced expressivity, and spectral shift. Second, these concepts are represented in a portable Clinical Knowledge Graph implemented in JSON-LD and loaded with NetworkX. Third, the retrieved graph evidence is now passed to a local Reporter Agent that can use Gemma through Ollama to generate psychologist- or patient-facing narratives. To avoid relying on arbitrary thresholds, acoustic operating thresholds were calibrated against DAIC-WOZ PHQ-8 labels using PHQ-8 >= 10 as the validated depression screening reference. The calibrated graph currently contains 24 nodes, 27 edges, and 11 biomarker rules.

The current system should be understood as an explainable screening-support prototype rather than a diagnostic tool. Its purpose is not to claim that a single acoustic feature detects depression. Instead, it uses dataset-calibrated acoustic evidence as an entry point for structured reasoning paths, and then uses a local language model only to express those paths in readable form. This report documents the motivation, literature basis, methodology, current implementation, calibrated thresholds, Month 3 Graph-RAG reporting layer, limitations, and next development steps.

## 1. Introduction

Mental-health assessment usually depends on interviews, questionnaires, behavioural observation, and clinical judgement. In many cases, speech is central to that process. A person may describe symptoms directly, but they may also reveal distress through how they speak: long pauses, low vocal energy, flattened intonation, unstable pitch, reduced verbal fluency, or hesitant responses. These behaviours are familiar to clinicians, but computational systems often treat them only as numerical input features. The result is a gap between machine prediction and clinical interpretation.

Recent machine-learning models can detect depression-related patterns from speech and text with promising accuracy. Work on DAIC-WOZ and related corpora has shown that acoustic features, interview transcripts, and temporal sequence models can help estimate depression severity. However, accuracy alone is not enough in a mental-health context. A psychologist needs to inspect the reasoning behind a system's output. A patient needs language that is careful, understandable, and non-stigmatizing. A supervisor or evaluator needs to know whether the system is grounded in evidence or simply generating plausible explanations.

The central problem addressed in this thesis is therefore explainability. A conventional model may output a score, for example "high depression risk." The more clinically useful question is: what evidence contributed to that assessment, what does the evidence mean, and how should it be interpreted cautiously? In speech-based systems this is especially important because individual acoustic features are not diagnoses. A high pause ratio can reflect depression, anxiety, fatigue, language difficulty, interviewer style, or recording artefacts. A low-energy voice may reflect reduced expressivity, microphone distance, or simple tiredness. A responsible system must preserve this uncertainty.

This thesis proposes an agentic Graph-RAG approach. The word "agentic" refers to a modular pipeline in which different components perform different roles. An Analyzer extracts features, a mapping layer translates those features into biomarker concepts, a graph retriever searches for relevant clinical reasoning paths, and a Reporter Agent turns the retrieved evidence into a narrative suitable for different audiences. The Graph-RAG component differs from standard vector retrieval because it retrieves structured relationships rather than only text chunks. Instead of retrieving a paragraph that mentions jitter or pauses, it can retrieve a path such as:

High Pause Ratio -> Speech Disruption -> Difficulty Concentrating -> PHQ-8 Depression Severity -> Clinical Review

This path is not a diagnosis. It is an explanation trace. It shows how a numerical feature becomes a clinical concept and how that concept connects to screening or follow-up.

The first three months of the project have focused on moving from static conceptual design to a working local Graph-RAG prototype. Month 1 produced the feature-to-concept mapping: audio features such as jitter, pause ratio, speech rate, energy, pitch variability, and MFCC summaries were mapped to clinical concepts. Month 2 implemented the lightweight knowledge graph and added data-driven threshold calibration. Month 3 connected threshold-matched biomarkers to graph retrieval and a local Reporter Agent using Gemma through Ollama. The calibration step is important because the first prototype used heuristic threshold values. Those values made the system executable, but they were not scientifically strong enough. The updated version now calibrates acoustic thresholds from DAIC-WOZ participant data using PHQ-8 >= 10 as the validated depression screening label.

The research question guiding the current stage is:

How can acoustic biomarkers from clinical interview speech be represented in a lightweight knowledge graph so that they support traceable, clinically cautious explanations rather than black-box predictions?

Several sub-questions follow from this:

1. Which acoustic biomarkers are useful enough to include in the first graph?
2. How can numerical features be translated into clinical concepts without overclaiming?
3. How can thresholds be justified using dataset calibration rather than arbitrary assumptions?
4. How can graph paths support later multi-persona reporting?

The rest of this report describes the current answer to these questions.

## 2. Literature Review

### 2.1 Speech Biomarkers in Depression and Anxiety Research

Speech is a complex behavioural signal. It depends on language planning, motor control, emotional state, respiratory control, and social context. Depression and anxiety can affect these processes in different ways. Depression has often been associated with psychomotor slowing, reduced expressivity, lower vocal energy, monotonic speech, slower speaking rate, and increased pause duration. Anxiety may appear through restlessness, arousal, tension, vocal instability, hesitation, or irregular prosody. These associations are not simple rules, but they provide a clinical motivation for analysing speech in mental-health assessment.

Cummins et al. (2015) provide an important review of depression and suicide-risk assessment using speech analysis. Their review is valuable because it connects acoustic features with possible clinical interpretations. For example, reduced pitch variability may relate to emotional blunting, while pauses and rhythm changes may relate to cognitive load or psychomotor slowing. This type of reasoning supports the feature-to-concept mapping used in this thesis. The system does not treat pitch or pause values as isolated numbers; it maps them to concepts that are easier to interpret clinically.

Low, Bentley, and Ghosh (2020) reviewed automated psychiatric assessment using speech and highlighted features such as MFCCs, jitter, shimmer, speaking rate, silence duration, intensity, and harmonicity. A key lesson from their review is that no single acoustic feature is sufficient for reliable psychiatric detection. This point is central to the thesis design. The knowledge graph must not imply that one feature diagnoses depression or anxiety. Instead, each feature should be treated as weak evidence that can support a broader explanation when combined with other evidence.

Al Hanai, Ghassemi, and Glass (2018) used the DAIC-WOZ dataset to detect depression with audio and text sequence modelling. Their work is especially relevant because it shows the value of temporal patterns in clinical interviews. Depression cues are not always visible in static averages alone; they can appear in pauses, response timing, and changes across an interview. This thesis currently uses participant-level summaries, but the agentic architecture is designed so that temporal summaries can be added later.

Scherer and colleagues have also contributed important work on automatic behavioural descriptors for psychological disorder analysis. Their studies support the idea that nonverbal and paralinguistic behaviours can be represented as computational descriptors. This is directly related to the graph design, where acoustic markers become graph nodes rather than remaining hidden inside a classifier.

### 2.2 DAIC-WOZ and PHQ-8 Labels

The DAIC-WOZ dataset is one of the most widely used corpora for depression detection from clinical interviews. It contains recordings of participants interacting with a virtual interviewer, along with transcripts and questionnaire-based depression labels. For this project, DAIC-WOZ is useful because it provides both speech behaviour and PHQ-8 depression scores.

The PHQ-8 is a clinically used screening measure derived from the Patient Health Questionnaire family. In DAIC-WOZ and related depression-detection work, PHQ-8 >= 10 is commonly used as the threshold for clinically relevant depressive symptoms. This threshold is not an acoustic threshold. It is a validated questionnaire-based screening label. In this project, it serves as the reference label for calibrating acoustic operating thresholds.

This distinction is essential. A threshold such as PHQ-8 >= 10 has clinical screening validity. A threshold such as jitter > 0.13 or pause ratio > 0.34 does not have universal clinical validity. The acoustic thresholds in this project are therefore not copied from a clinical manual. They are calibrated from the dataset using the validated PHQ-8 label. This makes the method more defensible because it does not pretend that one acoustic cutoff applies universally across microphones, speakers, languages, and preprocessing methods.

### 2.3 Explainability in Healthcare AI

Explainability is not only a technical preference in healthcare; it is a practical and ethical requirement. A model that influences mental-health screening must allow users to understand the basis of its output. Tjoa and Guan (2021) and Holzinger et al. argue that medical AI requires transparency, traceability, and meaningful human oversight. These requirements are even stronger in mental health because predictions may affect patient trust, stigma, follow-up decisions, and clinical judgement.

Feature-attribution methods such as LIME and SHAP are useful because they can indicate which variables influenced a model prediction. However, they are not sufficient for the main goal of this thesis. If SHAP says that pause ratio had high importance, a clinician still needs to understand what pause ratio means and why it might matter. The thesis therefore treats attribution and semantic reasoning as different layers. Attribution can identify influential features, while the knowledge graph explains those features through clinical concepts.

### 2.4 RAG, Graph-RAG, and Hallucination Risk

Retrieval-Augmented Generation (RAG) was introduced to improve the factual grounding of language models by retrieving external knowledge before generation. This is useful because language models can produce fluent but unsupported statements. In a clinical setting, unsupported generation is risky. Ji et al. (2023) discuss hallucination in natural language generation and show why grounding is important when factuality matters.

Standard RAG usually retrieves text chunks. This can help when the answer is contained in a passage, but it is less natural when the explanation depends on a chain of relationships. Graph-RAG retrieves entities and relationships, which makes it better suited to reasoning paths. For this thesis, the relevant explanation is not a single paragraph. It is a path from acoustic evidence to clinical concept to screening context to recommendation.

The Graph-RAG layer is therefore a way to reduce free-form speculation. The Reporter Agent should not simply invent an explanation for a feature. It should receive a graph path and generate a cautious narrative grounded in that path.

### 2.5 Knowledge Graphs in Healthcare

Knowledge graphs are useful in healthcare because they represent structured relationships among symptoms, biomarkers, diagnoses, interventions, and clinical concepts. Healthcare knowledge graph surveys describe their role in traceability, decision support, semantic reasoning, and interpretable medical AI. This thesis adopts a lightweight version of that idea. It does not attempt to build a complete psychiatric ontology. Instead, it creates a focused graph around the acoustic biomarkers needed for the current system.

The lightweight design is intentional. A master's thesis project must remain feasible, inspectable, and easy to modify. JSON-LD provides a portable graph representation, and NetworkX provides graph traversal without requiring a heavy graph database. This gives enough structure for Graph-RAG while keeping the implementation accessible.

### 2.6 Local LLM Reporting, Agentic Retrieval, and Month 3 Graph-RAG

The Month 3 milestone moves the project from graph construction toward a functioning Graph-RAG reporting pipeline. This stage is motivated by the broader RAG literature. Lewis et al. (2020) introduced retrieval-augmented generation as a way to combine parametric memory inside a language model with external non-parametric knowledge. Their work is important for this thesis because it supports the idea that generation should be grounded in retrieved evidence rather than relying only on what the model has memorised during pretraining.

Graph-RAG extends this idea by making the retrieved context relational rather than only textual. The Microsoft GraphRAG work argues that standard vector-based RAG can struggle when a task requires connecting information across entities and relationships. For the current thesis, this is exactly the problem: the system does not only need to retrieve a sentence about pause ratio. It needs to connect a detected acoustic biomarker to a clinical concept, a symptom-level interpretation, a screening construct, and a follow-up recommendation. Graph traversal is therefore more aligned with the explanation task than unstructured text retrieval.

Agentic reasoning literature also supports the Month 3 design. ReAct shows that language models can benefit when reasoning is combined with external actions and observations. In this thesis, the action is not web search or tool use in a general environment; it is constrained graph retrieval. The Analyzer extracts acoustic observations, the Retriever obtains graph paths, and the Reporter Agent converts those paths into a narrative. This modular design is useful because each step can be inspected separately. A supervisor can inspect the feature values, the threshold decisions, the graph paths, and the generated report.

Self-RAG is also relevant because it argues that retrieval should be used to improve factuality and that generated responses should be checked against retrieved evidence. The present implementation is simpler than Self-RAG because it does not train a model to critique its own retrieval. However, it adopts the same safety principle: generation should be conditioned on external evidence. In the current prototype, the prompt explicitly instructs the local model to use only the supplied biomarkers and graph paths, and not to infer a diagnosis.

The local language model used in Month 3 is Gemma through Ollama. Gemma is a family of lightweight open models released by Google, and Gemma 3 is designed for text generation and reasoning tasks. Ollama provides a local runtime and HTTP API for running models on the user's machine. This fits the thesis because mental-health data should be handled cautiously, and local inference avoids sending audio-derived findings to a remote model provider. In the current implementation, Ollama exposes the `/api/generate` endpoint at `http://localhost:11434`, and the Reporter Agent sends a graph-grounded prompt to the local Gemma model.

The Month 3 literature also strengthens the safety argument. Ji et al. (2023) show that natural language generation systems can hallucinate unsupported content. In a clinical setting this is especially dangerous because a fluent but unsupported sentence can look authoritative. The current design reduces this risk in three ways. First, it keeps acoustic rule matching separate from language generation. Second, it gives the model retrieved graph paths instead of asking it to reason from memory alone. Third, it uses explicit prompt constraints: the model must describe the output as screening support and must not diagnose depression.

## 3. Research Gap

The literature shows that speech-based depression detection is technically possible, but it also reveals a gap. Many systems focus on classification accuracy and do not explain why a speech pattern matters clinically. At the same time, conventional RAG systems retrieve text but do not necessarily provide structured reasoning over clinical relationships.

The research gap can be stated as follows:

Existing depression and anxiety detection systems can identify acoustic risk patterns, but they often fail to explain how specific acoustic biomarkers correspond to clinically meaningful interpretations. Conventional RAG systems can ground generated text in retrieved documents, but they do not naturally represent structured clinical reasoning over biomarker relationships. There is therefore a need for a lightweight, graph-grounded, agentic framework that translates acoustic biomarkers into traceable clinical narratives.

This thesis addresses the gap by combining dataset-calibrated acoustic biomarkers with a Clinical Knowledge Graph and a local Reporter Agent. The current contribution is not a final diagnostic model. It is a functioning foundation for explainable audio reasoning and graph-grounded report generation.

## 4. Methodology

### 4.1 Overview

The methodology has five main parts:

1. Extract participant-level acoustic and conversational features.
2. Calibrate feature thresholds using PHQ-8 labels.
3. Represent biomarkers and clinical concepts in a knowledge graph.
4. Retrieve graph paths from matched biomarkers to screening, risk, and follow-up nodes.
5. Generate audience-specific reports using deterministic summaries and an optional local LLM.

This workflow separates prediction evidence from explanation. The acoustic features provide measurable evidence. PHQ-8 provides the validated screening reference. The calibration step converts feature distributions into dataset-specific operating thresholds. The graph converts threshold crossings into semantic reasoning paths. The Reporter Agent then converts those retrieved paths into readable language while preserving the non-diagnostic framing.

### 4.2 Feature Extraction

The current implementation extracts features from three DAIC-WOZ sources: transcripts, WAV audio files, and COVAREP files. Transcript timing is used to derive pause ratio and speech rate. WAV audio is used to compute vocal energy and MFCC statistics. COVAREP F0 values are used to compute pitch mean, pitch variability, and a jitter proxy based on F0 instability.

Pause ratio is calculated from participant speaking turns. The system measures the total time span between the first and last participant response, the total duration of participant speech, and the estimated non-speaking time within that span. The pause ratio is then the non-speaking time divided by the total participant response span. This feature captures hesitation and silence patterns during the interview.

Speech rate is calculated as participant word count divided by participant speech duration. It is expressed as words per second. This is a simple but interpretable measure of verbal fluency and speech tempo.

Energy is calculated from the audio waveform as root mean square amplitude. Because raw energy is sensitive to recording conditions, it is treated cautiously and calibrated within the dataset rather than interpreted as a universal clinical value.

MFCC features are extracted from the waveform using a lightweight signal-processing implementation. The current system records the means and standard deviations of the first few MFCC coefficients. These features represent spectral shape and variability. They are useful for modelling but less directly interpretable than pause ratio or speech rate, so the graph maps them to the broader concept of spectral shift.

Pitch mean and pitch standard deviation are calculated from COVAREP F0 values. Positive F0 values are treated as voiced frames. A jitter proxy is computed as the average relative frame-to-frame period change. This is not a clinical voice-disorder jitter measurement from a laryngology protocol, so it is labelled as an operational acoustic instability feature rather than a universal diagnostic marker.

### 4.3 Threshold Calibration

The initial prototype contained hardcoded acoustic thresholds. These were useful for testing but not strong enough for thesis defence. The updated method calibrates thresholds using PHQ-8 labels from the DAIC-WOZ documentation.

Participants are divided into two groups:

PHQ-8 >= 10: positive depression-screening class  
PHQ-8 < 10: negative depression-screening class  

For every acoustic biomarker, the calibration script tests possible threshold values from the observed dataset distribution. For a high-direction marker, values above the threshold are treated as positive. For a low-direction marker, values below the threshold are treated as positive. For every candidate threshold, the script calculates true positives, true negatives, false positives, false negatives, sensitivity, specificity, Youden's J statistic, and balanced accuracy.

Youden's J is calculated as:

J = sensitivity + specificity - 1

The selected threshold is the value that maximizes Youden's J. This chooses the operating point that best balances sensitivity and specificity for that single feature. It is important to emphasize that this does not make the feature diagnostic. It only gives a dataset-specific operating threshold for use in the explanation pipeline.

### 4.4 Calibrated Thresholds

The calibrated threshold table is shown below. These values were computed from the currently available DAIC-WOZ data in the project workspace.

| Biomarker | Feature | Direction | Threshold | N | Sensitivity | Specificity | Balanced Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| High Jitter | jitter | high | 0.131945 | 111 | 0.109 | 0.982 | 0.546 |
| High Pause Ratio | pause_ratio | high | 0.339809 | 108 | 0.942 | 0.196 | 0.569 |
| Low Speech Rate | speech_rate | low | 3.837305 | 108 | 0.942 | 0.143 | 0.543 |
| Low Vocal Energy | energy | low | 0.004403 | 111 | 0.273 | 0.893 | 0.583 |
| High Pitch Variability | pitch_std | high | 42.658337 | 111 | 0.764 | 0.554 | 0.659 |
| High Mean Pitch | pitch_mean | high | 147.442277 | 111 | 0.727 | 0.571 | 0.649 |
| Low Mean Pitch | pitch_mean | low | 122.941355 | 111 | 0.145 | 0.929 | 0.537 |
| High MFCC 1 Mean | mfcc_mean_0 | high | -85.899466 | 111 | 0.582 | 0.464 | 0.523 |
| High MFCC 2 Variability | mfcc_std_1 | high | 4.798455 | 111 | 0.691 | 0.446 | 0.569 |
| Low MFCC 3 Mean | mfcc_mean_2 | low | -0.375294 | 111 | 0.800 | 0.339 | 0.570 |
| High MFCC 4 Variability | mfcc_std_3 | high | 2.556283 | 111 | 0.164 | 0.929 | 0.546 |

The table also reveals an important scientific point. The balanced accuracies are modest. This is not a failure of the thesis idea; it supports a core claim from the literature: individual acoustic features are weak and context-dependent when used alone. The system should therefore not use them as standalone diagnostic rules. Their role is to provide interpretable evidence for graph-based explanation.

### 4.5 Knowledge Graph Construction

The Clinical Knowledge Graph is implemented in JSON-LD and loaded into NetworkX. The graph contains nodes of the following types:

- Biomarker
- ClinicalConcept
- Symptom
- ScreeningConstruct
- Risk
- Intervention

The current calibrated graph contains 24 nodes and 27 edges. It includes 11 active biomarker rules. The graph is intentionally small and inspectable. This is suitable for the current stage because the goal is not to build a full medical ontology, but to create a focused reasoning structure for explainable audio biomarkers.

Edges use cautious relationship labels such as indicates, may_support, contributes_to, estimates, recommends, and may_benefit_from. The language is deliberately conservative. For example, the graph does not say "high pause ratio causes depression." It says that high pause ratio indicates speech disruption, which may support concepts such as difficulty concentrating or clinical review. This protects the system from making unsupported clinical claims.

### 4.6 Graph-RAG Retrieval

Once a biomarker threshold is crossed, the system uses graph traversal to retrieve explanation paths. A path starts from the detected biomarker node and ends at a target node such as depression screening risk, clinical review, PHQ-8 follow-up, supportive follow-up, or the PHQ-8 depression severity screening construct. The retrieval currently uses shortest-path logic in NetworkX.

For example, if a participant crosses the high pause ratio threshold, a possible path is:

High Pause Ratio -> Speech Disruption -> Difficulty Concentrating -> Screening Context

This path gives the Reporter Agent structured evidence. Instead of generating an explanation from memory, the model can be prompted with explicit graph context.

### 4.7 Local LLM Reporter Agent

The Month 3 implementation adds a local LLM-based Reporter Agent. The Reporter Agent receives extracted biomarkers, matched calibrated rules, and retrieved graph paths. It then builds a prompt for a specific audience. The psychologist persona uses concise clinical language and preserves technical terms such as biomarker, threshold, and graph path. The patient persona uses simpler language, avoids alarming phrasing, and clearly states that the result is not a diagnosis.

Gemma is accessed locally through Ollama. The code sends a prompt to Ollama's local HTTP endpoint and receives a generated narrative. If Ollama is not running, if the model is not installed, or if the response times out, the system does not fail. It records the error in `llm_status` and still returns a deterministic graph-grounded report. This fallback is important for demonstration reliability and clinical safety because the core evidence does not depend on the language model.

The LLM is therefore not the diagnostic engine. The evidence chain remains:

Audio feature -> calibrated threshold rule -> biomarker node -> graph path -> cautious report

The local LLM is used only for narrative generation after the graph has already selected the evidence.

## 5. System Implementation

### 5.1 Project Structure

The current implementation is organized into four main areas:

- `src/feature_extraction.py`: WAV loading and audio feature extraction for single-file reporting.
- `src/graph`: knowledge graph loading, validation, feature mapping, and inspection.
- `src/rag`: graph retrieval, local LLM client, and Reporter Agent generation.
- `src/calibration`: dataset feature extraction and threshold calibration.

The main graph files are stored in:

- `data/knowledge_graph/siegen_audio_depression_kg.jsonld`
- `data/knowledge_graph/siegen_audio_depression_kg.calibrated.jsonld`

The seed graph is the ontology template. The calibrated graph contains the same structure but includes dataset-calibrated thresholds and calibration metadata. The graph loader automatically uses the calibrated graph when it exists.

### 5.2 Feature Matrix Builder

The feature matrix builder creates a participant-level table. It reads PHQ-8 labels, finds participants with available audio files, extracts features, and writes a CSV file. The generated file is:

`data/month_02_knowledge_graph/calibration/acoustic_feature_matrix.csv`

This file is important because it makes the threshold calibration reproducible. The supervisor can inspect the participant-level values and see that the thresholds were not guessed.

### 5.3 Calibration Script

The calibration script reads the participant-level feature matrix and the PHQ-8 label files. It then calibrates each graph biomarker separately. This is important because the same raw feature can appear in different clinical directions. For example, pitch mean can support both high-pitch and low-pitch biomarker nodes, each with its own direction and threshold.

The calibration outputs are:

- `calibrated_thresholds.csv`
- `calibration_summary.json`
- `participant_feature_matrix.csv`
- `siegen_audio_depression_kg.calibrated.jsonld`

The calibrated graph stores threshold metadata such as calibration source, reference label, sample size, sensitivity, specificity, and Youden's J. This makes the graph more transparent and easier to defend.

### 5.4 Graph Loading and Rule Assessment

The graph loader first checks whether a calibrated graph exists. If it does, it loads that version. If not, it falls back to the seed graph. This means the normal analysis pipeline does not recalibrate thresholds every time. Calibration is performed when the dataset or feature extraction changes. Normal report generation simply uses the saved calibrated thresholds.

Feature assessment reads biomarker rules from the graph. If a feature crosses its threshold in the required direction, the corresponding biomarker node is marked as detected. This produces structured findings containing the feature name, value, threshold, direction, graph node, label, and clinical summary.

### 5.5 Month 3 Graph-RAG Reporter Agent

The Reporter Agent is now a working Graph-RAG component. It creates structured JSON reports containing biomarkers, rule findings, graph explanation paths, deterministic clinical summaries, LLM status metadata, and an optional local LLM report. The deterministic summary remains available even when the LLM is disabled. This makes the system robust for demonstrations and avoids making the clinical explanation dependent on a generative model.

The local LLM connection is implemented through an Ollama client. The client sends the graph-grounded prompt to `http://localhost:11434/api/generate`, with `stream` disabled so that the Python pipeline receives one complete JSON response. The current tested model is `gemma3:1b`, which is small enough to run locally on a laptop. The pipeline exposes command-line controls for the model name, timeout, temperature, persona, and whether the LLM should be disabled.

The command used for the current Month 3 test was:

`python -m src.pipeline.run_pipeline Dataset\300_AUDIO.wav --max-duration-seconds 30 --llm-provider ollama --llm-model gemma3:1b --llm-timeout 180 --persona psychologist --quiet --output data\month_03_graph_rag\300_gemma_report.json`

The `--max-duration-seconds` flag was added for practical demonstration because long DAIC-WOZ recordings can make live meetings slow. Omitting this flag keeps full-audio analysis available.

## 6. Current Results

The Month 2 graph validation remains valid after the Month 3 integration. The graph validation produced the following summary:

- Nodes: 24
- Edges: 27
- Active biomarker rules: 11
- Terminal target types: risk, intervention, and screening construct
- Validation problems: none

The calibrated thresholds were derived from 108 to 111 participants, depending on feature availability. Transcript-derived features were available for 108 participants. Audio and COVAREP-derived features were available for 111 participants.

The strongest single-feature balanced accuracies came from pitch variability and high mean pitch, but even these remained modest. This supports the design choice that the system should not make diagnosis from a single acoustic feature. The role of the threshold is to decide whether a feature should enter the reasoning graph. The final explanation should combine multiple findings, graph context, and clinical caution.

The Month 3 Graph-RAG pipeline was tested on `Dataset\300_AUDIO.wav` using a 30-second demonstration window. The run produced a saved JSON report in:

`data/month_03_graph_rag/300_gemma_report.json`

For this demonstration sample, the system extracted acoustic biomarkers, matched 5 calibrated biomarker rules, retrieved 22 graph explanation paths, and successfully generated an LLM narrative with Gemma through Ollama. The LLM status in the saved report showed:

`provider: ollama`  
`model: gemma3:1b`  
`used: true`  
`error: none`

This confirms that the Month 3 prototype is operational: the graph retrieval layer supplies structured context, and the local LLM converts it into a readable psychologist-facing report. A second fallback mode was also tested with the LLM disabled. In that mode, the system still saved a deterministic graph-grounded report. This fallback is important because the core Graph-RAG evidence chain should remain available even when local model inference is slow or unavailable.

## 7. Discussion

The work completed so far strengthens the thesis in four ways.

First, the system now has a clear evidence chain. It starts from extracted audio features, calibrates their operating thresholds against PHQ-8 labels, maps threshold crossings to biomarker nodes, retrieves graph paths, and produces structured explanations. This is more defensible than a black-box score or a hand-written rule list.

Second, the knowledge graph makes clinical reasoning inspectable. A supervisor can open the graph and see exactly how a biomarker connects to a concept or recommendation. This is important because the project is about explainability, not only classification.

Third, the calibration results are honest. They do not overstate the power of acoustic biomarkers. Some features show high sensitivity but poor specificity; others show high specificity but low sensitivity. This reflects the real difficulty of speech-based depression screening. Instead of hiding this, the thesis uses it to justify Graph-RAG: individual features are weak, but structured explanations can still help users understand what evidence was detected.

Fourth, the Month 3 local LLM layer makes the prototype closer to the intended thesis goal. The system now demonstrates how retrieved graph evidence can be translated into a readable clinical narrative. This is different from asking an LLM to make a judgement from audio alone. The Reporter Agent receives already selected evidence and graph paths, and its output can be compared against those paths for faithfulness.

There are also limitations. The jitter measure is currently a proxy derived from F0 instability rather than a full clinical jitter measure from a dedicated voice-analysis protocol. MFCCs are extracted with a lightweight implementation, which is useful for reproducibility but may differ from Librosa or openSMILE feature sets. The current graph is manually curated and should ideally be reviewed by a clinical expert. The system is also calibrated on available local DAIC-WOZ data and should be evaluated carefully on held-out data in the next phase. The local LLM layer can improve readability, but it can also introduce unsupported wording if prompts are not constrained. For that reason, the report stores the retrieved graph paths and LLM output separately, making it possible to audit whether the generated narrative follows the evidence.

## 8. Ethical and Clinical Considerations

The system must not be presented as a diagnostic tool. Depression and anxiety diagnosis requires clinical assessment, patient history, context, and professional judgement. Speech features can be affected by many non-clinical factors, including microphone quality, language, fatigue, personality, medication, and interview context.

The safer framing is screening support and explanation support. The system can say that a recording contains speech patterns that, in this dataset and under this preprocessing pipeline, are associated with PHQ-8 screening status. It can then explain those patterns through graph paths. It should not say that the person has depression because of a voice feature.

This distinction should remain visible in all generated reports. Patient-facing reports should avoid alarming language. Psychologist-facing reports can be more technical but should still mark uncertainty clearly.

## 9. Next Steps

The next development stage should focus on strengthening the agentic reporting and evaluation pipeline. The planned work includes:

1. Add a LangGraph-style orchestrator to make the Analyzer -> Retriever -> Reporter flow explicit.
2. Improve psychologist and patient report prompts through systematic prompt testing.
3. Add longitudinal context from simulated previous weeks.
4. Compare Graph-RAG reports with vector-RAG or LLM-only baselines.
5. Evaluate explanation quality using criteria such as clarity, traceability, usefulness, hallucination risk, and audience fit.
6. Add an automatic faithfulness check that compares the LLM report against the retrieved graph paths.

The evaluation stage is especially important for publication potential. A publishable paper should not only describe the architecture; it should show that graph-grounded explanations are better than a baseline.

## 10. Conclusion

The first three months of the thesis have produced a defensible working prototype for explainable audio biomarker reporting. The project now includes a feature-to-concept mapping, a lightweight Clinical Knowledge Graph, DAIC-WOZ PHQ-8-calibrated acoustic thresholds, graph validation tools, graph-path retrieval, a deterministic reporter, and a local Gemma/Ollama Reporter Agent.

The key contribution at this stage is the shift from raw acoustic prediction toward traceable semantic reasoning and grounded narrative generation. Acoustic features are not treated as universal diagnostic markers. Instead, they are calibrated against a validated screening label and used as evidence nodes inside a graph. The local LLM is then used only after retrieval, so the narrative is tied to explicit graph evidence rather than model memory alone.

The thesis remains a screening-support and explanation-support project, not a diagnostic system. That limitation is important, but it also makes the work more responsible. The next stages should focus on stronger orchestration, longitudinal memory, baseline comparison, and empirical evaluation of explanation quality.

## Sources Used for Month 3 Update

The Month 3 literature and implementation update is based on the following sources:

- Lewis et al. (2020) is used as the foundation for retrieval-augmented generation. It supports the idea that language generation becomes more factual and controllable when it is grounded in retrieved external knowledge rather than relying only on model parameters.
- Microsoft Research GraphRAG (2024) is used to justify graph-based retrieval instead of only vector-based text retrieval. It supports the claim that graph retrieval is useful when explanations require relationships among entities rather than isolated text chunks.
- Asai et al. (2023), Self-RAG, is used to support the safety argument that retrieval and self-checking can improve factuality. The current project does not implement Self-RAG training, but it follows the same principle that generated reports should be grounded in retrieved evidence.
- Yao et al. (2022), ReAct, is used to justify the agentic structure of the system. The thesis adapts the reasoning-and-acting idea into a constrained clinical pipeline where the actions are feature extraction, graph retrieval, and report generation.
- Ji et al. (2023) is used to motivate hallucination control. This is important because unsupported language generation is risky in clinical and mental-health contexts.
- Google DeepMind's Gemma 3 model card is used to justify the choice of Gemma as a lightweight local language model for report generation.
- Ollama API documentation is used to justify the local deployment method. Ollama exposes a local HTTP API at `http://localhost:11434/api`, which allows the Reporter Agent to call Gemma without sending report context to an external cloud service.

## References

Al Hanai, T., Ghassemi, M., & Glass, J. (2018). Detecting depression with audio/text sequence modeling of interviews. Interspeech 2018, 1716-1720. https://www.isca-archive.org/interspeech_2018/alhanai18_interspeech.html

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. arXiv. https://doi.org/10.48550/arXiv.2310.11511

Cummins, N., Scherer, S., Krajewski, J., Schnieder, S., Epps, J., & Quatieri, T. F. (2015). A review of depression and suicide risk assessment using speech analysis. Speech Communication, 71, 10-49. https://doi.org/10.1016/j.specom.2015.03.004

Google DeepMind. (2025). Gemma 3 model card. Google AI for Developers. https://ai.google.dev/gemma/docs/core/model_card_3

Gratch, J., Artstein, R., Lucas, G., Stratou, G., Scherer, S., Nazarian, A., Wood, R., Boberg, J., DeVault, D., Marsella, S., Traum, D., Rizzo, S., & Morency, L.-P. (2014). The Distress Analysis Interview Corpus of human and computer interviews. Proceedings of LREC 2014, 3123-3128. https://aclanthology.org/L14-1421/

Holzinger, A., Biemann, C., Pattichis, C. S., & Kell, D. B. (2017). What do we need to build explainable AI systems for the medical domain? arXiv. https://doi.org/10.48550/arXiv.1712.09923

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12), Article 248. https://doi.org/10.1145/3571730

Kroenke, K., Strine, T. W., Spitzer, R. L., Williams, J. B. W., Berry, J. T., & Mokdad, A. H. (2009). The PHQ-8 as a measure of current depression in the general population. Journal of Affective Disorders, 114(1-3), 163-173. https://doi.org/10.1016/j.jad.2008.06.026

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W.-T., Rocktaschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems 33. https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

Low, D. M., Bentley, K. H., & Ghosh, S. S. (2020). Automated assessment of psychiatric disorders using speech: A systematic review. Laryngoscope Investigative Otolaryngology, 5, 96-116. https://doi.org/10.1002/lio2.354

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30. https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

Microsoft Research. (2024). GraphRAG: Unlocking LLM discovery on narrative private data. https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/

Ollama. (n.d.). Ollama API documentation. https://docs.ollama.com/api/introduction

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. arXiv. https://arxiv.org/abs/1602.04938

Scherer, S., Stratou, G., Lucas, G., Mahmoud, M., Boberg, J., Gratch, J., Rizzo, A., & Morency, L.-P. (2014). Automatic audiovisual behavior descriptors for psychological disorder analysis. Image and Vision Computing, 32(10), 648-658. https://doi.org/10.1016/j.imavis.2014.06.001

Tjoa, E., & Guan, C. (2021). A survey on explainable artificial intelligence (XAI): Toward medical XAI. IEEE Transactions on Neural Networks and Learning Systems, 32(11), 4793-4813. https://doi.org/10.1109/TNNLS.2020.3027314

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing reasoning and acting in language models. arXiv. https://arxiv.org/abs/2210.03629
