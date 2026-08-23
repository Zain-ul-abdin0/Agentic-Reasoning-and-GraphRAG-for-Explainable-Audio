# Month 3 Graph-RAG Prototype

This milestone adds a functional Graph-RAG reporting layer on top of the Month 2
depression knowledge graph.

## What Changed

- The pipeline still extracts acoustic biomarkers from `.wav` files.
- Calibrated biomarker rules are still loaded from the JSON-LD knowledge graph.
- Matched biomarkers are used as graph start nodes.
- NetworkX shortest-path retrieval finds explanation paths from biomarkers to
  depression screening, risk, and follow-up nodes.
- The retrieved paths are passed to a Reporter Agent prompt.
- The Reporter Agent can call a local Ollama model such as Gemma 3.
- If the local LLM is unavailable, the system still returns a deterministic
  graph-grounded report.

## Run Without Local LLM

```powershell
python -m src.pipeline.run_pipeline "path\to\audio.wav" --llm-provider none --output data\month_03_graph_rag\report.json
```

For a fast meeting demo on a long recording:

```powershell
python -m src.pipeline.run_pipeline Dataset\300_AUDIO.wav --max-duration-seconds 60 --llm-provider none --quiet --output data\month_03_graph_rag\300_graph_rag_report.json
```

## Run With Local Gemma Through Ollama

Start Ollama and make sure the model is available:

```powershell
ollama pull gemma3
ollama serve
```

Then run:

```powershell
python -m src.pipeline.run_pipeline "path\to\audio.wav" --llm-provider ollama --llm-model gemma3 --persona psychologist --output data\month_03_graph_rag\report.json
```

For a patient-facing version:

```powershell
python -m src.pipeline.run_pipeline "path\to\audio.wav" --llm-provider ollama --llm-model gemma3 --persona patient --output data\month_03_graph_rag\patient_report.json
```

## Supervisor Explanation

The Month 3 prototype is not a free-form chatbot. The LLM receives only:

- extracted acoustic features,
- matched calibrated biomarker rules,
- retrieved graph paths,
- safety instructions that forbid diagnosis,
- a persona instruction for psychologist or patient wording.

This makes the report grounded in the knowledge graph instead of relying only on
the LLM's internal memory.
