from datetime import datetime
from pathlib import Path
import json
import re
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline.run_pipeline import analyze_audio, resolve_path
from src.rag.llm_client import LocalLLMConfig


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"
WEB_REPORT_DIR = BASE_DIR / "data" / "month_04_agentic_reporting_engine" / "web_reports"
UPLOAD_DIR = BASE_DIR / "data" / "month_04_agentic_reporting_engine" / "uploads"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Agentic Graph-RAG Audio Biomarker Dashboard",
    description="FastAPI frontend for explainable depression-screening reports from audio biomarkers.",
    version="2026-09-month-04",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "audio"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path.resolve())


def _resolve_report_path(path_value: str) -> Path:
    path = resolve_path(path_value)
    try:
        path.relative_to(BASE_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Report path must stay inside the project folder.") from exc

    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file not found.")

    return path


def _compact_report(report: dict, report_path: Path) -> dict:
    aggregate = report.get("aggregate_evidence") or {}
    recursive = report.get("recursive_context") or {}
    engine = report.get("agentic_engine") or {}
    features = report.get("biomarkers") or {}
    matched_findings = [
        finding for finding in report.get("rule_findings", [])
        if finding.get("matched")
    ]

    return {
        "report_path": _relative(report_path),
        "file": report.get("file"),
        "persona": report.get("persona"),
        "screen_positive": aggregate.get("screen_positive"),
        "evidence_level": aggregate.get("evidence_level"),
        "matched_biomarker_count": aggregate.get("matched_biomarker_count"),
        "available_biomarker_count": aggregate.get("available_biomarker_count"),
        "matched_biomarkers": aggregate.get("matched_biomarkers", []),
        "feature_sources": features.get("feature_sources", {}),
        "sidecar_files": features.get("sidecar_files", {}),
        "agent_flow": engine.get("flow", []),
        "trace": engine.get("trace", []),
        "persona_prompt_chain": report.get("persona_prompt_chain", []),
        "recursive_context": recursive,
        "clinical_summary": report.get("clinical_summary", []),
        "llm_status": report.get("llm_status", {}),
        "llm_report": report.get("llm_report"),
        "faithfulness_check": report.get("faithfulness_check", {}),
        "graph_paths": report.get("explanation_paths", [])[:8],
        "matched_findings": matched_findings,
    }


def _llm_config(provider: str, model: str, timeout: int, temperature: float):
    if provider == "none":
        return None

    return LocalLLMConfig(
        provider=provider,
        model=model,
        timeout_seconds=timeout,
        temperature=temperature,
    )


async def _save_upload(upload: Optional[UploadFile], destination: Path) -> Optional[Path]:
    if upload is None or not upload.filename:
        return None

    destination.mkdir(parents=True, exist_ok=True)
    saved_path = destination / _safe_name(upload.filename)
    with saved_path.open("wb") as file:
        shutil.copyfileobj(upload.file, file)
    return saved_path


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/dataset-audios")
def dataset_audios():
    files = []
    if DATASET_DIR.exists():
        for audio in sorted(DATASET_DIR.glob("*_AUDIO.wav")):
            transcript = audio.with_name(audio.name.replace("_AUDIO.wav", "_TRANSCRIPT.csv"))
            covarep = audio.with_name(audio.name.replace("_AUDIO.wav", "_COVAREP.csv"))
            files.append({
                "name": audio.name,
                "path": _relative(audio),
                "has_transcript": transcript.exists(),
                "has_covarep": covarep.exists(),
            })

    return {"audios": files}


@app.get("/api/report")
def read_report(path: str):
    report_path = _resolve_report_path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return _compact_report(report, report_path)


@app.post("/api/analyze")
async def analyze(
    dataset_audio: str = Form(""),
    persona: str = Form("psychologist"),
    llm_provider: str = Form("none"),
    llm_model: str = Form("gemma3:1b"),
    llm_timeout: int = Form(180),
    temperature: float = Form(0.2),
    max_duration_seconds: Optional[float] = Form(None),
    simulate_history_weeks: int = Form(0),
    use_auto_sidecars: bool = Form(True),
    audio_file: Optional[UploadFile] = File(None),
    transcript_file: Optional[UploadFile] = File(None),
    covarep_file: Optional[UploadFile] = File(None),
    history_file: Optional[UploadFile] = File(None),
):
    if persona not in {"psychologist", "patient"}:
        raise HTTPException(status_code=400, detail="Persona must be psychologist or patient.")
    if llm_provider not in {"none", "ollama"}:
        raise HTTPException(status_code=400, detail="LLM provider must be none or ollama.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="agentic_audio_", dir=UPLOAD_DIR))

    try:
        if audio_file and audio_file.filename:
            audio_path = await _save_upload(audio_file, temp_dir)
        elif dataset_audio:
            audio_path = resolve_path(dataset_audio)
        else:
            raise HTTPException(status_code=400, detail="Choose a dataset audio or upload a .wav file.")

        if not audio_path.exists() or audio_path.suffix.lower() != ".wav":
            raise HTTPException(status_code=400, detail="Audio input must be an existing .wav file.")

        transcript_path = await _save_upload(transcript_file, temp_dir)
        covarep_path = await _save_upload(covarep_file, temp_dir)
        history_path = await _save_upload(history_file, temp_dir)

        config = _llm_config(llm_provider, llm_model, llm_timeout, temperature)
        report = analyze_audio(
            audio_file=audio_path,
            persona=persona,
            llm_config=config,
            max_duration_seconds=max_duration_seconds,
            transcript_path=transcript_path,
            covarep_path=covarep_path,
            auto_sidecars=use_auto_sidecars,
            history_path=history_path,
            simulated_history_weeks=simulate_history_weeks,
        )

        WEB_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_name = f"{audio_path.stem}_{persona}_{run_id}.json"
        report_path = WEB_REPORT_DIR / _safe_name(report_name)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return _compact_report(report, report_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


