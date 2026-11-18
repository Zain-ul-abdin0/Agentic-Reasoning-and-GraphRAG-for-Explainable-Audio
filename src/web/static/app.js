const form = document.getElementById("analysisForm");
const apiStatus = document.getElementById("apiStatus");
const datasetAudio = document.getElementById("datasetAudio");
const runButton = document.querySelector(".run-button");

const panels = {
  overview: document.getElementById("overviewPanel"),
  agents: document.getElementById("agentsPanel"),
  graph: document.getElementById("graphPanel"),
  report: document.getElementById("reportPanel"),
  evaluation: document.getElementById("evaluationPanel"),
};

function setStatus(text, state = "") {
  apiStatus.textContent = text;
  apiStatus.className = `status-pill ${state}`.trim();
}

function text(value, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function clear(node) {
  node.replaceChildren();
}

function make(tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content !== undefined) element.textContent = content;
  return element;
}


function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdownLite(value) {
  const safe = escapeHtml(value || "Run with Ollama/Gemma to generate LLM text.");
  return safe
    .replace(/^###\s+(.+)$/gm, '<h3>$1</h3>')
    .replace(/^##\s+(.+)$/gm, '<h3>$1</h3>')
    .replace(/^#\s+(.+)$/gm, '<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^-\s+(.+)$/gm, '<div class="md-bullet">$1</div>')
    .replace(/\n/g, '<br>');
}

function setTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      Object.values(panels).forEach((panel) => panel.classList.remove("active"));
      tab.classList.add("active");
      panels[tab.dataset.tab].classList.add("active");
    });
  });
}

async function loadDatasetAudios() {
  try {
    const response = await fetch("/api/dataset-audios");
    const data = await response.json();
    clear(datasetAudio);
    datasetAudio.appendChild(new Option("Choose from Dataset", ""));
    data.audios.forEach((audio) => {
      const sidecars = [];
      if (audio.has_transcript) sidecars.push("transcript");
      if (audio.has_covarep) sidecars.push("COVAREP");
      const label = sidecars.length ? `${audio.name} (${sidecars.join(" + ")})` : audio.name;
      datasetAudio.appendChild(new Option(label, audio.path));
    });
  } catch (error) {
    clear(datasetAudio);
    datasetAudio.appendChild(new Option("Could not load Dataset folder", ""));
    setStatus("Dataset error", "error");
  }
}

function buildFormData() {
  const formData = new FormData();
  const fields = [
    "datasetAudio",
    "persona",
    "llmProvider",
    "llmModel",
    "timeout",
    "maxDuration",
    "historyWeeks",
  ];
  const mapping = {
    datasetAudio: "dataset_audio",
    persona: "persona",
    llmProvider: "llm_provider",
    llmModel: "llm_model",
    timeout: "llm_timeout",
    maxDuration: "max_duration_seconds",
    historyWeeks: "simulate_history_weeks",
  };

  fields.forEach((id) => {
    const field = document.getElementById(id);
    if (field.value !== "") formData.append(mapping[id], field.value);
  });

  formData.append("use_auto_sidecars", document.getElementById("autoSidecars").checked ? "true" : "false");

  const fileFields = [
    ["audioFile", "audio_file"],
    ["transcriptFile", "transcript_file"],
    ["covarepFile", "covarep_file"],
    ["historyFile", "history_file"],
  ];
  fileFields.forEach(([id, name]) => {
    const input = document.getElementById(id);
    if (input.files && input.files[0]) formData.append(name, input.files[0]);
  });

  return formData;
}

function renderMetrics(report) {
  const screenValue = document.getElementById("screenValue");
  screenValue.textContent = report.screen_positive ? "Positive" : "Below cutoff";
  screenValue.className = report.screen_positive ? "positive" : "negative";
  document.getElementById("evidenceValue").textContent = text(report.evidence_level);
  document.getElementById("matchedValue").textContent = `${text(report.matched_biomarker_count, "0")} / ${text(report.available_biomarker_count, "0")}`;
  document.getElementById("reportLocation").textContent = `Saved report: ${report.report_path}`;
}

function renderOverview(report) {
  const chips = document.getElementById("biomarkerChips");
  clear(chips);
  if (!report.matched_biomarkers || report.matched_biomarkers.length === 0) {
    chips.appendChild(make("span", "chip", "No matched biomarkers"));
  } else {
    report.matched_biomarkers.forEach((marker) => chips.appendChild(make("span", "chip", marker)));
  }

  const sources = document.getElementById("featureSources");
  clear(sources);
  Object.entries(report.feature_sources || {}).forEach(([feature, source]) => {
    const item = make("div", "source-item");
    item.innerHTML = `<b>${feature}</b><span>${source}</span>`;
    sources.appendChild(item);
  });

  document.getElementById("recursiveSummary").textContent = report.recursive_context?.summary || "No recursive context used.";
}

function renderAgents(report) {
  const flow = document.getElementById("agentFlow");
  clear(flow);
  (report.agent_flow || []).forEach((agent) => flow.appendChild(make("div", "flow-step", agent)));

  const chain = document.getElementById("promptChain");
  clear(chain);
  (report.persona_prompt_chain || []).forEach((step) => {
    const item = document.createElement("li");
    item.innerHTML = `<strong>${step.step}</strong>: ${step.instruction}`;
    chain.appendChild(item);
  });

  const trace = document.getElementById("traceList");
  clear(trace);
  (report.trace || []).forEach((stage) => {
    const item = make("div", "trace-item");
    item.innerHTML = `<b>${stage.agent}</b><span>${stage.action}</span>`;
    trace.appendChild(item);
  });
}

function renderGraph(report) {
  const paths = document.getElementById("graphPaths");
  clear(paths);
  if (!report.graph_paths || report.graph_paths.length === 0) {
    paths.appendChild(make("div", "path-item", "No graph paths retrieved."));
    return;
  }

  report.graph_paths.forEach((path) => {
    const item = make("div", "path-item");
    const labels = (path.path_labels || path.path || []).join(" -> ");
    const firstSourceStep = (path.steps || []).find((step) => step.sources_to_next && step.sources_to_next.length);
    const sources = firstSourceStep ? firstSourceStep.sources_to_next.join(", ") : "No edge source shown";
    item.innerHTML = `<div class="path-labels">${labels}</div><div class="path-source">${sources}</div>`;
    paths.appendChild(item);
  });
}

function renderReport(report) {
  const summary = document.getElementById("clinicalSummary");
  clear(summary);
  (report.user_facing_summary || report.clinical_summary || []).forEach((line) => summary.appendChild(make("li", "", line)));

  document.getElementById("llmReport").innerHTML = renderMarkdownLite(report.display_report || report.llm_report);

  const faithfulness = document.getElementById("faithfulness");
  clear(faithfulness);
  const check = report.faithfulness_check || {};
  faithfulness.appendChild(make("div", "faith-item", `Checked: ${text(check.checked)}`));
  faithfulness.appendChild(make("div", "faith-item", `Supported: ${text(check.supported)}`));
  const warnings = check.warnings && check.warnings.length ? check.warnings.join(" | ") : "No warnings";
  faithfulness.appendChild(make("div", "faith-item", warnings));
}

function renderAll(report) {
  renderMetrics(report);
  renderOverview(report);
  renderAgents(report);
  renderGraph(report);
  renderReport(report);
}

function formatNumber(value, digits = 3) {
  const number = Number(value);
  if (Number.isNaN(number)) return text(value);
  return number.toFixed(digits);
}

function metricCard(label, value, detail = "") {
  const card = make("div", "eval-card");
  card.innerHTML = `<span>${label}</span><strong>${value}</strong>${detail ? `<small>${detail}</small>` : ""}`;
  return card;
}

function renderSimpleTable(table, rows, columns) {
  clear(table);
  if (!rows || rows.length === 0) {
    const row = table.insertRow();
    const cell = row.insertCell();
    cell.textContent = "No rows available.";
    return;
  }

  const header = table.createTHead().insertRow();
  columns.forEach((column) => {
    const cell = document.createElement("th");
    cell.textContent = column.label;
    header.appendChild(cell);
  });

  const body = table.createTBody();
  rows.forEach((item) => {
    const row = body.insertRow();
    columns.forEach((column) => {
      const cell = row.insertCell();
      cell.textContent = text(item[column.key]);
    });
  });
}

function renderEvaluation(payload) {
  const metricsNode = document.getElementById("evaluationMetrics");
  clear(metricsNode);

  if (!payload.available || !payload.summary) {
    metricsNode.appendChild(metricCard("Status", "No results", "Run Month 5 evaluation first"));
    document.getElementById("evaluationAnalysis").textContent = "No Month 5 evaluation files were found yet.";
    return;
  }

  const summary = payload.summary;
  const metrics = summary.classification_metrics || {};
  const retrieval = summary.retrieval_comparison || {};
  const review = summary.mock_qualitative_review || {};

  metricsNode.appendChild(metricCard("Participants", text(summary.participants_evaluated), summary.reference_label));
  metricsNode.appendChild(metricCard("Accuracy", formatNumber(metrics.accuracy)));
  metricsNode.appendChild(metricCard("Sensitivity", formatNumber(metrics.sensitivity)));
  metricsNode.appendChild(metricCard("Specificity", formatNumber(metrics.specificity)));
  metricsNode.appendChild(metricCard("Balanced Accuracy", formatNumber(metrics.balanced_accuracy)));
  metricsNode.appendChild(metricCard("F1 Score", formatNumber(metrics.f1_score)));
  metricsNode.appendChild(metricCard("Automated Checklist", formatNumber(review.psychologist_mean_score), "not a human review"));
  renderSimpleTable(document.getElementById("legacyRetrievalMetrics"), [
    { diagnostic: "Graph path structural consistency", value: formatNumber(retrieval.graph_rag_mean_precision) },
    { diagnostic: `TF-IDF incident-edge precision@${retrieval.top_k || 3}`, value: formatNumber(retrieval.vector_mean_precision_at_k) },
  ], [{ key: "diagnostic", label: "Diagnostic (different criteria)" }, { key: "value", label: "Value" }]);

  document.getElementById("evaluationAnalysis").textContent = payload.comparative_analysis || "Comparative analysis file not found.";

  renderSimpleTable(
    document.getElementById("participantTable"),
    payload.tables?.participants || [],
    [
      { key: "participant_id", label: "ID" },
      { key: "phq8_score", label: "PHQ-8" },
      { key: "phq8_label", label: "Label" },
      { key: "prediction", label: "Prediction" },
      { key: "evidence_level", label: "Evidence" },
      { key: "matched_biomarker_count", label: "Matched" },
    ],
  );

  renderSimpleTable(
    document.getElementById("retrievalTable"),
    payload.tables?.retrieval || [],
    [
      { key: "participant_id", label: "ID" },
      { key: "matched_biomarker", label: "Biomarker" },
      { key: "precision_at_k", label: "TF-IDF incident-edge P@K" },
      { key: "graph_rag_precision", label: "Graph structural consistency" },
      { key: "top_chunks", label: "Top Chunks" },
    ],
  );
}

function renderRetrievalBenchmark(result) {
  const status = document.getElementById("retrievalBenchmarkStatus");
  const limitations = document.getElementById("retrievalBenchmarkLimitations");
  clear(limitations);
  if (!result) {
    status.textContent = "No shared-criteria comparison saved.";
    clear(document.getElementById("sharedRetrievalMetrics"));
    clear(document.getElementById("sharedRetrievalQueries"));
    return;
  }
  status.textContent = `${result.query_count} questions | ${result.corpus_size} graph-derived chunks | k=${result.top_k} | ${result.annotation_status}`;
  renderSimpleTable(document.getElementById("sharedRetrievalMetrics"), Object.entries(result.methods).map(([method, values]) => ({
    method: method === "graph" ? "Graph path retrieval" : "TF-IDF lexical retrieval",
    ...Object.fromEntries(Object.entries(values).map(([key, value]) => [key, formatNumber(value)])),
  })), [
    { key: "method", label: "Retriever" },
    { key: "precision_at_k", label: `Precision@${result.top_k}` },
    { key: "recall_at_k", label: `Recall@${result.top_k}` },
    { key: "mrr_at_k", label: `MRR@${result.top_k}` },
    { key: "ndcg_at_k", label: `nDCG@${result.top_k}` },
    { key: "latency_ms", label: "Query ms" },
  ]);
  renderSimpleTable(document.getElementById("sharedRetrievalQueries"), result.rows.map(row => ({
    ...row, precision_at_k: formatNumber(row.precision_at_k), recall_at_k: formatNumber(row.recall_at_k),
    retrieved_ids: row.retrieved_ids.join("; "), relevant_ids: row.relevant_ids.join("; "),
  })), [
    { key: "query_id", label: "Query" }, { key: "question", label: "Question" },
    { key: "method", label: "Retriever" }, { key: "precision_at_k", label: "P@k" },
    { key: "recall_at_k", label: "R@k" }, { key: "retrieved_ids", label: "Retrieved IDs" },
    { key: "relevant_ids", label: "Reference IDs" },
  ]);
  (result.limitations || []).forEach(value => limitations.appendChild(make("li", "", value)));
}

async function loadRetrievalBenchmark() {
  try {
    const response = await fetch("/api/retrieval-benchmark");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Could not load retrieval comparison");
    renderRetrievalBenchmark(payload.result);
  } catch (error) {
    document.getElementById("retrievalBenchmarkStatus").textContent = error.message;
  }
}

async function runRetrievalBenchmark() {
  const button = document.getElementById("runRetrievalBenchmark");
  button.disabled = true;
  document.getElementById("retrievalBenchmarkStatus").textContent = "Running retrieval comparison...";
  try {
    const data = new FormData();
    data.append("top_k", document.getElementById("evalTopK").value || "3");
    const response = await fetch("/api/retrieval-benchmark/run", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Invalid top-k or benchmark");
    renderRetrievalBenchmark(payload.result);
  } catch (error) {
    document.getElementById("retrievalBenchmarkStatus").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function loadEvaluation() {
  try {
    const response = await fetch("/api/evaluation");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Could not load evaluation");
    renderEvaluation(payload);
    loadRetrievalBenchmark();
  } catch (error) {
    document.getElementById("evaluationAnalysis").textContent = error.message;
  }
}

async function runEvaluationFromUi() {
  const button = document.getElementById("runEvaluation");
  setStatus("Evaluating", "busy");
  button.disabled = true;

  try {
    const formData = new FormData();
    const limit = document.getElementById("evalLimit").value;
    const maxDuration = document.getElementById("evalMaxDuration").value;
    const topK = document.getElementById("evalTopK").value;
    if (limit) formData.append("limit", limit);
    if (maxDuration) formData.append("max_duration_seconds", maxDuration);
    if (topK) formData.append("top_k", topK);

    const response = await fetch("/api/evaluation/run", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Evaluation failed");
    renderEvaluation(payload);
    await loadRetrievalBenchmark();
    setStatus("Complete");
  } catch (error) {
    setStatus("Error", "error");
    document.getElementById("evaluationAnalysis").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function submitAnalysis(event) {
  event.preventDefault();
  setStatus("Running", "busy");
  runButton.disabled = true;

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: buildFormData(),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Analysis failed");
    renderAll(payload);
    setStatus("Complete");
  } catch (error) {
    setStatus("Error", "error");
    document.getElementById("reportLocation").textContent = error.message;
  } finally {
    runButton.disabled = false;
  }
}

setTabs();
loadDatasetAudios();
loadEvaluation();
form.addEventListener("submit", submitAnalysis);
document.getElementById("refreshEvaluation").addEventListener("click", loadEvaluation);
document.getElementById("runEvaluation").addEventListener("click", runEvaluationFromUi);
document.getElementById("runRetrievalBenchmark").addEventListener("click", runRetrievalBenchmark);
