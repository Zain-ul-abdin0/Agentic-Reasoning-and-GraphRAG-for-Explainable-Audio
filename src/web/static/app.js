const form = document.getElementById("analysisForm");
const apiStatus = document.getElementById("apiStatus");
const datasetAudio = document.getElementById("datasetAudio");
const runButton = document.querySelector(".run-button");

const panels = {
  overview: document.getElementById("overviewPanel"),
  agents: document.getElementById("agentsPanel"),
  graph: document.getElementById("graphPanel"),
  report: document.getElementById("reportPanel"),
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
  (report.clinical_summary || []).forEach((line) => summary.appendChild(make("li", "", line)));

  document.getElementById("llmReport").innerHTML = renderMarkdownLite(report.llm_report);

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
form.addEventListener("submit", submitAnalysis);
