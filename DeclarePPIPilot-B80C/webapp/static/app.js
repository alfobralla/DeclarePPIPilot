const form = document.getElementById("run-form");
const statusEl = document.getElementById("status");
const summaryPanel = document.getElementById("summary");
const summaryChips = document.getElementById("summary-chips");
const resultsEl = document.getElementById("results");
const runBtn = document.getElementById("run-btn");
const periodEnabled = document.getElementById("period_enabled");
const periodControls = document.getElementById("period_controls");
const filtersPanel = document.getElementById("filters");
const hideLowCvCheckbox = document.getElementById("hide_low_cv");
const debugModeCheckbox = document.getElementById("debug_mode");
const sortKpisSelect = document.getElementById("sort_kpis");
const chartTypeSelect = document.getElementById("chart_type");

let lastData = null;

function setStatus(kind, text) {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = text;
  statusEl.classList.remove("hidden");
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function toNumericValue(raw) {
  if (raw == null) return Number.NaN;
  if (typeof raw === "number") return Number.isFinite(raw) ? raw : Number.NaN;
  const s = String(raw).trim();
  if (!s) return Number.NaN;
  const direct = Number(s);
  if (!Number.isNaN(direct)) return direct;
  const m = s.match(/^(?:(\d+)\s+days?\s+)?(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?$/i);
  if (!m) return Number.NaN;
  const days = Number(m[1] || 0);
  const hh = Number(m[2] || 0);
  const mm = Number(m[3] || 0);
  const ss = Number(m[4] || 0);
  const frac = m[5] ? Number(`0.${m[5]}`) : 0;
  return days * 86400 + hh * 3600 + mm * 60 + ss + frac;
}

function toCvNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function sanitizeDomId(value) {
  return String(value).replace(/[^a-zA-Z0-9_-]/g, "_");
}

function parseGroupedValues(groupedValues) {
  if (!Array.isArray(groupedValues)) return null;
  const labels = [];
  const values = [];
  for (const row of groupedValues) {
    labels.push(String(row.bucket));
    const num = toNumericValue(row.value);
    if (Number.isNaN(num)) return null;
    values.push(num);
  }
  return { labels, values };
}

function cvClass(cv) {
  if (cv == null) return "";
  if (cv < 0.2) return "low";
  if (cv < 0.5) return "mid";
  return "high";
}

function renderSummary(data, shownCount, totalCount, shownGoals) {
  summaryPanel.classList.remove("hidden");
  const cvT = data?.meta?.cv_threshold == null ? "-" : String(data.meta.cv_threshold);
  const period = data?.meta?.time_grouper_freq || "none";
  const debug = debugModeCheckbox.checked ? "on" : "off";
  const chips = [
    `KPIs shown: ${shownCount}/${totalCount}`,
    `Goals shown: ${shownGoals}/${data.meta.n_goals}`,
    `Period: ${period}`,
    `CV threshold: ${cvT}`,
    `Debug: ${debug}`,
  ];
  summaryChips.innerHTML = chips.map((t) => `<span class="chip">${escapeHtml(t)}</span>`).join("");
}

function sortKpis(kpis) {
  const key = sortKpisSelect.value;
  const arr = [...kpis];
  if (key === "id_asc") return arr.sort((a, b) => String(a.kpi_id).localeCompare(String(b.kpi_id)));
  if (key === "id_desc") return arr.sort((a, b) => String(b.kpi_id).localeCompare(String(a.kpi_id)));
  if (key === "cv_desc") {
    return arr.sort((a, b) => (toCvNumber(b.coefficient_of_variation) ?? -Infinity) - (toCvNumber(a.coefficient_of_variation) ?? -Infinity));
  }
  if (key === "cv_asc") {
    return arr.sort((a, b) => (toCvNumber(a.coefficient_of_variation) ?? Infinity) - (toCvNumber(b.coefficient_of_variation) ?? Infinity));
  }
  if (key === "status") {
    const rank = { success: 0, skipped: 1, failed: 2, null: 3 };
    return arr.sort((a, b) => (rank[a.execution_status] ?? 3) - (rank[b.execution_status] ?? 3));
  }
  return arr;
}

function chartConfig(parsed) {
  const t = chartTypeSelect.value;
  return {
    type: t,
    data: {
      labels: parsed.labels,
      datasets: [{
        label: "KPI value",
        data: parsed.values,
        borderColor: "#0a9396",
        backgroundColor: "rgba(10,147,150,0.20)",
        fill: t === "line",
        tension: 0.25,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { maxRotation: 45, minRotation: 45 } } },
    },
  };
}

function render(data) {
  resultsEl.innerHTML = "";
  const hideLowCv = hideLowCvCheckbox.checked;
  const debugMode = debugModeCheckbox.checked;
  const cvThreshold = data?.meta?.cv_threshold == null ? 0.2 : Number(data.meta.cv_threshold);
  let shownCount = 0;
  let totalCount = 0;
  let shownGoals = 0;

  for (const goal of data.goals) {
    totalCount += goal.kpis.length;
    const computed = goal.kpis.filter((k) => {
      if (debugMode) return true;
      if (k.execution_status !== "success") return false;
      if (data.meta.time_grouper_freq == null) return k.primary_value != null;
      return k.variability_status === "ok" && Array.isArray(k.grouped_values) && k.grouped_values.length > 0;
    });

    const filtered = computed.filter((k) => {
      if (!hideLowCv) return true;
      const cv = toCvNumber(k.coefficient_of_variation);
      if (cv == null) return true;
      return !(cv < cvThreshold);
    });

    const kpis = sortKpis(filtered);
    if (kpis.length === 0) continue;
    shownGoals += 1;
    shownCount += kpis.length;

    const section = document.createElement("details");
    section.className = "goal-section";
    section.open = true;
    section.innerHTML = `
      <summary class="goal-head">
        <h2>${escapeHtml(goal.goal_name)} <span class="goal-badge">${kpis.length}/${goal.kpis.length}</span></h2>
        <p>${escapeHtml(goal.goal_description || "")}</p>
      </summary>
      <div class="kpi-grid"></div>
    `;
    const grid = section.querySelector(".kpi-grid");

    for (const kpi of kpis) {
      const card = document.createElement("article");
      card.className = "kpi-card";
      const cv = toCvNumber(kpi.coefficient_of_variation);
      const safeKpiId = sanitizeDomId(kpi.kpi_id);

      let valueBlock = "";
      if (data.meta.time_grouper_freq == null) {
        valueBlock = `<div class="value-box"><strong>Computation value:</strong><br/>${escapeHtml(kpi.primary_value)}</div>`;
      } else {
        const parsed = parseGroupedValues(kpi.grouped_values);
        if (parsed) {
          valueBlock = `<div class="value-box"><strong>Grouped values</strong><canvas id="chart-${safeKpiId}" height="140"></canvas></div>`;
        } else if (Array.isArray(kpi.grouped_values) && kpi.grouped_values.length > 0) {
          const rows = kpi.grouped_values
            .map((x) => `<tr><td>${escapeHtml(x.bucket)}</td><td>${escapeHtml(x.value)}</td></tr>`)
            .join("");
          valueBlock = `<div class="value-box"><strong>Grouped values</strong><table class="group-table"><thead><tr><th>Bucket</th><th>Value</th></tr></thead><tbody>${rows}</tbody></table></div>`;
        } else {
          valueBlock = `<div class="value-box"><strong>Grouped values:</strong> not available</div>`;
        }
      }

      const statusClass = kpi.execution_status === "success" ? "success" : (kpi.execution_status === "failed" ? "failed" : "other");
      card.innerHTML = `
        <div class="kpi-head">
          <h3 class="kpi-title">${escapeHtml(kpi.kpi_id)}</h3>
          <span class="status-chip ${statusClass}">${escapeHtml(kpi.execution_status || "unknown")}</span>
        </div>
        <div class="kpi-meta">
          ${cv == null ? `<span class="cv-chip">CV: -</span>` : `<span class="cv-chip ${cvClass(cv)}">CV: ${escapeHtml(cv.toFixed(4))}</span>`}
        </div>
        <div class="ai-warning">${escapeHtml(kpi.human_readable_warning)}</div>
        <div><strong>Human-readable PPI:</strong> ${escapeHtml(kpi.human_readable_definition || "Not available")}</div>
        <div><strong>Objective:</strong> ${escapeHtml(kpi.objective || "")}</div>
        <div><strong>Category rationale:</strong> ${escapeHtml(kpi.category_rationale || "")}</div>
        ${valueBlock}
        <details>
          <summary>Technical details</summary>
          <pre>${escapeHtml(kpi.kpi_metric_str || "Not available")}</pre>
          ${kpi.execution_error ? `<pre>${escapeHtml(kpi.execution_error)}</pre>` : ""}
        </details>
      `;
      grid.appendChild(card);

      if (data.meta.time_grouper_freq != null) {
        const parsed = parseGroupedValues(kpi.grouped_values);
        if (parsed) {
          const ctx = card.querySelector(`#chart-${CSS.escape(safeKpiId)}`);
          if (ctx) new Chart(ctx, chartConfig(parsed));
        }
      }
    }

    resultsEl.appendChild(section);
  }

  renderSummary(data, shownCount, totalCount, shownGoals);
}

function storeUiState() {
  const state = {
    hideLowCv: hideLowCvCheckbox.checked,
    debugMode: debugModeCheckbox.checked,
    sortKpis: sortKpisSelect.value,
    chartType: chartTypeSelect.value,
  };
  localStorage.setItem("ppidp_ui", JSON.stringify(state));
}

function loadUiState() {
  try {
    const raw = localStorage.getItem("ppidp_ui");
    if (!raw) return;
    const state = JSON.parse(raw);
    if (typeof state.hideLowCv === "boolean") hideLowCvCheckbox.checked = state.hideLowCv;
    if (typeof state.debugMode === "boolean") debugModeCheckbox.checked = state.debugMode;
    if (typeof state.sortKpis === "string") sortKpisSelect.value = state.sortKpis;
    if (typeof state.chartType === "string") chartTypeSelect.value = state.chartType;
  } catch {}
}

function showSkeletons() {
  resultsEl.innerHTML = "";
  for (let i = 0; i < 3; i++) {
    const sk = document.createElement("div");
    sk.className = "skeleton";
    resultsEl.appendChild(sk);
  }
}

periodEnabled.addEventListener("change", () => {
  periodControls.style.display = periodEnabled.checked ? "grid" : "none";
});
periodControls.style.display = "none";

[hideLowCvCheckbox, debugModeCheckbox, sortKpisSelect, chartTypeSelect].forEach((el) => {
  el.addEventListener("change", () => {
    storeUiState();
    if (lastData) render(lastData);
  });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  runBtn.disabled = true;
  showSkeletons();
  setStatus("info", "Running full pipeline... this can take some minutes.");

  try {
    const formData = new FormData(form);
    const res = await fetch("/api/run", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    lastData = data;
    filtersPanel.classList.remove("hidden");
    render(data);
    setStatus("info", `Done. KPIs: ${data.meta.n_kpis}, goals: ${data.meta.n_goals}`);
  } catch (err) {
    setStatus("error", `Error: ${err.message}`);
    resultsEl.innerHTML = "";
  } finally {
    runBtn.disabled = false;
  }
});

loadUiState();
