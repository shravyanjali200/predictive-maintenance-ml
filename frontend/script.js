// ─────────────────────────────────────────────────────────────────
// API CONFIGURATION
// Auto-detects local vs deployed environment.
// Opens index.html directly (file://) or from localhost → local API.
// Served from any other origin (e.g. Render) → production Render API.
// To hard-code a URL, replace the block below with:
//   const API_BASE_URL = "https://your-url.onrender.com";
// ─────────────────────────────────────────────────────────────────

const API_BASE_URL = (
    window.location.protocol === "file:" ||
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
) ? "http://127.0.0.1:8000"
  : "https://predictive-maintenance-ml-kem2.onrender.com";


// ─────────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────────

let lastBatchResults = [];   // held in memory for CSV download


// ─────────────────────────────────────────────────────────────────
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────────────────────────

function showError(id, message) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = message;
    el.classList.remove("hidden");
}

function clearError(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = "";
    el.classList.add("hidden");
}

function setLoading(btn, isLoading, loadingText) {
    if (isLoading) {
        btn._originalText = btn.textContent;
        btn.textContent   = loadingText || "Loading…";
        btn.disabled      = true;
    } else {
        btn.textContent = btn._originalText || btn.textContent;
        btn.disabled    = false;
    }
}

function getRiskClass(riskLevel) {
    if (riskLevel === "Low Risk")    return "low";
    if (riskLevel === "Medium Risk") return "medium";
    if (riskLevel === "High Risk")   return "high";
    return "neutral";
}

function getResultBoxClass(riskLevel) {
    if (riskLevel === "High Risk")   return "result-box failure";
    if (riskLevel === "Medium Risk") return "result-box warning";
    return "result-box normal";
}

function getResultIcon(prediction, riskLevel) {
    if (prediction === 0)            return "✓";
    if (riskLevel === "High Risk")   return "⚠";
    return "!";
}

function formatTimestamp(ts) {
    try {
        return new Date(ts + "Z").toLocaleString();
    } catch (_) {
        return ts;
    }
}

function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = String(str);
    return d.innerHTML;
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}


// ─────────────────────────────────────────────────────────────────
// TAB NAVIGATION
// ─────────────────────────────────────────────────────────────────

document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const target = btn.dataset.tab;

        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        document.querySelectorAll(".tab-content").forEach(pane => pane.classList.remove("active"));
        document.getElementById("tab-" + target).classList.add("active");

        if (target === "history") loadHistory();
    });
});


// ─────────────────────────────────────────────────────────────────
// FEATURE IMPORTANCE CHART
// ─────────────────────────────────────────────────────────────────

function renderFeatureImportance(featureImportance) {
    const container = document.getElementById("importanceChart");
    const entries   = Object.entries(featureImportance || {});

    if (entries.length === 0) {
        container.innerHTML = '<p class="empty-state">Feature importance not available.</p>';
        return;
    }

    // Sort descending, use highest value as 100 % bar reference
    const sorted = entries.sort((a, b) => b[1] - a[1]);
    const maxVal = sorted[0][1];

    // Render bars with width:0 first, animate after paint
    container.innerHTML = sorted.map(([name, value]) => {
        const pct      = (value * 100).toFixed(1);
        const barWidth = ((value / maxVal) * 100).toFixed(1);
        return `
            <div class="importance-row">
                <div class="importance-label">${escapeHtml(name)}</div>
                <div class="importance-bar-track">
                    <div class="importance-bar-fill" data-width="${barWidth}"></div>
                </div>
                <div class="importance-pct">${pct}%</div>
            </div>
        `;
    }).join("");

    // Trigger CSS transition by setting width after next paint
    requestAnimationFrame(() => {
        container.querySelectorAll(".importance-bar-fill").forEach(bar => {
            bar.style.width = bar.dataset.width + "%";
        });
    });
}


// ─────────────────────────────────────────────────────────────────
// SINGLE PREDICTION
// ─────────────────────────────────────────────────────────────────

document.getElementById("predictBtn").addEventListener("click", async () => {
    clearError("predictError");

    const airTemperature     = parseFloat(document.getElementById("air_temperature").value);
    const processTemperature = parseFloat(document.getElementById("process_temperature").value);
    const rotationalSpeed    = parseFloat(document.getElementById("rotational_speed").value);
    const torque             = parseFloat(document.getElementById("torque").value);
    const toolWear           = parseFloat(document.getElementById("tool_wear").value);

    if ([airTemperature, processTemperature, rotationalSpeed, torque, toolWear].some(v => !Number.isFinite(v))) {
        showError("predictError", "Please enter valid numeric values for all sensor fields.");
        return;
    }

    const btn = document.getElementById("predictBtn");
    setLoading(btn, true, "Analyzing…");

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                air_temperature:     airTemperature,
                process_temperature: processTemperature,
                rotational_speed:    rotationalSpeed,
                torque:              torque,
                tool_wear:           toolWear,
            }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server returned ${response.status}`);
        }

        const data = await response.json();
        displaySingleResult(data, { airTemperature, processTemperature, rotationalSpeed, torque, toolWear });
        loadDashboardStats();   // refresh stats silently after each prediction

    } catch (err) {
        showError("predictError", "Could not reach the prediction server. " + err.message);
    } finally {
        setLoading(btn, false);
    }
});


function displaySingleResult(data, inputs) {
    // Result box colour driven by risk tier (3 states)
    document.getElementById("resultBox").className = getResultBoxClass(data.risk_level);

    // Status icon + text
    document.getElementById("resultIcon").textContent     = getResultIcon(data.prediction, data.risk_level);
    document.getElementById("resultText").textContent     = data.result;
    document.getElementById("recommendation").textContent = data.recommendation;

    // Risk badge
    const badge = document.getElementById("riskBadge");
    badge.textContent = data.risk_level;
    badge.className   = `risk-badge ${getRiskClass(data.risk_level)}`;

    // Probability text + progress bar (colour follows risk tier)
    document.getElementById("probability").textContent = data.failure_probability + "%";
    const pbar     = document.getElementById("progressBar");
    pbar.style.width = data.failure_probability + "%";
    pbar.className   = `progress-bar ${getRiskClass(data.risk_level)}`;

    // Sensor summary tiles
    document.getElementById("summaryAir").textContent     = inputs.airTemperature     + " K";
    document.getElementById("summaryProcess").textContent = inputs.processTemperature + " K";
    document.getElementById("summarySpeed").textContent   = inputs.rotationalSpeed    + " rpm";
    document.getElementById("summaryTorque").textContent  = inputs.torque             + " Nm";
    document.getElementById("summaryWear").textContent    = inputs.toolWear           + " min";

    // Feature importance chart
    if (data.feature_importance) {
        renderFeatureImportance(data.feature_importance);
    }
}


// ─────────────────────────────────────────────────────────────────
// DASHBOARD STATISTICS  (loaded from /history)
// ─────────────────────────────────────────────────────────────────

async function loadDashboardStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/history?limit=500`);
        if (!response.ok) return;

        const data  = await response.json();
        const preds = data.predictions || [];

        document.getElementById("statTotal").textContent  = preds.length;
        document.getElementById("statLow").textContent    = preds.filter(p => p.risk_level === "Low Risk").length;
        document.getElementById("statMedium").textContent = preds.filter(p => p.risk_level === "Medium Risk").length;
        document.getElementById("statHigh").textContent   = preds.filter(p => p.risk_level === "High Risk").length;

    } catch (_) {
        // Non-critical — leave dashes in place if API is unreachable
    }
}


// ─────────────────────────────────────────────────────────────────
// PREDICTION HISTORY
// ─────────────────────────────────────────────────────────────────

async function loadHistory() {
    clearError("historyError");

    const limit     = document.getElementById("historyLimit").value;
    const loading   = document.getElementById("historyLoading");
    const container = document.getElementById("historyContainer");

    loading.classList.remove("hidden");
    container.innerHTML = "";

    try {
        const response = await fetch(`${API_BASE_URL}/history?limit=${limit}`);
        if (!response.ok) throw new Error(`Server returned ${response.status}`);

        const data  = await response.json();
        const preds = data.predictions || [];

        if (preds.length === 0) {
            container.innerHTML = '<p class="empty-state">No predictions found. Run a prediction to get started.</p>';
            return;
        }

        container.innerHTML = buildHistoryTable(preds);

    } catch (err) {
        showError("historyError", "Failed to load history: " + err.message);
    } finally {
        loading.classList.add("hidden");
    }
}

function buildHistoryTable(predictions) {
    const rows = predictions.map(p => {
        const rc = getRiskClass(p.risk_level);
        return `
            <tr>
                <td class="td-time">${escapeHtml(formatTimestamp(p.timestamp))}</td>
                <td><span class="risk-badge ${rc} small">${escapeHtml(p.risk_level)}</span></td>
                <td class="td-num">${p.failure_probability}%</td>
                <td class="${p.prediction === 1 ? "failure-text" : "normal-text"}">
                    ${p.prediction === 1 ? "Failure Risk" : "Normal"}
                </td>
                <td class="td-sensors">
                    ${p.air_temperature}K &nbsp;·&nbsp;
                    ${p.rotational_speed}rpm &nbsp;·&nbsp;
                    ${p.torque}Nm &nbsp;·&nbsp;
                    ${p.tool_wear}min
                </td>
            </tr>
        `;
    }).join("");

    return `
        <div class="table-wrapper">
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Risk Level</th>
                        <th>Probability</th>
                        <th>Result</th>
                        <th>Sensors</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

document.getElementById("refreshHistory").addEventListener("click", loadHistory);
document.getElementById("historyLimit").addEventListener("change",  loadHistory);


// ─────────────────────────────────────────────────────────────────
// BATCH CSV
// ─────────────────────────────────────────────────────────────────

const csvFileInput = document.getElementById("csvFile");
const uploadArea   = document.getElementById("uploadArea");
const uploadBtn    = document.getElementById("uploadBtn");

// ── Drag and drop ─────────────────────────────
uploadArea.addEventListener("dragover", e => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
});
uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
});
uploadArea.addEventListener("drop", e => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (!file) return;
    // Programmatically assign the dropped file to the input
    const dt = new DataTransfer();
    dt.items.add(file);
    csvFileInput.files = dt.files;
    handleFileSelect(file);
});

// ── Native file picker ─────────────────────────
csvFileInput.addEventListener("change", () => {
    if (csvFileInput.files[0]) handleFileSelect(csvFileInput.files[0]);
});

function handleFileSelect(file) {
    clearError("batchError");
    if (!file.name.toLowerCase().endsWith(".csv")) {
        showError("batchError", "Please select a .csv file.");
        uploadBtn.disabled = true;
        document.getElementById("selectedFile").classList.add("hidden");
        return;
    }
    const label = document.getElementById("selectedFile");
    label.textContent = `✓  ${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
    label.classList.remove("hidden");
    uploadBtn.disabled = false;
}

// ── Upload and predict ─────────────────────────
uploadBtn.addEventListener("click", async () => {
    clearError("batchError");

    const file = csvFileInput.files[0];
    if (!file) { showError("batchError", "Please select a CSV file first."); return; }

    setLoading(uploadBtn, true, "Processing…");

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_BASE_URL}/predict/batch`, {
            method: "POST",
            body:   formData,
        });

        if (!response.ok) {
            const err    = await response.json().catch(() => ({}));
            const detail = err.detail;
            if (detail && typeof detail === "object") {
                const missing = (detail.missing_columns || []).join(", ");
                throw new Error(`${detail.error || "Validation error."} Missing: ${missing}`);
            }
            throw new Error(detail || `Server returned ${response.status}`);
        }

        const data     = await response.json();
        lastBatchResults = data.predictions || [];
        displayBatchResults(data);

    } catch (err) {
        showError("batchError", "Batch prediction failed: " + err.message);
        document.getElementById("batchResultsSection").classList.add("hidden");
    } finally {
        setLoading(uploadBtn, false);
    }
});

function displayBatchResults(data) {
    const preds = data.predictions || [];
    document.getElementById("batchResultsSection").classList.remove("hidden");

    const low    = preds.filter(p => p.risk_level === "Low Risk").length;
    const medium = preds.filter(p => p.risk_level === "Medium Risk").length;
    const high   = preds.filter(p => p.risk_level === "High Risk").length;

    // Summary chips
    document.getElementById("batchSummary").innerHTML = `
        <div class="batch-chips">
            <span class="chip neutral">Total: ${data.total_rows}</span>
            <span class="chip low">Low Risk: ${low}</span>
            <span class="chip medium">Medium Risk: ${medium}</span>
            <span class="chip high">High Risk: ${high}</span>
        </div>
    `;

    // Results table
    const rows = preds.map((p, i) => {
        const rc = getRiskClass(p.risk_level);
        return `
            <tr>
                <td class="td-num">${i + 1}</td>
                <td>${p.air_temperature}</td>
                <td>${p.process_temperature}</td>
                <td>${p.rotational_speed}</td>
                <td>${p.torque}</td>
                <td>${p.tool_wear}</td>
                <td class="td-num">${p.failure_probability}%</td>
                <td><span class="risk-badge ${rc} small">${p.risk_level}</span></td>
                <td class="${p.prediction === 1 ? "failure-text" : "normal-text"}">
                    ${p.prediction === 1 ? "Failure Risk" : "Normal"}
                </td>
            </tr>
        `;
    }).join("");

    document.getElementById("batchContainer").innerHTML = `
        <div class="table-wrapper">
            <table class="history-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Air Temp (K)</th>
                        <th>Process Temp (K)</th>
                        <th>Speed (rpm)</th>
                        <th>Torque (Nm)</th>
                        <th>Wear (min)</th>
                        <th>Probability</th>
                        <th>Risk Level</th>
                        <th>Result</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

// ── Download results CSV ───────────────────────
document.getElementById("downloadResults").addEventListener("click", () => {
    if (!lastBatchResults.length) return;

    const headers = [
        "Row", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "Prediction", "Failure Probability (%)", "Risk Level", "Result", "Recommendation",
    ];

    const rows = lastBatchResults.map((p, i) => [
        i + 1, p.air_temperature, p.process_temperature,
        p.rotational_speed, p.torque, p.tool_wear,
        p.prediction, p.failure_probability, p.risk_level, p.result, p.recommendation,
    ]);

    const csv = [headers, ...rows]
        .map(row => row.map(c => `"${String(c).replace(/"/g, '""')}"`).join(","))
        .join("\n");

    downloadFile(csv, "batch_results.csv", "text/csv");
});

// ── Download CSV template from backend ────────
document.getElementById("downloadTemplate").addEventListener("click", async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/predict/batch/template`);
        if (!response.ok) throw new Error("Server error.");
        const text = await response.text();
        downloadFile(text, "batch_template.csv", "text/csv");
    } catch (err) {
        showError("batchError", "Failed to download template: " + err.message);
    }
});


// ─────────────────────────────────────────────────────────────────
// INITIALISE — load stats on page load
// ─────────────────────────────────────────────────────────────────

loadDashboardStats();
