#!/usr/bin/env python3
"""Local web UI for editing positions.json.

Lets you view/edit positions by hand, or upload a screenshot of your
brokerage app and have an AI vision model draft position rows for you to
review and correct before saving. Runs entirely on your machine -- the
only network calls are to whichever AI provider you've configured, and
only when you click "Parse screenshot".

Usage:
    python web_ui.py
Then open http://127.0.0.1:5050 in your browser.
"""

from __future__ import annotations

import json
import os
import shutil

from flask import Flask, jsonify, render_template_string, request

import config
from portfolio_monitor import credentials, vision
from portfolio_monitor.models import Position

app = Flask(__name__)

VISION_DEFAULT_MODEL = {
    "anthropic": config.ANTHROPIC_VISION_MODEL,
    "openai": config.OPENAI_VISION_MODEL,
    "gemini": config.GEMINI_VISION_MODEL,
}


def _load_positions_raw() -> list[dict]:
    if not os.path.exists(config.POSITIONS_PATH):
        return []
    with open(config.POSITIONS_PATH) as f:
        return json.load(f)


def _save_positions_raw(positions: list[dict]) -> None:
    # Validate every row through the real model before writing anything --
    # a bad row here shouldn't corrupt the file the rest of the tool reads.
    for row in positions:
        Position.from_dict(row)

    if os.path.exists(config.POSITIONS_PATH):
        shutil.copy(config.POSITIONS_PATH, config.POSITIONS_PATH + ".bak")

    with open(config.POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)


@app.route("/")
def index():
    return render_template_string(PAGE_TEMPLATE, default_vision_provider=config.VISION_PROVIDER)


@app.route("/api/positions", methods=["GET"])
def get_positions():
    return jsonify(_load_positions_raw())


@app.route("/api/positions", methods=["POST"])
def save_positions():
    positions = request.get_json(force=True)
    if not isinstance(positions, list):
        return jsonify({"error": "expected a JSON array of positions"}), 400
    try:
        _save_positions_raw(positions)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "count": len(positions)})


@app.route("/api/parse-screenshot", methods=["POST"])
def parse_screenshot():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "no image uploaded"}), 400

    provider = request.form.get("provider", config.VISION_PROVIDER)
    if provider not in VISION_DEFAULT_MODEL:
        return jsonify({"error": f"unknown provider {provider!r}"}), 400
    model = request.form.get("model") or VISION_DEFAULT_MODEL[provider]

    api_key = credentials.resolve_api_key(provider, interactive=False)
    if not api_key:
        return (
            jsonify(
                {
                    "error": f"No saved API key for {provider}. Run "
                    f"'python main.py --interactive' once to register one, "
                    f"or set its environment variable."
                }
            ),
            400,
        )

    image_bytes = file.read()
    media_type = file.mimetype or "image/png"

    try:
        extracted = vision.extract_positions_from_image(image_bytes, media_type, provider, model, api_key=api_key)
    except Exception as exc:
        return jsonify({"error": f"{provider} extraction failed: {exc}"}), 502

    return jsonify({"positions": extracted})


PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Positions Editor</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; background: #0f1115; color: #e6e6e6; }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.05rem; margin-top: 2rem; color: #9fb4c7; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.75rem; }
  th, td { border: 1px solid #2a2f3a; padding: 4px 6px; font-size: 0.85rem; }
  th { background: #1a1e27; text-align: left; }
  input, select { width: 100%; box-sizing: border-box; background: #1a1e27; color: #e6e6e6; border: 1px solid #2a2f3a; padding: 3px; font-size: 0.85rem; }
  button { background: #2d6cdf; color: white; border: none; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-size: 0.9rem; margin-right: 8px; }
  button.secondary { background: #3a3f4b; }
  button.danger { background: #b3403a; padding: 3px 8px; font-size: 0.8rem; }
  .row { display: flex; align-items: center; gap: 10px; margin-top: 0.5rem; flex-wrap: wrap; }
  #status { margin-top: 1rem; padding: 8px 12px; border-radius: 4px; display: none; }
  #status.ok { background: #1f3a24; color: #8fe0a0; display: block; }
  #status.err { background: #3a1f1f; color: #e08f8f; display: block; }
  .source-tag { font-size: 0.7rem; color: #9fb4c7; }
  .col-actions { width: 40px; }
</style>
</head>
<body>

<h1>Positions Editor</h1>
<p>Edit positions.json directly below, or upload a screenshot of your brokerage app and let AI draft rows for you to review.</p>

<h2>Upload a screenshot</h2>
<div class="row">
  <select id="provider">
    <option value="anthropic">Anthropic (Claude)</option>
    <option value="openai">OpenAI (GPT)</option>
    <option value="gemini">Google (Gemini)</option>
  </select>
  <input type="file" id="imageInput" accept="image/*" multiple>
  <button onclick="parseScreenshots()">Parse screenshot(s)</button>
</div>

<h2>Positions</h2>
<table id="positionsTable">
  <thead>
    <tr>
      <th>Type</th><th>Ticker</th><th>Opt Type</th><th>Strike</th><th>Expiry</th>
      <th>Entry Price</th><th>Contracts/Shares</th><th>Entry Date</th>
      <th>Target</th><th>Stop</th><th class="col-actions"></th>
    </tr>
  </thead>
  <tbody id="positionsBody"></tbody>
</table>

<div class="row" style="margin-top: 1rem;">
  <button class="secondary" onclick="addRow()">+ Add row</button>
  <button onclick="savePositions()">Save positions.json</button>
</div>

<div id="status"></div>

<script>
let rows = [];

function emptyRow(source) {
  return {
    asset_type: "shares", ticker: "", option_type: "", strike: "", expiry: "",
    entry_price: "", contracts: "", entry_date: "", target_price: "", stop_price: "",
    _source: source || "manual",
  };
}

function render() {
  const body = document.getElementById("positionsBody");
  body.innerHTML = "";
  rows.forEach((row, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><select onchange="updateField(${i}, 'asset_type', this.value)">
            <option value="shares" ${row.asset_type === "shares" ? "selected" : ""}>shares</option>
            <option value="option" ${row.asset_type === "option" ? "selected" : ""}>option</option>
          </select></td>
      <td><input value="${row.ticker ?? ""}" onchange="updateField(${i}, 'ticker', this.value)"></td>
      <td><select onchange="updateField(${i}, 'option_type', this.value)">
            <option value="" ${!row.option_type ? "selected" : ""}></option>
            <option value="call" ${row.option_type === "call" ? "selected" : ""}>call</option>
            <option value="put" ${row.option_type === "put" ? "selected" : ""}>put</option>
          </select></td>
      <td><input value="${row.strike ?? ""}" onchange="updateField(${i}, 'strike', this.value)"></td>
      <td><input value="${row.expiry ?? ""}" placeholder="YYYY-MM-DD" onchange="updateField(${i}, 'expiry', this.value)"></td>
      <td><input value="${row.entry_price ?? ""}" onchange="updateField(${i}, 'entry_price', this.value)"></td>
      <td><input value="${row.contracts ?? ""}" onchange="updateField(${i}, 'contracts', this.value)"></td>
      <td><input value="${row.entry_date ?? ""}" placeholder="YYYY-MM-DD" onchange="updateField(${i}, 'entry_date', this.value)"></td>
      <td><input value="${row.target_price ?? ""}" onchange="updateField(${i}, 'target_price', this.value)"></td>
      <td><input value="${row.stop_price ?? ""}" onchange="updateField(${i}, 'stop_price', this.value)"></td>
      <td class="col-actions"><button class="danger" onclick="deleteRow(${i})">x</button></td>
    `;
    body.appendChild(tr);
    if (row._source && row._source !== "manual") {
      const tag = document.createElement("tr");
      tag.innerHTML = `<td colspan="11" class="source-tag">from screenshot -- please verify every field above</td>`;
      body.appendChild(tag);
    }
  });
}

function updateField(i, field, value) {
  rows[i][field] = value;
}

function addRow() {
  rows.push(emptyRow("manual"));
  render();
}

function deleteRow(i) {
  rows.splice(i, 1);
  render();
}

function showStatus(message, ok) {
  const el = document.getElementById("status");
  el.textContent = message;
  el.className = ok ? "ok" : "err";
}

async function loadPositions() {
  const res = await fetch("/api/positions");
  const data = await res.json();
  rows = data.map(r => ({ ...emptyRow("manual"), ...r, _source: "manual" }));
  render();
}

function toNumberOrNull(v) {
  if (v === "" || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
}

async function savePositions() {
  const payload = rows.map(r => {
    const row = {
      asset_type: r.asset_type,
      ticker: r.ticker,
      entry_price: toNumberOrNull(r.entry_price),
      contracts: toNumberOrNull(r.contracts),
      entry_date: r.entry_date || null,
      target_price: toNumberOrNull(r.target_price),
      stop_price: toNumberOrNull(r.stop_price),
    };
    if (r.asset_type === "option") {
      row.option_type = r.option_type;
      row.strike = toNumberOrNull(r.strike);
      row.expiry = r.expiry;
    }
    return row;
  });

  const res = await fetch("/api/positions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (res.ok) {
    showStatus(`Saved ${data.count} position(s) to positions.json.`, true);
  } else {
    showStatus(`Error: ${data.error}`, false);
  }
}

async function parseScreenshots() {
  const provider = document.getElementById("provider").value;
  const files = document.getElementById("imageInput").files;
  if (files.length === 0) {
    showStatus("Choose at least one screenshot first.", false);
    return;
  }
  showStatus(`Parsing ${files.length} screenshot(s) with ${provider}...`, true);

  for (const file of files) {
    const form = new FormData();
    form.append("image", file);
    form.append("provider", provider);

    try {
      const res = await fetch("/api/parse-screenshot", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        showStatus(`Error parsing ${file.name}: ${data.error}`, false);
        return;
      }
      data.positions.forEach(p => rows.push({ ...emptyRow("screenshot"), ...p, _source: "screenshot" }));
    } catch (err) {
      showStatus(`Error parsing ${file.name}: ${err}`, false);
      return;
    }
  }
  render();
  showStatus("Screenshot(s) parsed -- review the highlighted rows below, then Save.", true);
}

document.getElementById("provider").value = "{{ default_vision_provider }}";
loadPositions();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Positions editor running at http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)
