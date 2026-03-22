const CORE_KEYS = ["calories","carbohydrates","sugar","fat","saturated_fat","sodium","protein"];
const OPTIONAL_KEYS = ["fiber","cholesterol","added_sugar","vitamin_c","omega3"];

const NUTRIENTS = [
  {key:"calories", label:"Calories", unit:"kcal", dir:-1, max:900, desc:"Total energy per 100 g"},
  {key:"carbohydrates", label:"Carbohydrates", unit:"g", dir:-1, max:100, desc:"Total carbohydrates per 100 g"},
  {key:"sugar", label:"Sugar", unit:"g", dir:-1, max:60, desc:"Total sugars per 100 g"},
  {key:"fat", label:"Fat", unit:"g", dir:-1, max:50, desc:"Total fat per 100 g"},
  {key:"saturated_fat", label:"Saturated Fat", unit:"g", dir:-1, max:30, desc:"Saturated fat per 100 g"},
  {key:"sodium", label:"Sodium", unit:"mg", dir:-1, max:2000, desc:"Sodium per 100 g"},
  {key:"protein", label:"Protein", unit:"g", dir:+1, max:50, desc:"Protein per 100 g"},
  {key:"fiber", label:"Fiber", unit:"g", dir:+1, max:20, desc:"Not present in all foods"},
  {key:"cholesterol", label:"Cholesterol", unit:"mg", dir:-1, max:300, desc:"Absent in plant-based foods"},
  {key:"added_sugar", label:"Added Sugar", unit:"g", dir:-1, max:50, desc:"Strongest unhealthy signal"},
  {key:"vitamin_c", label:"Vitamin C", unit:"mg", dir:+1, max:100, desc:"Strong healthy signal"},
  {key:"omega3", label:"Omega-3", unit:"g", dir:+1, max:5, desc:"Healthy fat signal"},
];

const KEYS = NUTRIENTS.map(n => n.key);
const N_MAP = Object.fromEntries(NUTRIENTS.map(n => [n.key, n]));
const ALWAYS_ON = CORE_KEYS;

const activeNutrients = new Map();
CORE_KEYS.forEach(k => activeNutrients.set(k, 0));

const $foodName = document.getElementById('food-name');
const $search = document.getElementById('nutrient-search');
const $searchRes = document.getElementById('search-results');
const $panel = document.getElementById('active-panel');
const $count = document.getElementById('active-count');
const $btnPredict = document.getElementById('btn-predict');
const $btnReset = document.getElementById('btn-reset');
const $resultPanel = document.getElementById('result-panel');
const $resultInner = document.getElementById('result-inner');

function renderPanel() {
  const n = activeNutrients.size;
  $count.innerHTML = `<span>${n}</span> nutrient${n === 1 ? '' : 's'} active`;

  if (n === 0) {
    $panel.innerHTML = '<div class="empty-state">No nutrients added yet.<br>Search above to add nutrients.</div>';
    return;
  }

  $panel.innerHTML = '';
  for (const [key, val] of activeNutrients) {
    const meta = N_MAP[key];
    const isAlways = ALWAYS_ON.includes(key);
    const dirLabel = meta.dir === 1 ? '▲ MORE' : '▼ LESS';
    const dirCls = meta.dir === 1 ? 'dir-up' : 'dir-down';

    const row = document.createElement('div');
    row.className = 'nutrient-row';
    row.dataset.key = key;

    row.innerHTML = `
      <div class="nutrient-meta">
        <div class="nutrient-name">
          ${meta.label}
          <span class="dir-badge ${dirCls}">${dirLabel}</span>
          ${isAlways ? '<span class="always-badge">ALWAYS ON</span>' : ''}
        </div>
        <div class="nutrient-desc">${meta.desc}</div>
      </div>
      <div class="nutrient-input-group">
        <input type="number" class="nutrient-input" data-key="${key}"
          value="${val === null ? '' : val}" min="0" max="${meta.max * 3}" step="0.1"
          placeholder="0">
        <span class="nutrient-unit">${meta.unit}</span>
      </div>
      ${isAlways ? '<div style="width:26px"></div>' : `<button class="remove-btn" data-key="${key}" title="Remove">✕</button>`}
    `;

    $panel.appendChild(row);
  }

  $panel.querySelectorAll('.nutrient-input').forEach(inp => {
    inp.addEventListener('input', e => {
      const k = e.target.dataset.key;
      const raw = e.target.value.trim();
      activeNutrients.set(k, raw === '' ? null : parseFloat(raw));
    });
  });

  $panel.querySelectorAll('.remove-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      const k = e.currentTarget.dataset.key;
      activeNutrients.delete(k);
      renderPanel();
      renderSearch($search.value);
    });
  });
}

function renderSearch(query) {
  $searchRes.innerHTML = '';
  if (!query.trim()) return;

  const q = query.toLowerCase();
  const matches = NUTRIENTS.filter(n =>
    OPTIONAL_KEYS.includes(n.key) &&
    (n.label.toLowerCase().includes(q) || n.key.toLowerCase().includes(q)) &&
    !activeNutrients.has(n.key)
  ).slice(0, 9);

  if (matches.length === 0) {
    const alreadyAdded = NUTRIENTS.filter(n =>
      (n.label.toLowerCase().includes(q) || n.key.toLowerCase().includes(q)) &&
      activeNutrients.has(n.key)
    );
    $searchRes.innerHTML = `<span class="search-no-result">${
      alreadyAdded.length ? 'All matching nutrients already added.' : `No nutrients found for "${query}".`
    }</span>`;
    return;
  }

  matches.forEach(n => {
    const chip = document.createElement('button');
    chip.className = 'search-chip';
    chip.innerHTML = `<span class="chip-arrow ${n.dir === 1 ? 'up' : 'down'}">${n.dir === 1 ? '▲' : '▼'}</span>${n.label} <span style="color:#4a5e78;font-size:10px">${n.unit}</span>`;
    chip.addEventListener('click', () => {
      activeNutrients.set(n.key, null);
      renderPanel();
      $search.value = '';
      $searchRes.innerHTML = '';
      setTimeout(() => {
        const el = $panel.querySelector(`[data-key="${n.key}"]`);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 50);
    });
    $searchRes.appendChild(chip);
  });
}

$search.addEventListener('input', e => renderSearch(e.target.value));

$btnPredict.addEventListener('click', async () => {
  const blankCores = CORE_KEYS.filter(k => {
    const input = document.querySelector(`.nutrient-input[data-key="${k}"]`);
    return input && input.value.trim() === '';
  });

  if (blankCores.length > 0) {
    const labels = blankCores.map(k => N_MAP[k].label).join(', ');
    alert(`Please enter a value for: ${labels}\n\nEnter 0 if the food genuinely has none (e.g. Cholesterol = 0 for plant-based foods).`);
    blankCores.forEach(k => {
      const input = document.querySelector(`.nutrient-input[data-key="${k}"]`);
      if (input) {
        input.style.borderColor = 'var(--red)';
        setTimeout(() => input.style.borderColor = '', 3000);
      }
    });
    return;
  }

  const nutrients = Object.fromEntries(activeNutrients);
  const food = $foodName.value.trim() || 'Unknown Food';

  $resultPanel.classList.add('visible');
  $resultInner.innerHTML = `
    <div style="text-align:center;padding:32px;font-family:var(--mono);color:var(--muted);font-size:13px;">
      <div style="font-size:24px;margin-bottom:12px;animation:spin 1s linear infinite;display:inline-block">⟳</div>
      <div>Running models…</div>
    </div>
    <style>@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}</style>
  `;

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ food_name: food, nutrients })
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);
    const d = await response.json();
    if (d.error) throw new Error(d.error);

    const main = d.all_model;
    const isBorderline = main.is_borderline;
    const isHealthy = main.is_healthy;
    const verdictClass = isBorderline ? 'borderline' : (isHealthy ? 'healthy' : 'unhealthy');
    const verdictWord = isBorderline ? 'BORDERLINE' : (isHealthy ? 'HEALTHY' : 'UNHEALTHY');
    const pctH = main.prob_healthy;
    const pctU = main.prob_unhealthy;
    const barWidth = isBorderline ? 50 : (isHealthy ? pctH : pctU);

    const activeKeys = new Set(Object.keys(nutrients));
    const absentKeys = KEYS.filter(k => !activeKeys.has(k));
    const absentStr = absentKeys.slice(0, 14).map(k => N_MAP[k].label).join(', ')
      + (absentKeys.length > 14 ? ` … +${absentKeys.length - 14} more` : '');

    const coreHealthy = d.core_model.is_healthy;
    const coreColor = d.core_model.is_borderline ? 'var(--yellow)' : (coreHealthy ? 'var(--green)' : 'var(--red)');
    const allColor = isBorderline ? 'var(--yellow)' : (isHealthy ? 'var(--green)' : 'var(--red)');

    $resultInner.innerHTML = `
      ${d.warning ? `
        <div style="background:#1a1400;border:1px solid #5a4200;border-radius:8px;
          padding:10px 14px;margin-bottom:14px;font-family:var(--mono);font-size:11px;color:#fbbf24;">
          ${d.warning}
        </div>` : ''}
      ${isBorderline ? `
        <div style="background:#1a1200;border:1px solid var(--yellow);border-radius:8px;
          padding:10px 14px;margin-bottom:14px;font-family:var(--mono);font-size:11px;color:var(--yellow);line-height:1.6;">
          ⚠️ <strong>Borderline result</strong> — the model is not confident enough to call this Healthy or Unhealthy
          (${pctH}% healthy vs ${pctU}% unhealthy). Try adding optional features like Added Sugar, Vitamin C, or Omega-3 for a clearer verdict.
        </div>` : ''}

      <div class="result-verdict ${verdictClass}">
        <div class="verdict-label">${verdictWord}</div>
        <div class="verdict-details">
          <div class="verdict-food-name">${d.food_name}</div>
          <div class="verdict-bar-wrap">
            <div class="verdict-bar" style="width:${barWidth}%"></div>
          </div>
          <div style="font-family:var(--mono);font-size:9px;letter-spacing:0.12em;color:var(--muted);margin-bottom:8px;text-transform:uppercase;">Model Confidence</div>
          <div class="verdict-scores">
            <div class="score-item">
              <span class="score-pct h" style="color:${isBorderline ? 'var(--muted)' : (isHealthy ? 'var(--green)' : 'var(--muted)')}">${pctH}%</span>
              <span class="score-lbl">Confidence: Healthy</span>
            </div>
            <div class="score-item">
              <span class="score-pct u" style="color:${isBorderline ? 'var(--muted)' : (!isHealthy ? 'var(--red)' : 'var(--muted)')}">${pctU}%</span>
              <span class="score-lbl">Confidence: Unhealthy</span>
            </div>
            <div class="score-item">
              <span class="score-pct" style="font-size:18px;color:var(--muted)">${activeNutrients.size}</span>
              <span class="score-lbl">Nutrients Used</span>
            </div>
          </div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:16px;">
          <div style="font-family:var(--mono);font-size:9px;letter-spacing:0.15em;color:var(--muted);margin-bottom:8px;">CORE MODEL (9 features)</div>
          <div style="font-size:18px;font-weight:700;color:${coreColor};margin-bottom:4px;">${d.core_model.label.toUpperCase()}</div>
          <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:2px;letter-spacing:0.1em;">CONFIDENCE</div>
          <div style="font-family:var(--mono);font-size:11px;color:var(--muted);">
            Healthy: <span style="color:var(--green)">${d.core_model.prob_healthy}%</span> &nbsp;
            Unhealthy: <span style="color:var(--red)">${d.core_model.prob_unhealthy}%</span>
          </div>
          <div style="margin-top:10px;font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:4px;">TOP SIGNALS:</div>
          ${d.core_model.features.map(f => {
            const isGood = f.direction === 'Healthy signal';
            return `<div style="font-family:var(--mono);font-size:10px;padding:3px 0;color:${isGood ? 'var(--green)' : 'var(--red)'}">${isGood ? '▲' : '▼'} ${f.reason}</div>`;
          }).join('')}
        </div>
        <div style="background:var(--surface2);border:1px solid var(--border2);border-radius:10px;padding:16px;">
          <div style="font-family:var(--mono);font-size:9px;letter-spacing:0.15em;color:var(--muted);margin-bottom:8px;">ALL-FEATURES MODEL (12 features)</div>
          <div style="font-size:18px;font-weight:700;color:${allColor};margin-bottom:4px;">${d.all_model.label.toUpperCase()}</div>
          <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:2px;letter-spacing:0.1em;">CONFIDENCE</div>
          <div style="font-family:var(--mono);font-size:11px;color:var(--muted);">
            Healthy: <span style="color:var(--green)">${d.all_model.prob_healthy}%</span> &nbsp;
            Unhealthy: <span style="color:var(--red)">${d.all_model.prob_unhealthy}%</span>
          </div>
          <div style="margin-top:10px;font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:4px;">TOP SIGNALS:</div>
          ${d.all_model.features.map(f => {
            const isGood = f.direction === 'Healthy signal';
            return `<div style="font-family:var(--mono);font-size:10px;padding:3px 0;color:${isGood ? 'var(--green)' : 'var(--red)'}">${isGood ? '▲' : '▼'} ${f.reason}</div>`;
          }).join('')}
        </div>
      </div>

      <div style="margin-top:4px;">
        <div style="font-family:var(--mono);font-size:9px;letter-spacing:0.15em;color:var(--muted);text-transform:uppercase;margin-bottom:8px;">
          Full Nutrient Contribution Breakdown
        </div>
        ${buildContribTable(d.all_model.features)}
      </div>
    `;
  } catch (err) {
    $resultInner.innerHTML = `
      <div style="padding:20px;font-family:var(--mono);font-size:12px;color:var(--red);
        background:#1a0508;border:1px solid var(--red);border-radius:8px;">
        <strong>Error:</strong> ${err.message}<br>
        <span style="color:var(--muted);font-size:11px;">Make sure Flask is running on localhost:5000</span>
      </div>`;
  }

  $resultPanel.classList.add('visible');
  setTimeout(() => {
    $resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 80);
});

$btnReset.addEventListener('click', () => {
  activeNutrients.clear();
  CORE_KEYS.forEach(k => activeNutrients.set(k, 0));
  $foodName.value = '';
  $search.value = '';
  $searchRes.innerHTML = '';
  $resultPanel.classList.remove('visible');
  $resultInner.innerHTML = '';
  renderPanel();
});

renderPanel();

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  document.getElementById('pane-' + tab).classList.add('active');
}

let csvResultsData = [];

const $csvDrop = document.getElementById('csv-drop');
const $csvFileInput = document.getElementById('csv-file-input');
const $csvStatus = document.getElementById('csv-status');
const $csvResultsWrap = document.getElementById('csv-results-wrap');
const $csvTableBody = document.getElementById('csv-table-body');
const $csvSummary = document.getElementById('csv-summary');

$csvDrop.addEventListener('dragover',  e => { e.preventDefault(); $csvDrop.classList.add('drag-over'); });
$csvDrop.addEventListener('dragleave', ()  => $csvDrop.classList.remove('drag-over'));
$csvDrop.addEventListener('drop', e => {
  e.preventDefault();
  $csvDrop.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleCsvFile(file);
});
$csvDrop.addEventListener('click', () => $csvFileInput.click());
$csvFileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleCsvFile(e.target.files[0]);
});

async function handleCsvFile(file) {
  if (!file.name.endsWith('.csv')) {
    $csvStatus.innerHTML = '<span style="color:var(--red)">✗ Please upload a .csv file</span>';
    return;
  }
  $csvStatus.innerHTML = '<span style="color:var(--muted)">⟳ Uploading and analyzing…</span>';
  $csvResultsWrap.style.display = 'none';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/predict-csv', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    csvResultsData = data.results;
    $csvStatus.innerHTML = `<span style="color:var(--green)">✓ Analyzed ${data.results.length} food item${data.results.length !== 1 ? 's' : ''} from <strong>${file.name}</strong></span>`;
    renderCsvTable(data.results);
  } catch (err) {
    $csvStatus.innerHTML = `<span style="color:var(--red)">✗ ${err.message}</span>`;
  }
}

function verdictChip(label) {
  return `<span class="verdict-chip ${label.toLowerCase()}">${label.toUpperCase()}</span>`;
}

function renderCsvTable(results) {
  const counts = { Healthy: 0, Unhealthy: 0, Borderline: 0 };
  results.forEach(r => { counts[r.all_model.label] = (counts[r.all_model.label] || 0) + 1; });

  $csvSummary.innerHTML = [
    ['Healthy', 'var(--green)', counts.Healthy || 0],
    ['Borderline', 'var(--yellow)', counts.Borderline || 0],
    ['Unhealthy', 'var(--red)', counts.Unhealthy || 0],
  ].map(([label, color, n]) => `
    <div style="display:flex;flex-direction:column;align-items:center;
      background:var(--surface2);border:1px solid var(--border);border-radius:8px;
      padding:12px 24px;min-width:90px;">
      <span style="font-size:26px;font-weight:800;color:${color};line-height:1">${n}</span>
      <span style="font-size:9px;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;margin-top:4px">${label}</span>
    </div>`).join('');

  $csvTableBody.innerHTML = results.map((r, i) => `
    <tr>
      <td style="color:var(--muted)">${i + 1}</td>
      <td style="color:var(--text);font-weight:600">${r.food_name}</td>
      <td>${verdictChip(r.core_model.label)}</td>
      <td>${verdictChip(r.all_model.label)}</td>
      <td>
        <span style="color:var(--green);font-weight:600">${r.all_model.prob_healthy}%</span>
        <span style="color:var(--muted);font-size:9px"> healthy</span>
      </td>
      <td><button class="btn-view-detail" onclick="openModal(${i})">View Details</button></td>
    </tr>`).join('');

  $csvResultsWrap.style.display = 'block';
  setTimeout(() => $csvResultsWrap.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
}

function buildContribTable(features) {
  if (!features || features.length === 0) {
    return '<div style="font-family:var(--mono);font-size:11px;color:var(--muted);padding:12px">No feature data available.</div>';
  }

  const maxAbs = Math.max(...features.map(f => Math.abs(f.weight)), 0.001);

  return `<table class="contrib-table">
    <thead><tr>
      <th>Nutrient</th>
      <th>Value</th>
      <th>Effect on Score</th>
      <th>Weight</th>
    </tr></thead>
    <tbody>
    ${features.map(f => {
      const isPos = f.direction === 'Healthy signal';
      const barPct = Math.min(Math.abs(f.weight) / maxAbs * 100, 100);
      const unit = (N_MAP[f.feature] || {}).unit || '';
      const rawStr = (f.raw_value !== null && f.raw_value !== undefined)
        ? `${f.raw_value} <span style="color:#4a5e78;font-size:9px">${unit}</span>`
        : '<span style="color:#4a5e78;font-style:italic;font-size:10px">imputed</span>';
      return `<tr>
        <td style="color:var(--text);font-weight:500">${f.label}</td>
        <td style="font-family:var(--mono)">${rawStr}</td>
        <td><span class="effect-badge ${isPos?'pos':'neg'}">${isPos?'▲ healthier':'▼ unhealthier'}</span></td>
        <td>
          <div style="display:flex;align-items:center;gap:8px;">
            <div class="weight-bar-wrap">
              <div class="weight-bar ${isPos?'pos':'neg'}" style="width:${barPct}%"></div>
            </div>
            <span style="color:${isPos?'var(--green)':'var(--red)'};min-width:56px;text-align:right;font-family:var(--mono)">
              ${f.weight > 0 ? '+' : ''}${f.weight.toFixed(4)}
            </span>
          </div>
        </td>
      </tr>`;
    }).join('')}
    </tbody>
  </table>`;
}

const $modal = document.getElementById('detail-modal');
const $modalContent = document.getElementById('modal-content');
let currentModalIdx = 0;

function downloadFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportSummaryCSV() {
  if (!csvResultsData.length) return;

  const rows = [
    ['#','Food Name','Core Model','All-Features Model',
     'Confidence Healthy (%)','Confidence Unhealthy (%)',
     'Top Healthy Signal','Top Unhealthy Signal','Models Agree']
  ];

  csvResultsData.forEach((r, i) => {
    const allFeats = r.all_model.features || [];
    const topHealthy = allFeats.find(f => f.direction === 'Healthy signal');
    const topUnhealthy = allFeats.find(f => f.direction === 'Unhealthy signal');
    const agree = r.core_model.label === r.all_model.label ? 'Yes' : 'No';

    rows.push([
      i + 1,
      `"${r.food_name.replace(/"/g, '""')}"`,
      r.core_model.label,
      r.all_model.label,
      r.all_model.prob_healthy,
      r.all_model.prob_unhealthy,
      topHealthy ? `"${topHealthy.reason.replace(/"/g,'""')}"` : '',
      topUnhealthy ? `"${topUnhealthy.reason.replace(/"/g,'""')}"` : '',
      agree,
    ]);
  });

  const csv = rows.map(r => r.join(',')).join('\n');
  const ts = new Date().toISOString().slice(0,16).replace('T','_').replace(':','-');
  downloadFile(`nutriscan_summary_${ts}.csv`, csv, 'text/csv');
}

function exportFullJSON() {
  if (!csvResultsData.length) return;

  const payload = {
    exported_at: new Date().toISOString(),
    total_foods: csvResultsData.length,
    summary: {
      healthy: csvResultsData.filter(r => r.all_model.label === 'Healthy').length,
      borderline: csvResultsData.filter(r => r.all_model.label === 'Borderline').length,
      unhealthy: csvResultsData.filter(r => r.all_model.label === 'Unhealthy').length,
    },
    results: csvResultsData.map((r, i) => ({
      index: i + 1,
      food_name: r.food_name,
      verdict: {
        core_model: r.core_model.label,
        all_features_model: r.all_model.label,
        confidence_healthy: r.all_model.prob_healthy,
        confidence_unhealthy: r.all_model.prob_unhealthy,
        models_agree: r.core_model.label === r.all_model.label,
      },
      core_model_details: {
        label: r.core_model.label,
        prob_healthy: r.core_model.prob_healthy,
        prob_unhealthy:r.core_model.prob_unhealthy,
        feature_contributions: (r.core_model.features || []).map(f => ({
          feature: f.feature,
          value: f.raw_value,
          direction: f.direction,
          reason: f.reason,
          weight: f.weight,
        })),
      },
      all_model_details: {
        label: r.all_model.label,
        prob_healthy: r.all_model.prob_healthy,
        prob_unhealthy: r.all_model.prob_unhealthy,
        feature_contributions: (r.all_model.features || []).map(f => ({
          feature: f.feature,
          value: f.raw_value !== null ? f.raw_value : 'imputed',
          direction: f.direction,
          reason: f.reason,
          weight: f.weight,
        })),
      },
      data_notes: {
        blank_core_fields: r.blank_cores || [],
      },
    }))
  };

  const ts = new Date().toISOString().slice(0,16).replace('T','_').replace(':','-');
  downloadFile(`nutriscan_full_${ts}.json`, JSON.stringify(payload, null, 2), 'application/json');
}

function exportSingleJSON(idx) {
  const r = csvResultsData[idx];
  if (!r) return;

  const payload = {
    exported_at: new Date().toISOString(),
    food_name: r.food_name,
    verdict: {
      core_model: r.core_model.label,
      all_features_model: r.all_model.label,
      confidence_healthy: r.all_model.prob_healthy,
      confidence_unhealthy: r.all_model.prob_unhealthy,
      models_agree: r.core_model.label === r.all_model.label,
    },
    core_model_details: {
      label: r.core_model.label,
      prob_healthy: r.core_model.prob_healthy,
      prob_unhealthy:r.core_model.prob_unhealthy,
      feature_contributions: (r.core_model.features || []).map(f => ({
        feature: f.feature, value: f.raw_value,
        direction: f.direction, reason: f.reason, weight: f.weight,
      })),
    },
    all_model_details: {
      label: r.all_model.label,
      prob_healthy: r.all_model.prob_healthy,
      prob_unhealthy:r.all_model.prob_unhealthy,
      feature_contributions: (r.all_model.features || []).map(f => ({
        feature: f.feature,
        value: f.raw_value !== null ? f.raw_value : 'imputed',
        direction: f.direction, reason: f.reason, weight: f.weight,
      })),
    },
    data_notes: { blank_core_fields: r.blank_cores || [] },
  };

  const safeName = r.food_name.replace(/[^a-z0-9]/gi, '_').toLowerCase().slice(0, 40);
  downloadFile(`nutriscan_${safeName}.json`, JSON.stringify(payload, null, 2), 'application/json');
}

function openModal(idx) {
  currentModalIdx = idx;
  const r = csvResultsData[idx];
  const main = r.all_model;
  const isBorderline = main.is_borderline;
  const isHealthy = main.is_healthy;
  const verdictColor = isBorderline ? 'var(--yellow)' : (isHealthy ? 'var(--green)' : 'var(--red)');
  const verdictWord = isBorderline ? 'BORDERLINE' : (isHealthy ? 'HEALTHY' : 'UNHEALTHY');
  const coreColor = r.core_model.is_borderline ? 'var(--yellow)' : (r.core_model.is_healthy ? 'var(--green)' : 'var(--red)');
  const bgColor = isBorderline ? '#1a1200' : (isHealthy ? '#051a10' : '#1a0508');

  $modalContent.innerHTML = `
    <div style="margin-bottom:20px;">
      <div style="font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:0.12em;margin-bottom:4px;">FOOD ITEM</div>
      <div style="font-size:24px;font-weight:800;letter-spacing:-0.02em">${r.food_name}</div>
    </div>

    <div style="display:flex;align-items:center;gap:20px;background:${bgColor};
      border:1px solid ${verdictColor};border-radius:10px;padding:18px 22px;margin-bottom:16px;">
      <div style="font-size:36px;font-weight:800;color:${verdictColor};letter-spacing:-0.03em;line-height:1">${verdictWord}</div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--muted);line-height:2;">
        Confidence: Healthy <span style="color:var(--green);font-weight:600">${main.prob_healthy}%</span>
        &nbsp;·&nbsp;
        Unhealthy <span style="color:var(--red);font-weight:600">${main.prob_unhealthy}%</span>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px;">
      <div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:14px 16px;">
        <div style="font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:0.12em;margin-bottom:6px;">CORE MODEL · 7 features</div>
        <div style="font-size:18px;font-weight:700;color:${coreColor};margin-bottom:4px">${r.core_model.label.toUpperCase()}</div>
        <div style="font-family:var(--mono);font-size:10px;color:var(--muted);">
          Healthy <span style="color:var(--green)">${r.core_model.prob_healthy}%</span>
          &nbsp;·&nbsp; Unhealthy <span style="color:var(--red)">${r.core_model.prob_unhealthy}%</span>
        </div>
      </div>
      <div style="background:var(--surface2);border:1px solid var(--border2);border-radius:8px;padding:14px 16px;">
        <div style="font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:0.12em;margin-bottom:6px;">ALL-FEATURES MODEL · 12 features</div>
        <div style="font-size:18px;font-weight:700;color:${verdictColor};margin-bottom:4px">${main.label.toUpperCase()}</div>
        <div style="font-family:var(--mono);font-size:10px;color:var(--muted);">
          Healthy <span style="color:var(--green)">${main.prob_healthy}%</span>
          &nbsp;·&nbsp; Unhealthy <span style="color:var(--red)">${main.prob_unhealthy}%</span>
        </div>
      </div>
    </div>

    ${r.blank_cores && r.blank_cores.length > 0 ? `
      <div style="background:#0d1520;border:1px solid var(--border2);border-radius:8px;
        padding:10px 14px;margin-bottom:14px;font-family:var(--mono);font-size:11px;color:var(--muted);">
        ⚠️ These core fields were missing in the CSV and defaulted to 0:
        <span style="color:var(--yellow)">${r.blank_cores.join(', ')}</span>
      </div>` : ''}

    <div style="font-family:var(--mono);font-size:9px;letter-spacing:0.15em;color:var(--muted);
      text-transform:uppercase;margin-bottom:8px;">Full Nutrient Contribution Breakdown</div>
    <div style="overflow-x:auto;">
      ${buildContribTable(main.features)}
    </div>
  `;

  $modal.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  $modal.classList.remove('open');
  document.body.style.overflow = '';
}

$modal.addEventListener('click', e => { if (e.target === $modal) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
