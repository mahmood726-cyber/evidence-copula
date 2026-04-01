"""
build_dashboard.py — generates dashboard.html for EvidenceCopula results.
Run from C:/Models/EvidenceCopula/
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copula_engine import (
    EvidenceCopulaEngine, empirical_cdf, z_array_from_p
)


def run_pipeline() -> dict:
    """Run full pipeline and collect results for the dashboard."""
    eng = EvidenceCopulaEngine(
        "data/scores.csv",
        "data/verdicts.csv",
        "data/review_groups.csv",
    )
    df = eng.load()
    results = eng.fit_all()
    summary = eng.summary()

    overall = results["Overall"]
    best = overall.best_copula

    # Pseudo-observations for scatter
    u = empirical_cdf(df["final_score"].values).tolist()
    v = empirical_cdf(df["z_stat"].values).tolist()
    domains = df["domain"].fillna("Unknown").tolist()

    # Domain tail dependence
    domain_rows = []
    for group, res in results.items():
        bc = res.best_copula
        if bc is None:
            continue
        domain_rows.append({
            "group": group,
            "n": res.n,
            "best_copula": bc.family,
            "theta": round(bc.theta, 4),
            "lambda_L": round(bc.tail_lower, 4),
            "lambda_U": round(bc.tail_upper, 4),
            "kendall_tau": round(res.kendall_tau, 4),
            "spearman_rho": round(res.spearman_rho, 4),
            "aic": round(bc.aic_val, 3),
        })

    # All copula results for overall
    all_copulas = []
    for c in overall.copulas:
        all_copulas.append({
            "family": c.family,
            "theta": round(c.theta, 4),
            "loglik": round(c.loglik, 3),
            "aic": round(c.aic_val, 3),
            "lambda_L": round(c.tail_lower, 4),
            "lambda_U": round(c.tail_upper, 4),
        })

    return {
        "overall": {
            "n": overall.n,
            "kendall_tau": round(overall.kendall_tau, 4),
            "kendall_p": float(overall.kendall_p),
            "spearman_rho": round(overall.spearman_rho, 4),
            "spearman_p": float(overall.spearman_p),
            "best_copula": best.family,
            "theta": round(best.theta, 4),
            "aic": round(best.aic_val, 3),
            "lambda_L": round(best.tail_lower, 4),
            "lambda_U": round(best.tail_upper, 4),
        },
        "all_copulas": all_copulas,
        "domain_rows": domain_rows,
        "scatter_u": [round(x, 5) for x in u],
        "scatter_v": [round(x, 5) for x in v],
        "scatter_domains": domains,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EvidenceCopula Dashboard</title>
  <style>
    :root {
      --bg: #f6f3ee;
      --paper: rgba(255,255,255,0.93);
      --ink: #111;
      --muted: #5f5a53;
      --line: #ddd5ca;
      --line-strong: #b8aea2;
      --accent: #326891;
      --accent-soft: #e8f0f6;
      --good: #216c53;
      --warn: #a06a12;
      --bad: #922b21;
      --shadow: 0 16px 38px rgba(17,17,17,0.045);
      --radius: 8px;
      --serif: "Iowan Old Style","Palatino Linotype",Georgia,serif;
      --sans: "Segoe UI","Helvetica Neue",Arial,sans-serif;
      --mono: "SFMono-Regular",Consolas,"Liberation Mono",monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top,rgba(50,104,145,.08),transparent 28%),
                  linear-gradient(180deg,#fcfbf8 0%,var(--bg) 100%);
      color: var(--ink);
      font-family: var(--serif);
      line-height: 1.5;
    }
    .page { width: min(1200px,calc(100vw - 48px)); margin: 18px auto 72px; display: grid; gap: 24px; }
    .masthead {
      display: flex; justify-content: space-between; align-items: end; gap: 18px;
      padding: 10px 0 18px;
      border-top: 1px solid var(--line-strong);
      border-bottom: 3px double var(--line);
    }
    .masthead-brand { font-size: clamp(28px,4vw,44px); font-weight: 700; letter-spacing: -.035em; }
    .masthead-meta {
      font-family: var(--sans); font-size: 11px; color: var(--muted);
      text-transform: uppercase; letter-spacing: .15em; text-align: right; line-height: 1.6;
    }
    .hero {
      background: var(--paper); border: 1px solid var(--line); border-top: 4px solid var(--ink);
      box-shadow: var(--shadow); padding: 36px clamp(20px,4vw,48px) 30px; position: relative; overflow: hidden;
    }
    .hero::after {
      content:""; position:absolute; inset: auto -10% -40% auto;
      width:42%; aspect-ratio:1;
      background: radial-gradient(circle,rgba(50,104,145,.12),transparent 68%);
      pointer-events:none;
    }
    .eyebrow {
      position:relative; z-index:1; color:var(--accent); font-family:var(--sans);
      font-size:11px; letter-spacing:.18em; text-transform:uppercase; font-weight:700; margin-bottom:12px;
    }
    h1 {
      position:relative; z-index:1; margin:0 0 14px; font-size:clamp(38px,5.5vw,70px);
      line-height:.97; letter-spacing:-.05em; font-weight:700; max-width:14ch; text-wrap:balance;
    }
    .lede { position:relative; z-index:1; color:#464038; max-width:56rem; font-size:clamp(17px,2vw,22px); line-height:1.62; }
    .rail {
      position:relative; z-index:1; margin-top:24px; padding-top:16px;
      border-top:1px solid var(--line);
      display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px;
      font-family:var(--sans); font-size:11px; text-transform:uppercase; letter-spacing:.14em; color:var(--muted);
    }
    .rail > div { padding:10px 12px; border-top:1px solid var(--line-strong); background:rgba(255,255,255,.6); }
    .rail .val { font-family:var(--sans); font-size:24px; font-weight:800; color:var(--ink); letter-spacing:-.02em; line-height:1.1; margin-top:4px; }
    .grid2 { display:grid; grid-template-columns:minmax(0,1.7fr) minmax(280px,.9fr); gap:24px; }
    .card {
      background:var(--paper); border:1px solid var(--line); border-radius:var(--radius);
      box-shadow:var(--shadow); padding:24px clamp(16px,2.2vw,28px);
    }
    article.card { border-top: 3px solid var(--ink); }
    aside.card { border-top: 3px solid var(--accent); align-self:start; position:sticky; top:20px; }
    .section-title {
      margin:0 0 14px; padding-bottom:8px; border-bottom:1px solid var(--line);
      font-family:var(--sans); font-size:11px; text-transform:uppercase; letter-spacing:.16em; color:var(--muted);
    }
    .stat-grid { display:grid; gap:10px; }
    .stat { padding:14px 16px; border:1px solid var(--line); border-left:4px solid var(--accent); background:var(--paper); }
    .stat-label { color:var(--muted); font-family:var(--sans); font-size:10px; text-transform:uppercase; letter-spacing:.16em; margin-bottom:6px; }
    .stat-value { font-family:var(--sans); font-size:26px; font-weight:800; line-height:1.05; }
    .copula-table { width:100%; border-collapse:collapse; font-size:13px; }
    .copula-table th, .copula-table td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    .copula-table th { font-family:var(--sans); font-size:10px; text-transform:uppercase; letter-spacing:.14em; color:var(--muted); background:#f8f5ef; }
    .copula-table tr.best-row { background:rgba(50,104,145,.07); }
    .copula-table tr.best-row td:first-child { font-weight:700; color:var(--accent); }
    .chip {
      display:inline-flex; align-items:center; justify-content:center; min-width:64px;
      padding:4px 10px; border-radius:999px; font-family:var(--sans); font-size:10px; font-weight:800;
      letter-spacing:.12em; text-transform:uppercase;
    }
    .chip-frank { background:rgba(50,104,145,.12); color:var(--accent); border:1px solid rgba(50,104,145,.2); }
    .chip-clayton { background:rgba(33,108,83,.1); color:var(--good); border:1px solid rgba(33,108,83,.18); }
    .chip-gumbel { background:rgba(160,106,18,.1); color:var(--warn); border:1px solid rgba(160,106,18,.2); }
    canvas { display:block; max-width:100%; border:1px solid var(--line); border-radius:4px; }
    .bar-wrap { display:grid; gap:8px; }
    .bar-item { display:grid; grid-template-columns:140px 1fr 70px; align-items:center; gap:8px; font-family:var(--sans); font-size:12px; }
    .bar-bg { height:12px; background:#eee; border-radius:2px; overflow:hidden; }
    .bar-fill { height:100%; border-radius:2px; background:var(--accent); transition:width .4s ease; }
    .bar-val { text-align:right; color:var(--muted); font-size:11px; }
    .method-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin-top:4px; }
    .method-card { padding:14px 16px; border:1px solid var(--line); background:#faf8f4; border-left:3px solid var(--accent); }
    .method-title { font-family:var(--sans); font-size:10px; text-transform:uppercase; letter-spacing:.14em; color:var(--muted); margin-bottom:8px; }
    .method-body { font-family:var(--mono); font-size:12px; line-height:1.7; color:#214159; }
    @media(max-width:800px){ .grid2{grid-template-columns:1fr;} aside.card{position:static;} }
  </style>
</head>
<body>
<div class="page">

  <div class="masthead">
    <div class="masthead-brand">EvidenceCopula</div>
    <div class="masthead-meta">Cochrane Meta-Analysis Dependence Study<br>n = __N__ reviews &mdash; Frank copula selected</div>
  </div>

  <!-- HERO -->
  <div class="hero">
    <div class="eyebrow">Copula-Based Dependence Analysis</div>
    <h1>Trust &amp; Significance Are Inversely Linked</h1>
    <p class="lede">
      In __N__ Cochrane meta-analyses, trust score (robustness &times; quality) and statistical
      significance (|z|) share a <strong>Frank copula</strong> with &theta;&nbsp;=&nbsp;__THETA__
      (Kendall&rsquo;s &tau;&nbsp;=&nbsp;__TAU__, p&nbsp;&lt;&nbsp;0.001), confirming symmetric
      negative dependence with <em>no tail concentration</em>.
    </p>
    <div class="rail">
      <div>Best Family<div class="val">Frank</div></div>
      <div>&theta; (Frank)<div class="val">__THETA__</div></div>
      <div>Kendall&rsquo;s &tau;<div class="val">__TAU__</div></div>
      <div>Spearman &rho;<div class="val">__RHO__</div></div>
      <div>&lambda;<sub>L</sub> (lower tail)<div class="val">0.00</div></div>
      <div>&lambda;<sub>U</sub> (upper tail)<div class="val">0.00</div></div>
    </div>
  </div>

  <div class="grid2">
    <!-- MAIN CARD: Scatter -->
    <article class="card">
      <div class="section-title">Pseudo-Observations Scatter &mdash; (U, V) = (rank trust, rank |z|)</div>
      <canvas id="scatterCanvas" width="680" height="420"></canvas>
      <p style="margin:12px 0 0;font-family:var(--sans);font-size:12px;color:var(--muted);">
        Each point is one Cochrane meta-analysis. U&nbsp;=&nbsp;empirical CDF of final trust score;
        V&nbsp;=&nbsp;empirical CDF of |z-statistic|. Diagonal reference line = independence.
        The negative slope confirms inverse dependence.
      </p>
    </article>

    <!-- ASIDE: Copula comparison -->
    <aside class="card">
      <div class="section-title">Copula Model Comparison</div>
      <div class="stat-grid" style="margin-bottom:18px;">
        <div class="stat">
          <div class="stat-label">Selected Model (lowest AIC)</div>
          <div class="stat-value" style="color:var(--accent);">Frank</div>
        </div>
        <div class="stat" style="border-left-color:var(--good);">
          <div class="stat-label">AIC (Frank)</div>
          <div class="stat-value">__AIC__</div>
        </div>
      </div>
      <table class="copula-table">
        <thead>
          <tr><th>Family</th><th>&theta;</th><th>LogL</th><th>AIC</th><th>&lambda;<sub>L</sub></th><th>&lambda;<sub>U</sub></th></tr>
        </thead>
        <tbody id="copulaTableBody"></tbody>
      </table>
    </aside>
  </div>

  <!-- TAIL DEPENDENCE BY DOMAIN -->
  <div class="card" style="border-top:3px solid var(--ink);">
    <div class="section-title">Kendall&rsquo;s &tau; by Domain</div>
    <div class="bar-wrap" id="domainBars"></div>
  </div>

  <!-- METHODOLOGY -->
  <div class="card" style="border-top:3px solid var(--accent);">
    <div class="section-title">Methodology</div>
    <div class="method-grid">
      <div class="method-card">
        <div class="method-title">Marginals</div>
        <div class="method-body">U = rank(score)/(n+1)<br>V = rank(|z|)/(n+1)<br>z = |Phi<sup>-1</sup>(p/2)|</div>
      </div>
      <div class="method-card">
        <div class="method-title">Clayton Copula</div>
        <div class="method-body">C = (u<sup>-&theta;</sup>+v<sup>-&theta;</sup>-1)<sup>-1/&theta;</sup><br>&lambda;<sub>L</sub> = 2<sup>-1/&theta;</sup><br>&theta; &gt; 0</div>
      </div>
      <div class="method-card">
        <div class="method-title">Frank Copula</div>
        <div class="method-body">Symmetric, no tail dependence<br>&lambda;<sub>L</sub> = &lambda;<sub>U</sub> = 0<br>&theta; &isin; &#8477; &#8726; {0}</div>
      </div>
      <div class="method-card">
        <div class="method-title">Gumbel Copula</div>
        <div class="method-body">C = exp(-((-ln u)<sup>&theta;</sup>+(-ln v)<sup>&theta;</sup>)<sup>1/&theta;</sup>)<br>&lambda;<sub>U</sub> = 2&minus;2<sup>1/&theta;</sup><br>&theta; &ge; 1</div>
      </div>
      <div class="method-card">
        <div class="method-title">Selection</div>
        <div class="method-body">AIC = 2k &minus; 2&#8467;<br>MLE via scipy.optimize<br>Best: lowest AIC</div>
      </div>
      <div class="method-card">
        <div class="method-title">Data Source</div>
        <div class="method-body">403 Cochrane MAs<br>Trust = 0.6&times;robustness +<br>0.4&times;quality score</div>
      </div>
    </div>
  </div>

</div>

<script>
const DATA = __JSON_DATA__;

// ---- Scatter plot ----
(function() {
  const canvas = document.getElementById('scatterCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const PAD = 52;
  const plotW = W - 2*PAD, plotH = H - 2*PAD;

  // Domain colour map
  const domainColours = {
    'Generic Inverse Variance': 'rgba(50,104,145,0.35)',
    'Binary Outcomes': 'rgba(33,108,83,0.55)',
    'Sensitivity Analysis': 'rgba(160,106,18,0.55)',
    'Bias-Corrected': 'rgba(146,43,33,0.55)',
    'Continuous Outcomes': 'rgba(100,50,150,0.55)',
    'Mixed/Other': 'rgba(100,100,100,0.55)',
  };

  function toX(u) { return PAD + u * plotW; }
  function toY(v) { return PAD + (1-v) * plotH; }

  // Background
  ctx.fillStyle = '#fbfaf7';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#e8e0d8';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const x = PAD + (i/4)*plotW;
    const y = PAD + (i/4)*plotH;
    ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, PAD+plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(PAD+plotW, y); ctx.stroke();
  }

  // Independence diagonal
  ctx.strokeStyle = 'rgba(146,43,33,0.4)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6,4]);
  ctx.beginPath();
  ctx.moveTo(PAD, PAD+plotH);
  ctx.lineTo(PAD+plotW, PAD);
  ctx.stroke();
  ctx.setLineDash([]);

  // Points
  const u = DATA.scatter_u, v = DATA.scatter_v, domains = DATA.scatter_domains;
  for (let i = 0; i < u.length; i++) {
    const colour = domainColours[domains[i]] || 'rgba(50,104,145,0.35)';
    ctx.beginPath();
    ctx.arc(toX(u[i]), toY(v[i]), 3.5, 0, 2*Math.PI);
    ctx.fillStyle = colour;
    ctx.fill();
  }

  // Axes
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD, PAD); ctx.lineTo(PAD, PAD+plotH); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD, PAD+plotH); ctx.lineTo(PAD+plotW, PAD+plotH); ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#5f5a53';
  ctx.font = '12px "Segoe UI",sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('U  (trust score pseudo-observation)', PAD + plotW/2, H - 8);
  ctx.save();
  ctx.translate(14, PAD + plotH/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('V  (|z| pseudo-observation)', 0, 0);
  ctx.restore();

  // Tick labels
  ctx.fillStyle = '#888';
  ctx.font = '10px "Segoe UI",sans-serif';
  for (let i = 0; i <= 4; i++) {
    const val = (i/4).toFixed(2);
    ctx.textAlign = 'center';
    ctx.fillText(val, PAD + (i/4)*plotW, PAD+plotH+14);
    ctx.textAlign = 'right';
    ctx.fillText(val, PAD-6, PAD + (1-i/4)*plotH + 4);
  }
})();

// ---- Copula table ----
(function() {
  const tbody = document.getElementById('copulaTableBody');
  const bestFamily = DATA.overall.best_copula;
  DATA.all_copulas.forEach(c => {
    const isBest = c.family === bestFamily;
    const chipClass = 'chip chip-' + c.family.toLowerCase();
    const row = document.createElement('tr');
    if (isBest) row.className = 'best-row';
    row.innerHTML = '<td><span class="' + chipClass + '">' + c.family + '</span>' + (isBest?' ★':'') + '</td>'
      + '<td>' + c.theta.toFixed(3) + '</td>'
      + '<td>' + c.loglik.toFixed(1) + '</td>'
      + '<td>' + c.aic.toFixed(1) + '</td>'
      + '<td>' + c.lambda_L.toFixed(3) + '</td>'
      + '<td>' + c.lambda_U.toFixed(3) + '</td>';
    tbody.appendChild(row);
  });
})();

// ---- Domain bars ----
(function() {
  const container = document.getElementById('domainBars');
  const rows = DATA.domain_rows.sort((a,b) => a.kendall_tau - b.kendall_tau);
  const maxAbs = Math.max(...rows.map(r => Math.abs(r.kendall_tau)));

  rows.forEach(r => {
    const pct = maxAbs > 0 ? Math.abs(r.kendall_tau) / maxAbs * 100 : 0;
    const colour = r.kendall_tau < 0 ? 'var(--bad)' : 'var(--good)';
    const div = document.createElement('div');
    div.className = 'bar-item';
    div.innerHTML = '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;" title="'
      + r.group + '">' + r.group + ' (n=' + r.n + ')</span>'
      + '<div class="bar-bg"><div class="bar-fill" style="width:' + pct.toFixed(1) + '%;background:' + colour + ';"></div></div>'
      + '<span class="bar-val" style="color:' + colour + ';">' + r.kendall_tau.toFixed(3) + '</span>';
    container.appendChild(div);
  });
})();
</script>
</body>
</html>
"""


def build_dashboard(data: dict, out_path: str) -> None:
    """Fill template placeholders and write dashboard HTML."""
    overall = data["overall"]

    html = HTML_TEMPLATE
    html = html.replace("__N__", str(overall["n"]))
    html = html.replace("__THETA__", f"{overall['theta']:.3f}")
    html = html.replace("__TAU__", f"{overall['kendall_tau']:.3f}")
    html = html.replace("__RHO__", f"{overall['spearman_rho']:.3f}")
    html = html.replace("__AIC__", f"{overall['aic']:.1f}")
    html = html.replace("__JSON_DATA__", json.dumps(data, separators=(",", ":")))

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Dashboard written to {out_path}")


if __name__ == "__main__":
    print("Running EvidenceCopula pipeline...")
    data = run_pipeline()
    print(f"Overall: best={data['overall']['best_copula']}, "
          f"theta={data['overall']['theta']}, "
          f"kendall_tau={data['overall']['kendall_tau']}, "
          f"AIC={data['overall']['aic']}")
    build_dashboard(data, "dashboard.html")
    print("Done.")
