# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Render a computation graph dump JSON as an interactive HTML page."""

import argparse
import json
import sys
from pathlib import Path


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _render_tensor(t: dict) -> str:
    shape = "×".join(str(d) for d in t.get("shape", []))
    return f'<span class="shape">{shape}</span> <span class="dtype">{t.get("dtype", "")}</span>'


def _render_op_table(ops: list[dict], layer_name: str, start_idx: int) -> str:
    rows = []
    for i, op in enumerate(ops):
        idx = start_idx + i
        sub = op["metadata"].get("subgraph", "other")
        name = _escape(op.get("display_name", op.get("op_name", "")))
        op_name = _escape(op.get("op_name", ""))
        mod_type = _escape(op["metadata"].get("module_type", ""))
        inputs_html = ", ".join(_render_tensor(t) for t in op.get("inputs", []))
        outputs_html = ", ".join(_render_tensor(t) for t in op.get("outputs", []))
        rows.append(f"""<tr class="sub-{sub}">
            <td class="id-cell">{idx}</td>
            <td class="op-name" title="{op_name}">{name}</td>
            <td class="mod-type">{mod_type}</td>
            <td class="shape-cell">{inputs_html or "—"}</td>
            <td class="shape-cell">{outputs_html or "—"}</td>
            <td class="sub-{sub}"><span class="sub-tag">{sub}</span></td>
        </tr>""")
    return "\n".join(rows)


def _count_by(ops: list[dict], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in ops:
        k = key_fn(op)
        counts[k] = counts.get(k, 0) + 1
    return counts


def render(json_path: str, output_path: str) -> None:
    with open(json_path) as f:
        dump = json.load(f)

    mode = dump.get("mode", "unknown")
    model_name = dump.get("model_name", "unknown")
    config = dump.get("model_config", {})
    layers = dump.get("layers", [])

    # Collect all ops and stats
    all_ops = []
    for layer in layers:
        all_ops.extend(layer.get("ops", []))

    sub_counts = _count_by(all_ops, lambda o: o["metadata"].get("subgraph", "other"))
    op_name_counts = _count_by(all_ops, lambda o: o.get("op_name", "?"))

    # Summary
    total_ops = len(all_ops)
    unique_names = len(op_name_counts)
    layer_count = len(layers)
    total_layer_instances = sum(l.get("count", 1) for l in layers)

    # Pie chart data for subgraph distribution
    sub_labels = list(sub_counts.keys())
    sub_values = list(sub_counts.values())
    colors = {
        "attention": "#4ecdc4", "mlp": "#ff6b6b", "norm": "#45b7d1",
        "embedding": "#f9ca24", "other": "#95a5a6", "kvcache": "#e056a0",
        "rope": "#9b59b6", "split": "#6c5ce7", "memory": "#dfe6e9",
    }
    sub_colors = [colors.get(s, "#95a5a6") for s in sub_labels]

    # Bar chart for top ops
    top_n = 15
    top_ops = sorted(op_name_counts.items(), key=lambda x: -x[1])[:top_n]

    # --- Collect layer info ---
    layer_rows = []
    for li, layer in enumerate(layers):
        l_ops = layer.get("ops", [])
        l_name = _escape(layer.get("name", str(li)))
        l_count = layer.get("count", 1)
        layer_rows.append(f"""<tr>
            <td>{l_name}</td>
            <td class="num">{l_count}</td>
            <td class="num">{len(l_ops)}</td>
        </tr>""")

    # --- Build HTML ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>vLLM Computation Graph — {_escape(model_name)} ({_escape(mode)})</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1 {{ font-size: 1.5em; margin-bottom: 4px; }}
h2 {{ font-size: 1.1em; margin: 24px 0 12px; color: #a0a0ff; }}
.meta {{ color: #888; font-size: 0.85em; margin-bottom: 20px; }}
/* Summary cards */
.cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
.card {{ background: #16213e; border-radius: 8px; padding: 16px 20px; min-width: 120px; }}
.card .value {{ font-size: 2em; font-weight: 700; color: #00d2ff; }}
.card .label {{ font-size: 0.8em; color: #999; }}
/* Charts row */
.charts {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 16px 0; }}
.chart-box {{ background: #16213e; border-radius: 8px; padding: 16px; }}
.chart-box h3 {{ font-size: 0.95em; color: #a0a0ff; margin-bottom: 10px; }}
/* Table */
.table-wrap {{ background: #16213e; border-radius: 8px; overflow: hidden; margin: 16px 0; }}
.toolbar {{ display: flex; gap: 10px; padding: 12px 16px; align-items: center; flex-wrap: wrap;
            background: #0f3460; }}
.toolbar input {{ padding: 6px 12px; border-radius: 4px; border: 1px solid #333;
                   background: #1a1a2e; color: #e0e0e0; min-width: 200px; }}
.toolbar select {{ padding: 6px 12px; border-radius: 4px; border: 1px solid #333;
                    background: #1a1a2e; color: #e0e0e0; }}
.toolbar .count {{ color: #888; font-size: 0.85em; margin-left: auto; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ background: #0f3460; padding: 10px 12px; text-align: left; font-size: 0.85em;
      color: #a0a0ff; position: sticky; top: 0; cursor: pointer; }}
th:hover {{ background: #1a4a7a; }}
td {{ padding: 7px 12px; border-top: 1px solid #2a2a4a; font-size: 0.85em; }}
tr:hover {{ background: rgba(255,255,255,0.03); }}
.id-cell {{ color: #666; width: 50px; text-align: right; }}
.op-name {{ max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.mod-type {{ color: #888; max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.shape-cell {{ max-width: 350px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.shape {{ color: #00d2ff; font-family: 'SF Mono', 'Cascadia Code', monospace; font-size: 0.85em; }}
.dtype {{ color: #888; font-size: 0.8em; }}
.sub-tag {{ padding: 1px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 600; }}
/* Subgraph colors */
.sub-attention {{ border-left: 3px solid #4ecdc4; }}
.sub-attention .sub-tag {{ background: #4ecdc440; color: #4ecdc4; }}
.sub-mlp {{ border-left: 3px solid #ff6b6b; }}
.sub-mlp .sub-tag {{ background: #ff6b6b40; color: #ff6b6b; }}
.sub-norm {{ border-left: 3px solid #45b7d1; }}
.sub-norm .sub-tag {{ background: #45b7d140; color: #45b7d1; }}
.sub-embedding {{ border-left: 3px solid #f9ca24; }}
.sub-embedding .sub-tag {{ background: #f9ca2440; color: #f9ca24; }}
.sub-other {{ border-left: 3px solid #95a5a6; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
/* Layer table */
.layers-table {{ margin: 16px 0; }}
</style>
</head>
<body>
<h1>vLLM Computation Graph</h1>
<div class="meta">
  Model: <strong>{_escape(model_name)}</strong> &nbsp;|&nbsp;
  Mode: <strong>{_escape(mode)}</strong> &nbsp;|&nbsp;
  Ops: <strong>{total_ops}</strong> &nbsp;|&nbsp;
  Unique op types: <strong>{unique_names}</strong>
</div>

<!-- Summary cards -->
<div class="cards">
  <div class="card"><div class="value">{total_ops}</div><div class="label">Total Operators</div></div>
  <div class="card"><div class="value">{unique_names}</div><div class="label">Unique Op Types</div></div>
  <div class="card"><div class="value">{layer_count}</div><div class="label">Layer Groups</div></div>
  <div class="card"><div class="value">{total_layer_instances}</div><div class="label">Total Layers</div></div>
</div>

<!-- Charts -->
<div class="charts">
  <div class="chart-box">
    <h3>Ops by Subgraph</h3>
    <canvas id="subgraphPie" width="300" height="300"></canvas>
  </div>
  <div class="chart-box">
    <h3>Top {top_n} Op Types</h3>
    <canvas id="opBar" width="500" height="300"></canvas>
  </div>
</div>

<!-- Layer summary -->
<h2>Layer Groups</h2>
<div class="table-wrap layers-table">
  <table><thead><tr>
    <th>Layer Name</th><th>Instances</th><th>Ops per Layer</th>
  </tr></thead><tbody>
    {"".join(layer_rows)}
  </tbody></table>
</div>

<!-- Operator table -->
<h2>Operators</h2>
<div class="table-wrap">
  <div class="toolbar">
    <input type="text" id="searchInput" placeholder="Filter ops..." oninput="filterOps()">
    <select id="subgraphFilter" onchange="filterOps()">
      <option value="all">All subgraphs</option>
      {"".join(f'<option value="{s}">{s} ({c})</option>' for s, c in sorted(sub_counts.items()))}
    </select>
    <span class="count" id="rowCount"></span>
  </div>
  <div style="max-height: 70vh; overflow-y: auto;">
  <table><thead><tr>
    <th onclick="sortTable(0)">#</th>
    <th onclick="sortTable(1)">Operator</th>
    <th onclick="sortTable(2)">Module Type</th>
    <th onclick="sortTable(3)">Inputs</th>
    <th onclick="sortTable(4)">Outputs</th>
    <th onclick="sortTable(5)">Subgraph</th>
  </tr></thead>
  <tbody id="opTableBody">
    {_render_op_table(all_ops, "", 0)}
  </tbody></table>
  </div>
</div>

<!-- Config -->
<h2>Model Config</h2>
<details>
  <summary style="cursor:pointer; color:#a0a0ff; margin:12px 0;">Show config</summary>
  <pre style="background:#16213e; padding:16px; border-radius:8px; overflow-x:auto;
              font-size:0.8em; color:#999;">{_escape(json.dumps(config, indent=2))}</pre>
</details>

<script>
// Pie chart
(() => {{
  const c = document.getElementById('subgraphPie');
  const ctx = c.getContext('2d');
  const data = {sub_values};
  const labels = {sub_labels};
  const colors = {sub_colors};
  const total = data.reduce((a,b) => a + b, 0);
  let start = -Math.PI / 2;
  const cx = 150, cy = 150, r = 120;
  data.forEach((v, i) => {{
    const slice = (v / total) * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, start, start + slice);
    ctx.closePath();
    ctx.fillStyle = colors[i];
    ctx.fill();
    // Label
    const mid = start + slice / 2;
    const lx = cx + Math.cos(mid) * (r + 30);
    const ly = cy + Math.sin(mid) * (r + 30);
    ctx.fillStyle = '#e0e0e0';
    ctx.font = '11px system-ui';
    ctx.textAlign = mid > Math.PI / 2 && mid < 3 * Math.PI / 2 ? 'right' : 'left';
    ctx.fillText(labels[i] + ' (' + v + ')', lx, ly);
    start += slice;
  }});
}})();

// Bar chart
(() => {{
  const c = document.getElementById('opBar');
  const ctx = c.getContext('2d');
  const data = {[v for _, v in top_ops]};
  const labels = {[repr(n.split(".")[-1][:20] for n, _ in top_ops)]};
  const max = Math.max(...data);
  const w = 480, h = 280, left = 110, top = 20, bw = (w - left) / data.length - 4;
  ctx.fillStyle = '#e0e0e0';
  ctx.font = '10px system-ui';
  data.forEach((v, i) => {{
    const barH = (v / max) * (h - top - 30);
    const x = left + i * (bw + 4);
    const y = h - 30 - barH;
    ctx.fillStyle = '#00d2ff';
    ctx.fillRect(x, y, bw, barH);
    ctx.fillStyle = '#e0e0e0';
    ctx.textAlign = 'right';
    ctx.fillText(labels[i], left - 8, h - 30 - barH / 2);
    ctx.textAlign = 'center';
    ctx.fillText(v, x + bw / 2, y - 4);
  }});
  // Axis
  ctx.strokeStyle = '#333';
  ctx.beginPath();
  ctx.moveTo(left, top); ctx.lineTo(left, h - 30); ctx.lineTo(w, h - 30);
  ctx.stroke();
}})();

// Filter
function filterOps() {{
  const search = document.getElementById('searchInput').value.toLowerCase();
  const sub = document.getElementById('subgraphFilter').value;
  const rows = document.querySelectorAll('#opTableBody tr');
  let visible = 0;
  rows.forEach(r => {{
    const text = r.textContent.toLowerCase();
    const matchSearch = !search || text.includes(search);
    const matchSub = sub === 'all' || r.classList.contains('sub-' + sub);
    r.style.display = (matchSearch && matchSub) ? '' : 'none';
    if (matchSearch && matchSub) visible++;
  }});
  document.getElementById('rowCount').textContent = 'Showing ' + visible + ' of ' + rows.length;
}}
document.getElementById('rowCount').textContent = 'Showing all {total_ops} ops';

// Sort
let sortDir = {{}};
function sortTable(col) {{
  const tbody = document.getElementById('opTableBody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  sortDir[col] = !sortDir[col];
  rows.sort((a, b) => {{
    let va = a.cells[col].textContent.trim();
    let vb = b.cells[col].textContent.trim();
    let na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return sortDir[col] ? na - nb : nb - na;
    return sortDir[col] ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"✓ HTML written to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render computation graph dump JSON as interactive HTML",
    )
    parser.add_argument("json_path", type=str, help="Path to dump JSON file")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output HTML path (default: <json_path>.html)",
    )
    args = parser.parse_args()

    output = args.output or str(Path(args.json_path).with_suffix(".html"))
    render(args.json_path, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
