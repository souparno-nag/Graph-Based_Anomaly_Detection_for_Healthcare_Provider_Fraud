"""Generate Fig. 1 (pipeline architecture) as SVG sized for one IEEE column (3.5 in).

All coordinates are in points (1/72 in), so the canvas is 252 pt wide.
"""
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
W = 252
FONT = "Arial, Helvetica, sans-serif"

INK = "#222222"
SUB = "#555555"
BOX_EDGE = "#A39E98"
ACCENT = "#7B2D42"          # burgundy: title, data band, flow rail
ACCENT_DARK = "#5E1F31"
BG = "#F4EFE6"
BG_EDGE = "#BFA98A"
NODE = {"provider": "#1E7F7A", "physician": "#C97A2B", "beneficiary": "#B0432E"}  # match Fig. 3

STAGES = {
    "prep":  dict(edge="#D9733F", label="#FCDCC9", panel="#FFF5EE"),
    "graph": dict(edge="#4F8F4A", label="#D8EDD2", panel="#F3F8F0"),
    "model": dict(edge="#C4921F", label="#FBE6B0", panel="#FFF9E9"),
    "ens":   dict(edge="#8A4F9E", label="#EBDCF1", panel="#F9F3FB"),
    "eval":  dict(edge="#857464", label="#E9E2D9", panel="#F9F6F2"),
}

parts = []
add = parts.append


def rect(x, y, w, h, fill, stroke, sw=0.9, r=4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, lines, size=6, weight="normal", fill=INK, anchor="middle", style="normal", lh=1.22):
    """Multi-line text; y is the baseline of the first line. A line may be a list of
    (string, dict(weight=..., fill=..., size=..., baseline=...)) runs."""
    if isinstance(lines, str):
        lines = [lines]
    out = [f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
           f'fill="{fill}" text-anchor="{anchor}" font-style="{style}">']
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * lh
        if isinstance(line, str):
            out.append(f'<tspan x="{x}" dy="{dy}">{line}</tspan>')
        else:
            out.append(f'<tspan x="{x}" dy="{dy}">')
            for s, a in line:
                attrs = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in a.items())
                out.append(f'<tspan {attrs}>{s}</tspan>')
            out.append("</tspan>")
    out.append("</text>")
    add("".join(out))


def arrow(x1, y1, x2, y2, color="#3F4650", sw=0.8, label=None):
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" '
        f'marker-end="url(#head)"/>')
    if label:
        text((x1 + x2) / 2, y1 - 2.2, label, size=4.6, fill=SUB)


def icon(kind, cx, cy, color, s=1.0):
    g = [f'<g transform="translate({cx},{cy}) scale({s})" fill="none" stroke="{color}" '
         f'stroke-width="0.9" stroke-linecap="round" stroke-linejoin="round">']
    if kind == "table":
        g += ['<rect x="-5" y="-4" width="10" height="8" rx="1"/>',
              '<line x1="-5" y1="-1.3" x2="5" y2="-1.3"/>', '<line x1="-5" y1="1.4" x2="5" y2="1.4"/>',
              '<line x1="-1.7" y1="-4" x2="-1.7" y2="4"/>', '<line x1="1.7" y1="-4" x2="1.7" y2="4"/>']
    elif kind == "hist":
        g += ['<line x1="-5" y1="4" x2="5" y2="4"/>']
        for x, h in [(-4, 3), (-1.5, 7), (1, 5), (3.5, 2)]:
            g.append(f'<rect x="{x}" y="{4 - h}" width="1.8" height="{h}" fill="{color}" stroke="none"/>')
    elif kind == "sigma":
        g = [f'<g transform="translate({cx},{cy}) scale({s})">',
             f'<text x="0" y="3.6" font-family="{FONT}" font-size="11" font-weight="bold" '
             f'fill="{color}" text-anchor="middle">Σ</text>']
    elif kind == "graph":
        g = [f'<g transform="translate({cx},{cy}) scale({s})" stroke="#8A8A8A" stroke-width="0.6">']
        P = [(-2, -1), (5, 3)]
        B = [(-7, -5), (1, -6), (8, -3), (-6, 5)]
        H = [(2, 6), (-1, 4.5)]
        for (px, py) in P:
            for (qx, qy) in B + H:
                if abs(px - qx) + abs(py - qy) < 12:
                    g.append(f'<line x1="{px}" y1="{py}" x2="{qx}" y2="{qy}"/>')
        for pts, c, r in [(B, NODE["beneficiary"], 1.5), (H, NODE["physician"], 1.5), (P, NODE["provider"], 2.1)]:
            for (x, y) in pts:
                g.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" stroke="white" stroke-width="0.4"/>')
    elif kind == "community":
        g += [f'<circle cx="-3" cy="-1" r="4" stroke-dasharray="1.2 0.9"/>',
              f'<circle cx="4" cy="2" r="3.3" stroke-dasharray="1.2 0.9"/>']
        for (x, y) in [(-4.5, -2), (-1.5, 0.5), (-3, -3), (3.5, 1), (5, 3.2)]:
            g.append(f'<circle cx="{x}" cy="{y}" r="0.9" fill="{color}" stroke="none"/>')
    elif kind == "tree":
        g += ['<line x1="0" y1="-4" x2="-3.5" y2="0"/>', '<line x1="0" y1="-4" x2="3.5" y2="0"/>',
              '<line x1="-3.5" y1="0" x2="-5" y2="4"/>', '<line x1="-3.5" y1="0" x2="-2" y2="4"/>',
              '<line x1="3.5" y1="0" x2="2" y2="4"/>', '<line x1="3.5" y1="0" x2="5" y2="4"/>']
        for (x, y) in [(0, -4), (-3.5, 0), (3.5, 0), (-5, 4), (-2, 4), (2, 4), (5, 4)]:
            g.append(f'<circle cx="{x}" cy="{y}" r="1.1" fill="{color}" stroke="none"/>')
    elif kind == "gnn":
        L = [[(-5, -3.5), (-5, 0), (-5, 3.5)], [(0, -4.5), (0, -1.5), (0, 1.5), (0, 4.5)], [(5, 0)]]
        for a, b in zip(L, L[1:]):
            for (x1, y1) in a:
                for (x2, y2) in b:
                    g.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="0.35"/>')
        for layer in L:
            for (x, y) in layer:
                g.append(f'<circle cx="{x}" cy="{y}" r="1.1" fill="{color}" stroke="none"/>')
    elif kind == "merge":
        g += ['<path d="M-5,-4 C-1,-4 -1,0 2,0"/>', '<path d="M-5,0 L2,0"/>', '<path d="M-5,4 C-1,4 -1,0 2,0"/>',
              f'<path d="M2,-2 L5,0 L2,2 Z" fill="{color}"/>']
    elif kind == "gauge":
        g += ['<path d="M-5,3 A5,5 0 0 1 5,3"/>', '<line x1="0" y1="3" x2="2.8" y2="-1.4"/>',
              f'<circle cx="0" cy="3" r="0.9" fill="{color}" stroke="none"/>']
    elif kind == "bars":
        g += ['<line x1="-5" y1="4.5" x2="5" y2="4.5"/>']
        for x, h, o in [(-4.2, 3.5, 0.55), (-1, 6, 0.8), (2.2, 8.5, 1)]:
            g.append(f'<rect x="{x}" y="{4.5 - h}" width="2.2" height="{h}" fill="{color}" '
                     f'fill-opacity="{o}" stroke="none"/>')
    elif kind == "lens":
        g += ['<circle cx="-1" cy="-1" r="3.6"/>', '<line x1="1.7" y1="1.7" x2="5" y2="5" stroke-width="1.5"/>',
              '<line x1="-1" y1="-2.8" x2="-1" y2="-0.6"/>', f'<circle cx="-1" cy="0.6" r="0.35" fill="{color}"/>']
    elif kind == "doc":
        g += ['<rect x="-6" y="-7.5" width="10" height="13" rx="1.3" stroke="white" stroke-width="1.1"/>',
              '<rect x="-4" y="-6" width="10" height="13" rx="1.3" fill="white" stroke-width="1.1"/>']
        for yy in (-3, -0.5, 2):
            g.append(f'<line x1="-2" y1="{yy}" x2="4" y2="{yy}" stroke-width="1"/>')
        g.append(f'<circle cx="5" cy="6" r="3" fill="{color}" stroke="white" stroke-width="0.8"/>')
        g.append('<path d="M3.6,6 L4.6,7 L6.5,5" stroke="white" stroke-width="0.9"/>')
    g.append("</g>")
    add("".join(g))


def node_box(x, y, w, h, title, sub, ic, color, ic_s=0.95):
    rect(x, y, w, h, "white", BOX_EDGE, sw=0.7, r=3)
    icon(ic, x + w / 2, y + 8.5, color, ic_s)
    text(x + w / 2, y + 21, title, size=5.6, weight="bold", lh=1.15)
    if sub:
        n_title = 1 if isinstance(title, str) else len(title)
        text(x + w / 2, y + 21 + n_title * 5.6 * 1.15 + 0.6, sub, size=4.8, fill=SUB, lh=1.18)


# ---------------------------------------------------------------- layout
M = 7                     # inner margin
RAIL_X, RAIL_W = 10, 46   # stage label column
PX = RAIL_X + RAIL_W + 7  # panel left edge
PW = W - M - 3 - PX       # panel width
ROW_H, ROW_GAP = 54, 9

band_y = 9
band_h = 26
rows_y0 = band_y + band_h + 13
row_ys = [rows_y0 + i * (ROW_H + ROW_GAP) for i in range(5)]
H = row_ys[-1] + ROW_H + 9

# background
rect(2, 2, W - 4, H - 4, BG, BG_EDGE, sw=1.1, r=8)

# data band
rect(M, band_y, W - 2 * M, band_h, ACCENT, ACCENT_DARK, sw=0.9, r=4)
add(f'<circle cx="{M + 14}" cy="{band_y + band_h / 2}" r="9.5" fill="white"/>')
icon("doc", M + 14, band_y + band_h / 2, ACCENT, 0.72)
text(M + 28, band_y + 11.2, ["Healthcare Provider", "Fraud Dataset"], size=6.3, weight="bold",
     fill="white", anchor="start", lh=1.18)
pill_w, pill_h = 41, 8.6
for i, name in enumerate(["Inpatient", "Outpatient", "Beneficiary", "Provider labels"]):
    px = W - M - 5 - (2 - i % 2) * pill_w - (1 - i % 2) * 3
    py = band_y + 3.4 + (i // 2) * (pill_h + 1.8)
    rect(px, py, pill_w, pill_h, "white", "white", sw=0, r=pill_h / 2)
    text(px + pill_w / 2, py + 6, name, size=4.7, weight="bold", fill=ACCENT)

# flow rail
rail_cx = RAIL_X + RAIL_W / 2
add(f'<line x1="{rail_cx}" y1="{band_y + band_h + 1}" x2="{rail_cx}" y2="{row_ys[0] + 10}" '
    f'stroke="{ACCENT}" stroke-width="3.4" marker-end="url(#rail)"/>')

labels = [("prep", ["DATA", "PREPARATION"]), ("graph", ["GRAPH", "CONSTRUCTION"]),
          ("model", ["MODEL", "TRAINING"]), ("ens", ["ENSEMBLE &", "DECISION"]),
          ("eval", ["EVALUATION &", "EXPLAINABILITY"])]
lab_h = 22
for i, ((key, lab), ry) in enumerate(zip(labels, row_ys)):
    c = STAGES[key]
    ly = ry + ROW_H / 2 - lab_h / 2
    if i < 4:
        add(f'<line x1="{rail_cx}" y1="{ly + lab_h + 1}" x2="{rail_cx}" '
            f'y2="{row_ys[i + 1] + ROW_H / 2 - lab_h / 2 - 7.5}" stroke="{ACCENT}" stroke-width="3.4" '
            f'marker-end="url(#rail)"/>')
    rect(RAIL_X, ly, RAIL_W, lab_h, c["label"], c["edge"], sw=1.0, r=3)
    text(rail_cx, ly + 9.2, lab, size=4.9, weight="bold", lh=1.25)
    rect(PX, ry, PW, ROW_H, c["panel"], c["edge"], sw=1.0, r=4)
    # connector from label to panel
    add(f'<line x1="{RAIL_X + RAIL_W}" y1="{ry + ROW_H / 2}" x2="{PX}" y2="{ry + ROW_H / 2}" '
        f'stroke="{c["edge"]}" stroke-width="0.8" stroke-dasharray="1.6 1.2"/>')

bw_gap = 9
pad = 5


def three_boxes(ry, specs, color, labels=(None, None), top=5, h=37):
    bw = (PW - 2 * pad - 2 * bw_gap) / 3
    xs = [PX + pad + i * (bw + bw_gap) for i in range(3)]
    for x, (t, s, ic) in zip(xs, specs):
        node_box(x, ry + top, bw, h, t, s, ic, color)
    for i in range(2):
        arrow(xs[i] + bw, ry + top + h / 2, xs[i + 1] - 0.6, ry + top + h / 2, label=labels[i])
    return xs, bw


# 1. data preparation
ry = row_ys[0]; c = STAGES["prep"]["edge"]
three_boxes(ry, [("Merge &amp; clean", ["inpatient + outpatient", "+ beneficiary + labels"], "table"),
                 ("EDA", ["reimbursement,", "claim volume, states"], "hist"),
                 ("Aggregate", ["12 provider-level", "billing features"], "sigma")], c)
text(PX + PW / 2, ry + ROW_H - 4.6, "claim-level → provider-level · missing physician kept as “None”",
     size=4.6, fill=SUB, style="italic")

# 2. graph construction
ry = row_ys[1]; c = STAGES["graph"]["edge"]
bw = (PW - 2 * pad - bw_gap) / 2
gx = [PX + pad, PX + pad + bw + bw_gap]
node_box(gx[0], ry + 5, bw, 37, "Heterogeneous graph",
         ["provider–beneficiary +", "provider–physician links"], "graph", c, ic_s=0.82)
node_box(gx[1], ry + 5, bw, 37, "Graph features",
         ["Louvain community ID", "2-hop fraud-neighbour density"], "community", c)
arrow(gx[0] + bw, ry + 5 + 18.5, gx[1] - 0.6, ry + 5 + 18.5)
leg_y = ry + ROW_H - 5.6
lx = PX + 26
text(lx, leg_y, "226,029 nodes:", size=4.6, fill=SUB, anchor="start", style="italic")
lx += 38
for name in ["provider", "physician", "beneficiary"]:
    add(f'<circle cx="{lx}" cy="{leg_y - 1.6}" r="1.7" fill="{NODE[name]}"/>')
    text(lx + 3.2, leg_y, name, size=4.6, fill=SUB, anchor="start", style="italic")
    lx += 3.2 + len(name) * 2.3 + 9

# 3. model training
ry = row_ys[2]; c = STAGES["model"]["edge"]
bw = (PW - 2 * pad - 2 * 6) / 3
mx = [PX + pad + i * (bw + 6) for i in range(3)]
node_box(mx[0], ry + 5, bw, 37, "XGBoost", ["baseline:", "12 tabular features"], "tree", c)
node_box(mx[1], ry + 5, bw, 37, "XGBoost", ["+ graph: 12 tabular", "+ 2 graph features"], "tree", c)
node_box(mx[2], ry + 5, bw, 37, "Hetero GNN", ["GraphSAGE · h = 64", "4 relation types"], "gnn", c)
text(PX + PW / 2, ry + ROW_H - 4.6, "stratified 80/20 split · class weighting · early stopping on 15% val. split",
     size=4.6, fill=SUB, style="italic")

# 4. ensemble & decision
ry = row_ys[3]; c = STAGES["ens"]["edge"]
bx = PX + pad
score_w = 30
for i, s in enumerate(["base", "graph", "GNN"]):
    sy = ry + 6 + i * 11.6
    rect(bx, sy, score_w, 9.2, "white", BOX_EDGE, sw=0.7, r=2.5)
    text(bx + score_w / 2, sy + 6.3, [[("p", {}), (s, dict(font_size=3.9, baseline_shift="sub"))]],
         size=5.4, weight="bold")
ex = bx + score_w + 9
ew = 78
node_box(ex, ry + 5, ew, 37, "Combine scores",
         ["average · PR-AUC-weighted ·", "logistic-regression stacking"], "merge", c)
for i in range(3):
    sy = ry + 6 + i * 11.6 + 4.6
    add(f'<line x1="{bx + score_w}" y1="{sy}" x2="{ex - 0.6}" y2="{ry + 5 + 18.5}" stroke="#3F4650" '
        f'stroke-width="0.6" marker-end="url(#head)"/>' if i == 1 else
        f'<path d="M{bx + score_w},{sy} L{ex - 4.5},{sy} L{ex - 4.5},{ry + 23.5} L{ex - 0.6},{ry + 23.5}" '
        f'fill="none" stroke="#3F4650" stroke-width="0.6"/>')
dx = ex + ew + 9
dw = PX + PW - pad - dx
node_box(dx, ry + 5, dw, 37, "Fraud flag", ["threshold τ = 0.35", "(recall-oriented)"], "gauge", c)
arrow(ex + ew, ry + 23.5, dx - 0.6, ry + 23.5)
text(PX + PW / 2, ry + ROW_H - 4.6, "stacker fitted on one half of the test providers, scored on the other half",
     size=4.6, fill=SUB, style="italic")

# 5. evaluation & explainability
ry = row_ys[4]; c = STAGES["eval"]["edge"]
bw = (PW - 2 * pad - bw_gap) / 2
vx = [PX + pad, PX + pad + bw + bw_gap]
for x, ic, t, s in [(vx[0], "bars", "Performance", ["AUC-ROC · AUC-PR", "recall · precision · F1", "5-fold CV (XGBoost)"]),
                    (vx[1], "lens", "Explainability", ["SHAP (XGBoost models)", "GNNExplainer", "relation ablation (GNN)"])]:
    rect(x, ry + 6, bw, ROW_H - 12, "white", BOX_EDGE, sw=0.7, r=3)
    icon(ic, x + 10, ry + ROW_H / 2, "#5A4E44" if ic == "bars" else "#7B2D42", 1.15)
    text(x + 21, ry + 18.5, t, size=5.6, weight="bold", anchor="start")
    text(x + 21, ry + 26.5, s, size=4.8, fill=SUB, anchor="start", lh=1.3)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}pt" height="{H}pt">
<defs>
<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
<path d="M0,0 L10,5 L0,10 Z" fill="#3F4650"/></marker>
<marker id="rail" viewBox="0 0 10 10" refX="4" refY="5" markerWidth="2.6" markerHeight="2.6" orient="auto">
<path d="M0,0 L10,5 L0,10 Z" fill="{ACCENT}"/></marker>
</defs>
{chr(10).join(parts)}
</svg>
'''
OUT.write_text(svg)
print(f"{W}x{H:.1f} pt = {W / 72:.2f} x {H / 72:.2f} in")
