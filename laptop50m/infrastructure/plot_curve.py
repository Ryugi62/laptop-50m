"""Dependency-free SVG of validation loss vs. training tokens from runs/<run>/train_log.jsonl.

    python -m laptop50m.infrastructure.plot_curve runs/l50m-v1/train_log.jsonl results/loss_curve.svg
"""
from __future__ import annotations

import json
import sys

SERIES = [("val_loss", "FineWeb-Edu val", "#3182f6"), ("wikitext103_val_loss", "WikiText-103 val", "#f04452")]


def loss_curve_svg(log_path: str, w: int = 720, h: int = 360) -> str:
    recs = [json.loads(l) for l in open(log_path) if l.strip()]
    pts = {k: [(r["step"], r[k]) for r in recs if k in r] for k, _, _ in SERIES}
    xs = [s for v in pts.values() for s, _ in v]
    ys = [y for v in pts.values() for _, y in v]
    x0, x1, y0, y1 = 0, max(xs), min(ys) - 0.1, max(ys) + 0.1
    L, R, T, B = 60, 20, 20, 50
    X = lambda s: L + (s - x0) / (x1 - x0 or 1) * (w - L - R)
    Y = lambda v: T + (y1 - v) / (y1 - y0 or 1) * (h - T - B)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
           f'font-family="sans-serif" font-size="12"><rect width="{w}" height="{h}" fill="#fff"/>',
           f'<line x1="{L}" y1="{h-B}" x2="{w-R}" y2="{h-B}" stroke="#888"/>',
           f'<line x1="{L}" y1="{T}" x2="{L}" y2="{h-B}" stroke="#888"/>',
           f'<text x="{(w)/2}" y="{h-12}" text-anchor="middle">optimizer step (32,768 tokens/step)</text>',
           f'<text x="14" y="{h/2}" transform="rotate(-90 14 {h/2})" text-anchor="middle">loss (nats/token)</text>']
    for i in range(5):
        v = y0 + (y1 - y0) * i / 4
        out.append(f'<text x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
        s = x1 * i / 4
        out.append(f'<text x="{X(s):.1f}" y="{h-B+16}" text-anchor="middle">{int(s)}</text>')
    for j, (k, label, color) in enumerate(SERIES):
        p = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in pts[k])
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{p}"/>')
        out.append(f'<text x="{w-R-150}" y="{T+16+16*j}" fill="{color}">{label}</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    open(sys.argv[2], "w").write(loss_curve_svg(sys.argv[1]))
