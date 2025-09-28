#!/usr/bin/env python3
"""
Benchmark Suite: Handwritten Digit Recognition Engine (Deep Learning Project 2)
Evaluates Deep Feedforward Neural Networks (128-64 ReLU & 50-50 Notebook MLP)
against Multinomial Softmax Regression across 10-class Optical Digit Matrices.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier


def render_svg_chart(results: list[dict], output_path: Path) -> None:
    """Generates a dark-theme vector SVG chart of 10-class digit classification metrics."""
    width, height = 880, 440
    metrics = ["macro_roc_auc", "accuracy", "macro_f1", "top2_accuracy"]
    metric_labels = {
        "macro_roc_auc": "Macro ROC-AUC (OvR)",
        "accuracy": "Top-1 Accuracy",
        "macro_f1": "Macro F1-Score",
        "top2_accuracy": "Top-2 Accuracy",
    }
    colors = {
        "macro_roc_auc": "#38bdf8",
        "accuracy": "#34d399",
        "macro_f1": "#a78bfa",
        "top2_accuracy": "#fbbf24",
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        "  <defs>",
        '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '      <stop offset="0%" stop-color="#0f172a"/>',
        '      <stop offset="100%" stop-color="#1e293b"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect width="{width}" height="{height}" rx="12" fill="url(#bgGrad)" stroke="#334155" stroke-width="1.5"/>',
        '  <text x="32" y="42" fill="#f8fafc" font-family="Segoe UI, Inter, sans-serif" font-size="20" font-weight="700">'
        "Handwritten Digit Recognition Engine — 10-Class Neural Benchmark</text>",
        '  <text x="32" y="66" fill="#94a3b8" font-family="Segoe UI, Inter, sans-serif" font-size="13">'
        "Optical Digit Tensors (Classes 0–9, N = 1,797) | 5-Fold Stratified Cross-Validation</text>",
    ]

    lx = 32
    for m in metrics:
        svg.append(f'  <rect x="{lx}" y="84" width="12" height="12" rx="3" fill="{colors[m]}"/>')
        svg.append(
            f'  <text x="{lx + 18}" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" '
            f'font-size="12" font-weight="600">{metric_labels[m]}</text>'
        )
        lx += 200

    chart_left = 290
    chart_right = 810
    chart_width = chart_right - chart_left
    top_y = 125
    group_height = 94

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = chart_left + tick * chart_width
        svg.append(
            f'  <line x1="{gx:.1f}" y1="{top_y - 10}" x2="{gx:.1f}" y2="{top_y + 3 * group_height - 10}" '
            f'stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>'
        )
        svg.append(
            f'  <text x="{gx:.1f}" y="{top_y + 3 * group_height + 10}" fill="#64748b" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="11" text-anchor="middle">{tick:.2f}</text>'
        )

    for idx, entry in enumerate(results):
        gy = top_y + idx * group_height
        model_name = entry["model"]
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 32}" fill="#f1f5f9" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="12.5" font-weight="600" text-anchor="end">{model_name}</text>'
        )
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 50}" fill="#64748b" '
            f'font-family="Consolas, monospace" font-size="11" text-anchor="end">Cross-Entropy Loss: {entry["log_loss"]:.4f}</text>'
        )
        for m_idx, m in enumerate(metrics):
            val = entry[m]
            bar_w = max(2.0, val * chart_width)
            by = gy + m_idx * 18
            svg.append(
                f'  <rect x="{chart_left}" y="{by}" width="{bar_w:.1f}" height="14" rx="3" fill="{colors[m]}" opacity="0.92"/>'
            )
            svg.append(
                f'  <text x="{chart_left + bar_w + 8:.1f}" y="{by + 11}" fill="#e2e8f0" '
                f'font-family="Consolas, monospace" font-size="11" font-weight="600">{val:.4f}</text>'
            )

    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    digits = load_digits()
    # Normalize pixel intensities to [0, 1] matching notebook preprocessing (X / max_val)
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Deep MLP (128-64 ReLU + Adam + L2)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.002,
            max_iter=400,
            random_state=42,
        ),
        "Two-Layer MLP (Notebook 50-50 ReLU)": MLPClassifier(
            hidden_layer_sizes=(50, 50),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=350,
            random_state=3,
        ),
        "Multinomial Softmax Baseline": LogisticRegression(
            C=0.5, max_iter=500, random_state=42
        ),
    }

    results = []
    for name, model in models.items():
        oof_probs = np.zeros((len(y), 10), dtype=np.float64)
        oof_preds = np.zeros(len(y), dtype=int)

        for train_idx, val_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            probs = model.predict_proba(X[val_idx])
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = np.argmax(probs, axis=1)

        results.append(
            {
                "model": name,
                "accuracy": round(float(accuracy_score(y, oof_preds)), 4),
                "top2_accuracy": round(float(top_k_accuracy_score(y, oof_probs, k=2)), 4),
                "macro_precision": round(float(precision_score(y, oof_preds, average="macro")), 4),
                "macro_recall": round(float(recall_score(y, oof_preds, average="macro")), 4),
                "macro_f1": round(float(f1_score(y, oof_preds, average="macro")), 4),
                "macro_roc_auc": round(float(roc_auc_score(y, oof_probs, multi_class="ovr", average="macro")), 4),
                "log_loss": round(float(log_loss(y, oof_probs)), 4),
            }
        )

    results.sort(key=lambda r: (r["accuracy"], r["macro_f1"]), reverse=True)

    root_dir = Path(__file__).resolve().parent.parent
    json_path = root_dir / "benchmarks" / "benchmark_results.json"
    svg_path = root_dir / "assets" / "benchmark_chart.svg"

    payload = {
        "project": "Handwritten Digit Recognition Engine (Deep Learning Project 2)",
        "dataset": "Optical Handwritten Digit Tensors (10 Classes, N=1797, Normalized [0,1])",
        "validation_protocol": "5-Fold Stratified Cross-Validation",
        "results": results,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_svg_chart(results, svg_path)
    print(f"[OK] Saved digit recognition benchmark JSON -> {json_path}")
    print(f"[OK] Saved digit recognition benchmark SVG  -> {svg_path}")


if __name__ == "__main__":
    main()
