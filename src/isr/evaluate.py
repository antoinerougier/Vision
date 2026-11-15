"""Benchmark the ONNX pipeline and (optionally) enforce the CI quality gate.

Usage:
    python -m isr.evaluate --config configs/default.yaml           # benchmark + report
    python -m isr.evaluate --config configs/default.yaml --gate    # exit 1 if below floors

For every degradation kind and severity we measure, on a held-out synthetic test set:

    AP@0.5 of the detector on the DEGRADED image          (what happens today)
    AP@0.5 of the detector on the RESTORED image          (with the pre-processing layer)
    PSNR / SSIM of degraded vs restored against the clean image

The evaluation runs on the exact `.onnx` artefacts that ship in the Docker image, through
the exact production code path (`OnnxPipeline`). What we test is what we deploy.

IMPORTANT caveat, repeated in the README: the test degradations come from the same
simulator as the training ones. These numbers prove the pipeline works and that the
restorer generalises to unseen *severities*; they do NOT prove a gain on real sensors.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classical import classical_enhance
from .common import load_config, sample_rng, write_json
from .degradations import KINDS, degrade
from .inference import OnnxPipeline
from .metrics import average_precision, psnr, ssim
from .synthetic import render_scene

GATE_SEVERITY = 1.0  # the severity at which the gate is evaluated


def _make_test_set(config: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate the held-out clean test images and their ground-truth boxes."""
    eval_cfg, size = config["evaluation"], config["data"]["image_size"]
    images, boxes = [], []
    for i in range(eval_cfg["n_images"]):
        image, gt = render_scene(sample_rng(eval_cfg["test_seed"], i), size)
        images.append(image)
        boxes.append(gt)
    return images, boxes


def _detect_all(
    pipeline: OnnxPipeline, images: list[np.ndarray], threshold: float
) -> list[np.ndarray]:
    return [
        pipeline.detect(img, restore=False, score_threshold=threshold) for img in images
    ]


def _measure_latency(pipeline: OnnxPipeline, images: list[np.ndarray]) -> float:
    """Average milliseconds per image for the full restore + detect path."""
    for img in images[:5]:  # warm-up: first runs allocate buffers and are slower
        pipeline.detect(img, restore=True)
    sample = images[:40]
    start = time.perf_counter()
    for img in sample:
        pipeline.detect(img, restore=True)
    return (time.perf_counter() - start) / len(sample) * 1000.0


def run_benchmark(config: dict[str, Any], model_dir: str | Path) -> dict[str, Any]:
    """Run the full benchmark and return a JSON-serialisable results dictionary."""
    eval_cfg = config["evaluation"]
    iou, ap_threshold = eval_cfg["iou_threshold"], eval_cfg["ap_score_threshold"]
    pipeline = OnnxPipeline(model_dir)
    clean_images, ground_truth = _make_test_set(config)

    # Optional baseline: the same detector trained WITH degradation augmentation, evaluated
    # directly on degraded images (no restoration layer). Present when `make train` produced it.
    baseline_pipeline = None
    if (Path(model_dir) / "detector_augmented.onnx").exists():
        baseline_pipeline = OnnxPipeline(
            model_dir, detector_file="detector_augmented.onnx"
        )

    clean_ap = average_precision(
        _detect_all(pipeline, clean_images, ap_threshold), ground_truth, iou
    )

    results: dict[str, dict[str, dict[str, float | None]]] = {}
    for kind in KINDS:
        results[kind] = {}
        for severity in eval_cfg["severities"]:
            # Seeded per (kind, severity, image) so that the benchmark is fully reproducible.
            kind_offset = KINDS.index(kind) * 10_000 + int(severity * 100)
            degraded = [
                degrade(
                    img,
                    sample_rng(eval_cfg["test_seed"] + 1 + kind_offset, i),
                    kind,
                    severity,
                )
                for i, img in enumerate(clean_images)
            ]
            restored = [pipeline.restore(img) for img in degraded]

            ap_degraded = average_precision(
                _detect_all(pipeline, degraded, ap_threshold), ground_truth, iou
            )
            ap_restored = average_precision(
                _detect_all(pipeline, restored, ap_threshold), ground_truth, iou
            )

            # Baseline 1: simple non-learned filtering in front of the SAME clean-trained detector.
            classical = [classical_enhance(img) for img in degraded]
            ap_classical = average_precision(
                _detect_all(pipeline, classical, ap_threshold), ground_truth, iou
            )
            # Baseline 2: a detector retrained with degradation augmentation, no restoration.
            ap_augmented = (
                average_precision(
                    _detect_all(baseline_pipeline, degraded, ap_threshold),
                    ground_truth,
                    iou,
                )
                if baseline_pipeline is not None
                else None
            )

            results[kind][str(severity)] = {
                "ap_degraded": ap_degraded,
                "ap_classical": ap_classical,
                "ap_restored": ap_restored,
                "ap_augmented_detector": ap_augmented,
                "ap_gain": ap_restored - ap_degraded,
                "psnr_degraded": float(
                    np.mean(
                        [
                            psnr(c, d)
                            for c, d in zip(clean_images, degraded, strict=True)
                        ]
                    )
                ),
                "psnr_restored": float(
                    np.mean(
                        [
                            psnr(c, r)
                            for c, r in zip(clean_images, restored, strict=True)
                        ]
                    )
                ),
                "ssim_degraded": float(
                    np.mean(
                        [
                            ssim(c, d)
                            for c, d in zip(clean_images, degraded, strict=True)
                        ]
                    )
                ),
                "ssim_restored": float(
                    np.mean(
                        [
                            ssim(c, r)
                            for c, r in zip(clean_images, restored, strict=True)
                        ]
                    )
                ),
            }
            entry = results[kind][str(severity)]
            entry["psnr_gain_db"] = entry["psnr_restored"] - entry["psnr_degraded"]
            augmented_text = (
                f"{ap_augmented:.3f}" if ap_augmented is not None else "n/a"
            )
            print(
                f"{kind:12s} sev {severity:.1f}  AP degraded {ap_degraded:.3f} | "
                f"classical {ap_classical:.3f} | restored {ap_restored:.3f} | "
                f"augmented-detector {augmented_text}",
                flush=True,
            )

    # Summaries: mean over the four degradation kinds, for every severity.
    summary = {}
    for severity in eval_cfg["severities"]:
        rows = [results[k][str(severity)] for k in KINDS]
        summary[str(severity)] = {
            "mean_ap_degraded": float(np.mean([r["ap_degraded"] for r in rows])),
            "mean_ap_classical": float(np.mean([r["ap_classical"] for r in rows])),
            "mean_ap_restored": float(np.mean([r["ap_restored"] for r in rows])),
            "mean_ap_augmented_detector": (
                float(np.mean([r["ap_augmented_detector"] for r in rows]))
                if baseline_pipeline is not None
                else None
            ),
            "mean_ap_gain": float(np.mean([r["ap_gain"] for r in rows])),
            "mean_psnr_gain_db": float(np.mean([r["psnr_gain_db"] for r in rows])),
        }

    return {
        "meta": {
            "n_images": eval_cfg["n_images"],
            "image_size": config["data"]["image_size"],
            "test_seed": eval_cfg["test_seed"],
            "iou_threshold": iou,
            "severities": eval_cfg["severities"],
            "train_severity_range": [0.3, 1.0],
            "data_source": config["data"]["source"],
        },
        "clean_ap": clean_ap,
        "clean_ap_augmented_detector": (
            average_precision(
                _detect_all(baseline_pipeline, clean_images, ap_threshold),
                ground_truth,
                iou,
            )
            if baseline_pipeline is not None
            else None
        ),
        "results": results,
        "summary": summary,
        "latency_ms_per_image": _measure_latency(pipeline, clean_images),
    }


# --------------------------------------------------------------------------- reports


def write_markdown_table(report: dict[str, Any], path: Path) -> None:
    """Human-readable results table (pasted into the README)."""
    clean_augmented = report.get("clean_ap_augmented_detector")
    lines = [
        f"Clean images (no degradation): AP@0.5 = **{report['clean_ap']:.3f}** "
        f"(clean-trained detector)"
        + (
            f", **{clean_augmented:.3f}** (augmentation-trained detector)"
            if clean_augmented
            else ""
        )
        + "\n",
        "| Degradation | Severity | Detector alone | + classical filter | + restoration (ours) "
        "| Retrained detector (augmentation) | PSNR before → after |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, by_severity in report["results"].items():
        for severity, r in by_severity.items():
            marker = (
                " (unseen)"
                if float(severity) > report["meta"]["train_severity_range"][1]
                else ""
            )
            augmented = r.get("ap_augmented_detector")
            augmented_text = f"{augmented:.3f}" if augmented is not None else "n/a"
            lines.append(
                f"| {kind} | {severity}{marker} | {r['ap_degraded']:.3f} | {r['ap_classical']:.3f} "
                f"| {r['ap_restored']:.3f} | {augmented_text} "
                f"| {r['psnr_degraded']:.1f} → {r['psnr_restored']:.1f} dB |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(
    config: dict[str, Any], model_dir: str | Path, report: dict[str, Any], out_dir: Path
) -> None:
    """Qualitative grid (clean | degraded | restored) and an AP bar chart."""
    pipeline = OnnxPipeline(model_dir)
    eval_cfg = config["evaluation"]
    images, boxes = _make_test_set(config)

    # Pick a test image that contains several ships so the figure is informative.
    index = int(np.argmax([len(b) for b in boxes[:50]]))
    clean, gt = images[index], boxes[index]

    def to_bgr(img: np.ndarray, detections: np.ndarray | None) -> np.ndarray:
        canvas = cv2.cvtColor(
            (np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        canvas = cv2.resize(canvas, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        for b in gt:  # ground truth in green
            cv2.rectangle(
                canvas,
                (int(b[0] * 2), int(b[1] * 2)),
                (int(b[2] * 2), int(b[3] * 2)),
                (0, 200, 0),
                1,
            )
        if detections is not None:  # predictions in red
            for d in detections:
                cv2.rectangle(
                    canvas,
                    (int(d[0] * 2), int(d[1] * 2)),
                    (int(d[2] * 2), int(d[3] * 2)),
                    (0, 0, 255),
                    1,
                )
        return canvas

    rows = []
    for kind in KINDS:
        degraded = degrade(
            clean, sample_rng(eval_cfg["test_seed"] + 99, 0), kind, GATE_SEVERITY
        )
        restored = pipeline.restore(degraded)
        panels = [
            to_bgr(clean, pipeline.detect(clean, restore=False)),
            to_bgr(degraded, pipeline.detect(degraded, restore=False)),
            to_bgr(restored, pipeline.detect(restored, restore=False)),
        ]
        for panel, label in zip(
            panels, (f"{kind}: clean", "degraded", "restored"), strict=False
        ):
            cv2.putText(
                panel, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
            )
        rows.append(np.hstack(panels))
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "examples.png"), np.vstack(rows))

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend, works in CI and Docker
        import matplotlib.pyplot as plt
    except ImportError:
        return
    severity = str(GATE_SEVERITY)
    x = np.arange(len(KINDS))
    # One group of bars per degradation kind; one bar per approach (augmented one is optional).
    approaches = [
        ("ap_degraded", "detector alone", "#c0504d"),
        ("ap_classical", "+ classical filter", "#e0a030"),
        ("ap_augmented_detector", "retrained detector", "#8064a2"),
        ("ap_restored", "+ restoration (ours)", "#4f81bd"),
    ]
    approaches = [
        a
        for a in approaches
        if report["results"][KINDS[0]][severity].get(a[0]) is not None
    ]
    width = 0.8 / len(approaches)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (key, label, color) in enumerate(approaches):
        values = [report["results"][k][severity][key] for k in KINDS]
        ax.bar(
            x + (i - (len(approaches) - 1) / 2) * width,
            values,
            width,
            label=label,
            color=color,
        )
    ax.axhline(
        report["clean_ap"], color="gray", linestyle="--", label="detector on clean"
    )
    ax.set_xticks(x, KINDS)
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Detection quality at severity {severity} (synthetic test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ap_by_degradation.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- gate


def check_gate(report: dict[str, Any], gate: dict[str, float]) -> list[str]:
    """Return the list of violated floors (empty list = gate passed)."""
    summary = report["summary"][str(GATE_SEVERITY)]
    failures = []
    if summary["mean_ap_gain"] < gate["min_mean_ap_gain"]:
        failures.append(
            f"mean AP gain {summary['mean_ap_gain']:.3f} < required {gate['min_mean_ap_gain']}"
        )
    if summary["mean_psnr_gain_db"] < gate["min_mean_psnr_gain_db"]:
        failures.append(
            f"mean PSNR gain {summary['mean_psnr_gain_db']:.2f} dB < required "
            f"{gate['min_mean_psnr_gain_db']} dB"
        )
    if report["latency_ms_per_image"] > gate["max_latency_ms_per_image"]:
        failures.append(
            f"latency {report['latency_ms_per_image']:.0f} ms > allowed "
            f"{gate['max_latency_ms_per_image']} ms"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-dir", default=None, help="defaults to export.onnx_dir")
    parser.add_argument(
        "--gate", action="store_true", help="exit with code 1 if the gate fails"
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_dir = args.model_dir or config["export"]["onnx_dir"]
    report_dir = Path(config["evaluation"]["report_dir"])

    report = run_benchmark(config, model_dir)
    write_json(report_dir / "metrics.json", report)
    write_markdown_table(report, report_dir / "results.md")
    if not args.no_figures:
        write_figures(config, model_dir, report, report_dir)
    print(
        f"Latency: {report['latency_ms_per_image']:.1f} ms / image (restore + detect, CPU)"
    )

    if args.gate:
        failures = check_gate(report, config["gate"])
        if failures:
            print("QUALITY GATE FAILED:\n  - " + "\n  - ".join(failures))
            sys.exit(1)
        print("Quality gate passed.")


if __name__ == "__main__":
    main()
