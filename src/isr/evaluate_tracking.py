"""Benchmark of the tracking + forecasting layer on simulated sequences.

For every simulated scenario (sharp photo at t=0, then a degraded frame every 2 minutes)
we run several ways of obtaining detections, feed them to the SAME tracker, and score:

    detection   did the detector see the ships? (recall, false alarms)
    tracking    are the ships followed without gaps or identity swaps? (completeness,
                ID switches, false tracks)
    forecast    how far is the predicted position from the true one, 2 to 10 minutes
                ahead? (average / final displacement error, in pixels) and are the
                uncertainty ellipses honest? (coverage of the 95 % ellipse)

Detection sources compared (all feed the same tracker, so only the input changes):

    oracle      true ship centres + small noise: the best the tracker can ever do
    restored    restoration network + detector       (the product)
    classical   median + contrast stretch + detector (no-ML baseline)
    raw         detector alone on degraded frames    (do nothing)
    retrained   detector retrained with augmentation (the "just retrain" alternative)

Forecast methods compared on exactly the same (ship, frame) samples:

    kalman      the tracker's own forecast (ours)
    linear5     straight line fitted (least squares) on the last five detections
    linear2     straight line through the last two detections
    stationary  the ship stays where it was last seen

Usage:
    python -m isr.evaluate_tracking --config configs/default.yaml [--gate]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .common import load_config, sample_rng, write_json
from .degradations import KINDS
from .inference import OnnxPipeline
from .scenario import Scenario, ScenarioConfig, build_scenario
from .sequence import TrackSnapshot, detect_sequence, horizons_for, run_tracking
from .tracking import CHI2_95_2DOF, TrackerParams, inside_ellipse

METHODS = ("kalman", "linear5", "linear2", "stationary")
ORACLE_NOISE_PX = 1.5  # std of the simulated "perfect" detector


# ------------------------------------------------------------------------- helpers


def _greedy_match(a: np.ndarray, b: np.ndarray, max_dist: float) -> dict[int, int]:
    """Greedy nearest-neighbour matching of two point sets. Returns {index in a: index in b}."""
    if len(a) == 0 or len(b) == 0:
        return {}
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    matches: dict[int, int] = {}
    used_b: set[int] = set()
    for flat in np.argsort(dist, axis=None):
        i, j = np.unravel_index(flat, dist.shape)
        if dist[i, j] > max_dist:
            break
        if int(i) in matches or int(j) in used_b:
            continue
        matches[int(i)] = int(j)
        used_b.add(int(j))
    return matches


def _oracle_detections(scenario: Scenario) -> list[np.ndarray]:
    """'Perfect' detector: true centres plus a little noise, one small box per ship."""
    rng = sample_rng(scenario.seed + 31337, 0)
    detections = []
    for t in range(scenario.n_frames):
        centers = scenario.positions[:, t] + rng.normal(
            0, ORACLE_NOISE_PX, (scenario.n_ships, 2)
        )
        boxes = np.concatenate(
            [centers - 5.0, centers + 5.0, np.ones((scenario.n_ships, 1))], axis=1
        ).astype(np.float32)
        detections.append(boxes)
    return detections


def _predict_baselines(
    snapshot: TrackSnapshot, t_now: float, horizons: list[float]
) -> dict[str, np.ndarray]:
    """Forecasts (K, 2) of the three methods for one tracked ship."""
    last = snapshot.last_measurements
    t1, x1, y1 = last[-1]
    stationary = np.tile([x1, y1], (len(horizons), 1))
    if len(last) >= 2:
        t0, x0, y0 = last[-2]
        velocity = np.array([x1 - x0, y1 - y0]) / max(t1 - t0, 1e-6)
    else:
        velocity = np.zeros(2)
    taus = np.array(horizons)[:, None]
    linear2 = np.array([x1, y1]) + velocity * (t_now + taus - t1)

    # Least-squares straight line on the last (up to) five detections: x(t) = a + b t.
    if len(last) >= 2:
        ts = np.array([m[0] for m in last])
        xs = np.array([[m[1], m[2]] for m in last])
        slope = np.polyfit(ts, xs, 1)  # (2, 2): row 0 = slopes, row 1 = intercepts
        linear5 = slope[1] + slope[0] * (t_now + taus)
    else:
        linear5 = stationary
    return {
        "kalman": snapshot.forecast_means,
        "linear5": linear5,
        "linear2": linear2,
        "stationary": stationary,
    }


def _score_scenario(
    scenario: Scenario,
    detections: list[np.ndarray],
    snapshots: list[list[TrackSnapshot]],
    horizons: list[float],
    match_distance: float,
) -> dict[str, Any]:
    """Raw counts and error samples for one (scenario, detection source)."""
    T, S = scenario.n_frames, scenario.n_ships
    n_h = len(horizons)
    out: dict[str, Any] = {
        "kind": scenario.kind,
        "det_tp": 0, "det_gt": 0, "det_fp": 0,
        "covered": 0, "gt_total": 0, "false_tracks": 0, "frames": 0,
        "id_switches": 0, "ships": S,
        "avail_covered": 0, "avail_total": 0,
        "errors": {m: [[] for _ in range(n_h)] for m in METHODS},
        "inside": [[] for _ in range(n_h)],
    }  # fmt: skip
    last_track_of_ship: dict[int, int] = {}

    for t in range(
        1, T
    ):  # frame 0 is the sharp reference: not part of the degraded benchmark
        gt = scenario.positions[:, t].astype(np.float64)
        out["frames"] += 1

        # --- detection level ---
        if len(detections[t]):
            centers = np.stack(
                [
                    (detections[t][:, 0] + detections[t][:, 2]) / 2,
                    (detections[t][:, 1] + detections[t][:, 3]) / 2,
                ],
                axis=1,
            )
        else:
            centers = np.zeros((0, 2))
        det_matches = _greedy_match(gt, centers, match_distance)
        out["det_tp"] += len(det_matches)
        out["det_gt"] += S
        out["det_fp"] += len(centers) - len(det_matches)

        # --- track level ---
        tracks = snapshots[t]
        track_xy = (
            np.array([[s.x, s.y] for s in tracks]) if tracks else np.zeros((0, 2))
        )
        matches = _greedy_match(gt, track_xy, match_distance)
        out["covered"] += len(matches)
        out["gt_total"] += S
        out["false_tracks"] += len(tracks) - len(matches)

        for ship, track_index in matches.items():
            snapshot = tracks[track_index]
            previous = last_track_of_ship.get(ship)
            if previous is not None and previous != snapshot.id:
                out["id_switches"] += 1
            last_track_of_ship[ship] = snapshot.id

        # --- forecast level: only frames whose full horizon lies inside the sequence ---
        if t + n_h < T:
            out["avail_total"] += S
            out["avail_covered"] += len(matches)
            for ship, track_index in matches.items():
                snapshot = tracks[track_index]
                predictions = _predict_baselines(snapshot, scenario.times[t], horizons)
                for k in range(n_h):
                    truth = scenario.positions[ship, t + 1 + k].astype(np.float64)
                    for method in METHODS:
                        out["errors"][method][k].append(
                            float(np.linalg.norm(predictions[method][k] - truth))
                        )
                    out["inside"][k].append(
                        inside_ellipse(
                            truth,
                            snapshot.forecast_means[k],
                            snapshot.forecast_covs[k],
                            CHI2_95_2DOF,
                        )
                    )
    return out


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine the raw counts of several scenarios into the final metrics."""

    def ratio(num: float, den: float) -> float | None:
        return float(num / den) if den else None

    if not results:  # e.g. a degradation kind that has no scenario in a small benchmark
        no_method = {m: None for m in METHODS}
        return {
            "detection": {"recall": None, "false_alarms_per_frame": None},
            "tracking": {
                "completeness": None,
                "false_tracks_per_frame": None,
                "id_switches_per_ship": None,
            },
            "forecast": {
                "availability": None,
                "n_samples": 0,
                "ade": dict(no_method),
                "fde": dict(no_method),
                "error_by_horizon": dict(no_method),
                "ellipse_coverage_by_horizon": [],
            },
        }

    total = {
        key: sum(r[key] for r in results)
        for key in (
            "det_tp", "det_gt", "det_fp", "covered", "gt_total", "false_tracks",
            "frames", "id_switches", "ships", "avail_covered", "avail_total",
        )
    }  # fmt: skip
    n_h = len(results[0]["inside"])
    by_horizon = {
        m: [
            float(np.mean(sum((r["errors"][m][k] for r in results), []) or [np.nan]))
            for k in range(n_h)
        ]
        for m in METHODS
    }
    all_inside = [sum((r["inside"][k] for r in results), []) for k in range(n_h)]
    has_samples = not np.isnan(by_horizon["kalman"][0])
    return {
        "detection": {
            "recall": ratio(total["det_tp"], total["det_gt"]),
            "false_alarms_per_frame": ratio(total["det_fp"], total["frames"]),
        },
        "tracking": {
            "completeness": ratio(total["covered"], total["gt_total"]),
            "false_tracks_per_frame": ratio(total["false_tracks"], total["frames"]),
            "id_switches_per_ship": ratio(total["id_switches"], total["ships"]),
        },
        "forecast": {
            "availability": ratio(total["avail_covered"], total["avail_total"]),
            "n_samples": len(all_inside[0]),
            # Average displacement error over all horizons, and error at the last one.
            "ade": {
                m: float(np.mean(by_horizon[m])) if has_samples else None
                for m in METHODS
            },
            "fde": {m: by_horizon[m][-1] if has_samples else None for m in METHODS},
            "error_by_horizon": {
                m: by_horizon[m] if has_samples else None for m in METHODS
            },
            "ellipse_coverage_by_horizon": [
                float(np.mean(x)) if x else None for x in all_inside
            ],
        },
    }


# ------------------------------------------------------------------------- benchmark


def run_tracking_benchmark(
    config: dict[str, Any], model_dir: str | Path
) -> dict[str, Any]:
    """Run every detection source on the same simulated scenarios and score them."""
    tcfg = config["tracking"]
    scenario_cfg = ScenarioConfig.from_config(config)
    params = TrackerParams.from_config(config)
    ev = tcfg["evaluation"]
    threshold = float(tcfg["detection"]["score_threshold"])
    horizon = float(tcfg["forecast_horizon_minutes"])
    horizons = horizons_for(scenario_cfg.dt_minutes, horizon)
    match_distance = float(ev["match_distance_px"])

    pipeline = OnnxPipeline(model_dir)
    retrained_path = Path(model_dir) / "detector_augmented.onnx"
    retrained = (
        OnnxPipeline(model_dir, detector_file="detector_augmented.onnx")
        if retrained_path.exists()
        else None
    )

    per_source: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("oracle", "restored", "classical", "raw", "retrained")
    }
    if retrained is None:
        del per_source["retrained"]

    for i in range(int(ev["n_scenarios"])):
        kind = KINDS[i % len(KINDS)]
        scenario = build_scenario(
            int(ev["seed"]) + i, scenario_cfg, kind, float(ev["severity"])
        )
        if scenario.n_ships == 0:
            continue
        detections_by_source = {
            "oracle": _oracle_detections(scenario),
            "restored": detect_sequence(
                pipeline, scenario.frames, "restored", threshold
            )[0],
            "classical": detect_sequence(
                pipeline, scenario.frames, "classical", threshold
            )[0],
            "raw": detect_sequence(pipeline, scenario.frames, "raw", threshold)[0],
        }
        if retrained is not None:
            detections_by_source["retrained"] = detect_sequence(
                retrained, scenario.frames, "raw", threshold
            )[0]
        for name, detections in detections_by_source.items():
            snapshots = run_tracking(detections, scenario.times, params, horizon)
            per_source[name].append(
                _score_scenario(
                    scenario, detections, snapshots, horizons, match_distance
                )
            )

    report: dict[str, Any] = {
        "meta": {
            "n_scenarios": len(per_source["oracle"]),
            "severity": float(ev["severity"]),
            "dt_minutes": scenario_cfg.dt_minutes,
            "horizon_minutes": horizon,
            "horizons": horizons,
            "n_frames": scenario_cfg.n_frames,
            "match_distance_px": match_distance,
            "seed": int(ev["seed"]),
        },
        "sources": {},
    }
    for name, results in per_source.items():
        entry = _aggregate(results)
        entry["by_kind"] = {
            kind: _aggregate([r for r in results if r["kind"] == kind])
            for kind in KINDS
        }
        report["sources"][name] = entry
    return report


# ---------------------------------------------------------------------------- reports

_LABELS = {
    "oracle": "oracle detections (upper bound)",
    "restored": "restoration + detector (ours)",
    "classical": "classical filter + detector",
    "raw": "detector alone",
    "retrained": "retrained detector (augmentation)",
}


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown_table(report: dict[str, Any], path: Path) -> None:
    horizon = report["meta"]["horizon_minutes"]
    lines = [
        f"{report['meta']['n_scenarios']} simulated scenarios, severity "
        f"{report['meta']['severity']}, error in pixels at +{horizon:g} min.\n",
        "| Detections from | Ship recall | False alarms / frame | Track completeness "
        "| ID switches / ship | Forecast availability | Kalman error | Linear (5 pts) error "
        "| Linear (2 pts) error | Stationary error | 95 % ellipse coverage |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in report["sources"].items():
        f = entry["forecast"]
        lines.append(
            f"| {_LABELS[name]} | {_fmt(entry['detection']['recall'])} "
            f"| {_fmt(entry['detection']['false_alarms_per_frame'])} "
            f"| {_fmt(entry['tracking']['completeness'])} "
            f"| {_fmt(entry['tracking']['id_switches_per_ship'])} "
            f"| {_fmt(f['availability'])} | {_fmt(f['fde']['kalman'], 1)} "
            f"| {_fmt(f['fde']['linear5'], 1)} | {_fmt(f['fde']['linear2'], 1)} "
            f"| {_fmt(f['fde']['stationary'], 1)} "
            f"| {_fmt(f['ellipse_coverage_by_horizon'][-1] if f['ellipse_coverage_by_horizon'] else None)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(report: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    horizons = report["meta"]["horizons"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # Left: forecast error as a function of the horizon, for the forecasting methods,
    # using the detections of the product (restoration + detector).
    source = report["sources"]["restored"]["forecast"]["error_by_horizon"]
    if source:
        for method, color in (
            ("kalman", "#4f81bd"),
            ("linear5", "#9bbb59"),
            ("linear2", "#e0a030"),
            ("stationary", "#c0504d"),
        ):
            ax1.plot(horizons, source[method], marker="o", label=method, color=color)
    ax1.set_xlabel("forecast horizon (minutes)")
    ax1.set_ylabel("position error (pixels)")
    ax1.set_title("Forecast error (restoration + detector)")
    ax1.legend(fontsize=8)

    # Right: how much of the time each detection source keeps every ship tracked.
    names = list(report["sources"])
    values = [report["sources"][n]["tracking"]["completeness"] or 0.0 for n in names]
    ax2.bar(range(len(names)), values, color="#4f81bd")
    ax2.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("track completeness")
    ax2.set_title("Ships followed (share of ship-frames)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------ gate


def check_gate(report: dict[str, Any], gate: dict[str, float]) -> list[str]:
    """Violated quality floors of the tracking layer (empty list = passed)."""
    failures = []
    ours = report["sources"]["restored"]
    completeness = ours["tracking"]["completeness"] or 0.0
    if completeness < gate["min_track_completeness"]:
        failures.append(
            f"track completeness {completeness:.2f} < required {gate['min_track_completeness']}"
        )
    f = ours["forecast"]
    if f["fde"]["kalman"] is None:
        failures.append("no forecast could be evaluated")
        return failures
    ratio = f["fde"]["kalman"] / f["fde"]["stationary"]
    if ratio > gate["max_kalman_to_stationary_error_ratio"]:
        failures.append(
            f"Kalman/stationary error ratio {ratio:.2f} > allowed "
            f"{gate['max_kalman_to_stationary_error_ratio']}"
        )
    coverage = f["ellipse_coverage_by_horizon"][-1]
    if coverage is None or coverage < gate["min_ellipse_coverage"]:
        failures.append(
            f"95% ellipse coverage {coverage} < required {gate['min_ellipse_coverage']}"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-dir", default=None, help="defaults to export.onnx_dir")
    parser.add_argument(
        "--gate", action="store_true", help="exit with code 1 if the gate fails"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_dir = args.model_dir or config["export"]["onnx_dir"]
    report_dir = Path(config["evaluation"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    report = run_tracking_benchmark(config, model_dir)
    write_json(report_dir / "tracking_metrics.json", report)
    write_markdown_table(report, report_dir / "tracking_results.md")
    write_figure(report, report_dir / "tracking_forecast.png")
    print((report_dir / "tracking_results.md").read_text(encoding="utf-8"))

    if args.gate:
        failures = check_gate(report, config["tracking"]["gate"])
        if failures:
            print("TRACKING QUALITY GATE FAILED:\n  - " + "\n  - ".join(failures))
            sys.exit(1)
        print("Tracking quality gate passed.")


if __name__ == "__main__":
    main()
