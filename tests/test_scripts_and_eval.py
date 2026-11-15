"""VOC -> YOLO converter, config loading and the CI quality gate logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from isr.common import load_config
from isr.evaluate import check_gate

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Import a file from scripts/ (it is not part of the installed package)."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VOC_XML = """<annotation>
  <size><width>200</width><height>100</height><depth>1</depth></size>
  <object><name>ship</name>
    <bndbox><xmin>80</xmin><ymin>40</ymin><xmax>120</xmax><ymax>60</ymax></bndbox>
  </object>
</annotation>"""


def test_voc_to_yolo_conversion(tmp_path: Path) -> None:
    voc = _load_script("voc_to_yolo")
    images, annotations, out = tmp_path / "img", tmp_path / "ann", tmp_path / "out"
    images.mkdir()
    annotations.mkdir()
    cv2.imwrite(str(images / "a.png"), np.zeros((100, 200), dtype=np.uint8))
    (annotations / "a.xml").write_text(VOC_XML)

    voc.convert(images, annotations, out, val_fraction=0.0, seed=0)

    label = (out / "labels" / "train" / "a.txt").read_text().split()
    assert label[0] == "0"
    np.testing.assert_allclose([float(v) for v in label[1:]], [0.5, 0.5, 0.2, 0.2], atol=1e-5)
    assert (out / "images" / "train" / "a.png").exists()


def test_default_config_is_valid_and_train_test_seeds_do_not_overlap() -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    for section in ("data", "detector", "restoration", "export", "evaluation", "gate"):
        assert section in config
    train_seeds = {
        config["detector"]["train_seed"],
        config["restoration"]["train_seed"],
    }
    assert config["evaluation"]["test_seed"] not in train_seeds
    # The unseen-severity test only makes sense if it is really outside the training range.
    assert max(config["evaluation"]["severities"]) > 1.0


GOOD_REPORT = {
    "summary": {"1.0": {"mean_ap_gain": 0.20, "mean_psnr_gain_db": 5.0}},
    "latency_ms_per_image": 20.0,
}
GATE = {
    "min_mean_ap_gain": 0.05,
    "min_mean_psnr_gain_db": 1.0,
    "max_latency_ms_per_image": 250,
}


def test_gate_passes_on_a_good_report() -> None:
    assert check_gate(GOOD_REPORT, GATE) == []


@pytest.mark.parametrize(
    ("override", "expected_fragment"),
    [
        (
            {"summary": {"1.0": {"mean_ap_gain": 0.0, "mean_psnr_gain_db": 5.0}}},
            "AP gain",
        ),
        ({"summary": {"1.0": {"mean_ap_gain": 0.2, "mean_psnr_gain_db": 0.1}}}, "PSNR"),
        ({"latency_ms_per_image": 900.0}, "latency"),
    ],
)
def test_gate_fails_when_a_floor_is_violated(override: dict, expected_fragment: str) -> None:
    failures = check_gate({**GOOD_REPORT, **override}, GATE)
    assert len(failures) == 1 and expected_fragment in failures[0]
