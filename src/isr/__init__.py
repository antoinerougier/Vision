"""isr: AI-assisted restoration of degraded ISR imagery.

The package is organised so that the *serving* path (ONNX Runtime + numpy + OpenCV)
never imports PyTorch. Only the training / export modules do.

    degradations.py  physics-inspired sensor / weather degradation simulator
    synthetic.py     procedural maritime scene generator with free ground truth
    postprocess.py   numpy decoding of the detector output (shared by train + serve)
    metrics.py       AP@IoU, PSNR, SSIM (pure numpy / OpenCV)
    inference.py     ONNX Runtime pipeline: restore -> detect
    api.py           FastAPI service wrapping the pipeline

    models.py        PyTorch networks (restoration U-Net, CenterNet-style detector)   [train]
    data.py          PyTorch datasets (synthetic + YOLO-format folders)               [train]
    train_*.py       training entry points                                            [train]
    export.py        PyTorch -> ONNX export with numerical parity check               [train]
    evaluate.py      benchmark + CI quality gate (runs on the ONNX artefacts)
"""

__version__ = "0.1.0"
