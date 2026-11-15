"""Physics-inspired degradation simulator.

Every function takes a clean grayscale image (float32, shape (H, W), values in [0, 1])
and returns a degraded image of the same shape and range. The goal is NOT to be a
perfect sensor model: it is to produce *qualitatively faithful* failure modes that make
a detector trained on clean data collapse, so that a restoration layer has something
measurable to fix.

Four degradation families are implemented:

    sar_speckle  multiplicative, gamma-distributed speckle (fewer "looks" = noisier)
    low_light    photon shot noise + read noise + strong gain loss + 8-bit quantisation
    fog          Koschmieder atmospheric scattering model  I = J*t + A*(1 - t)
    thermal      optical blur + low resolution + fixed-pattern noise + low contrast

`severity` is a single knob in [0, ~1.3]: 0 = almost clean, 1 = hard, >1 = harder than
anything seen during training (used to test generalisation).
"""

from __future__ import annotations

import cv2
import numpy as np

KINDS: tuple[str, ...] = ("sar_speckle", "low_light", "fog", "thermal")

# Range of severities sampled during training of the restoration model.
TRAIN_SEVERITY_RANGE: tuple[float, float] = (0.3, 1.0)


def _clip01(img: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] and make sure the dtype is float32."""
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def sar_speckle(img: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    """Multiplicative speckle noise.

    In SAR intensity images, speckle follows a Gamma distribution with unit mean and
    shape L (the number of "looks"). One look = very strong speckle (the std equals the
    mean); more looks average the noise away. We map severity -> looks:
    severity 0 -> 8 looks, severity 1 -> 1 look.
    """
    looks = max(8.0 ** (1.0 - severity), 0.7)
    speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=img.shape)
    return _clip01(img * speckle)


def low_light(img: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    """Dark scene captured with a noisy sensor.

    1. The signal is attenuated (gain loss).
    2. Photon arrival is Poisson distributed (shot noise): fewer photons = noisier.
    3. Gaussian read noise is added by the electronics.
    4. The result is quantised to 8 bits, which crushes the dark levels.
    """
    gain = max(1.0 - 0.85 * severity, 0.08)  # severity 1 -> only 15% of the light
    full_well = 60.0  # photons collected by a pixel at full brightness
    photons = rng.poisson(np.clip(img, 0, 1) * gain * full_well).astype(np.float32) / full_well
    read_noise = rng.normal(0.0, 0.008 + 0.012 * severity, size=img.shape)
    out = photons + read_noise
    return _clip01(np.round(_clip01(out) * 255.0) / 255.0)


def fog(img: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    """Koschmieder atmospheric scattering: I = J * t + A * (1 - t).

    J is the clean radiance, A the airlight (bright haze), and t = exp(-beta * depth)
    the transmission. The depth map is a smooth random field so that the haze is
    patchy rather than a flat veil.
    """
    h, w = img.shape
    # Low-resolution random field, upsampled -> smooth depth map in [0.4, 1.0]
    coarse = rng.random((max(h // 32, 2), max(w // 32, 2))).astype(np.float32)
    depth = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    depth = 0.4 + 0.6 * (depth - depth.min()) / (np.ptp(depth) + 1e-6)

    beta = 0.3 + 2.2 * severity  # scattering coefficient
    transmission = np.exp(-beta * depth)
    airlight = 0.85
    out = img * transmission + airlight * (1.0 - transmission)
    out = out + rng.normal(0.0, 0.01, size=img.shape)  # mild sensor noise
    return _clip01(out)


def thermal(img: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    """Low-cost uncooled thermal camera.

    Blur (optics) + low native resolution + column fixed-pattern noise (detector
    non-uniformity) + temporal noise + reduced contrast.
    """
    h, w = img.shape
    sigma = 0.8 + 1.6 * severity
    out = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

    # Low native resolution: shrink then enlarge back.
    factor = 1.0 + 1.5 * severity
    small = cv2.resize(
        out,
        (max(int(w / factor), 8), max(int(h / factor), 8)),
        interpolation=cv2.INTER_AREA,
    )
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Fixed-pattern noise: one random offset per detector column.
    column_offsets = rng.normal(0.0, 0.02 + 0.03 * severity, size=(1, w)).astype(np.float32)
    out = out + column_offsets

    # Reduced contrast around mid-gray + temporal noise.
    contrast = 0.75 - 0.3 * min(severity, 1.2)
    out = 0.5 + contrast * (out - 0.5)
    out = out + rng.normal(0.0, 0.02 + 0.04 * severity, size=img.shape)
    return _clip01(out)


_DISPATCH = {
    "sar_speckle": sar_speckle,
    "low_light": low_light,
    "fog": fog,
    "thermal": thermal,
}


def degrade(
    img: np.ndarray,
    rng: np.random.Generator,
    kind: str | None = None,
    severity: float | None = None,
) -> np.ndarray:
    """Apply one degradation. `kind` / `severity` are drawn at random when omitted.

    Drawing them at random is what we use to train a *blind* restoration model: the
    network is never told which degradation it is facing.
    """
    if kind is None:
        kind = KINDS[int(rng.integers(len(KINDS)))]
    if kind not in _DISPATCH:
        raise ValueError(f"Unknown degradation '{kind}'. Valid kinds: {KINDS}")
    if severity is None:
        severity = float(rng.uniform(*TRAIN_SEVERITY_RANGE))
    return _DISPATCH[kind](img.astype(np.float32), rng, severity)
