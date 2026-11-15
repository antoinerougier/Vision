"""Multi-ship tracking and short-term trajectory forecasting (numpy only, no PyTorch).

The detector answers "where are the ships in THIS frame?". Tracking answers "which ship is
which across frames?" and, once a ship is followed, "where will it be in the next minutes?".

Method (classical and deliberately simple, so that it is easy to audit):

    * Each ship is a **Kalman filter** with the state [x, y, vx, vy] (pixels, pixels/minute)
      and a *constant velocity* motion model. The filter blends the motion model with the
      noisy detections, so a missed detection does not break the track (the filter "coasts")
      and a noisy detection does not make the track jump.
    * **Association** between tracks and detections uses the Mahalanobis distance (a distance
      that accounts for the filter's own uncertainty), a statistical gate to reject
      implausible pairs, and a greedy global assignment (cheapest pairs first).
    * **Track management**: a track is *tentative* until it has been seen `min_hits` times
      (this removes isolated false alarms), then *confirmed*, and it is deleted after
      `max_misses` consecutive frames without detection. Detections on the sharp reference
      photo create confirmed tracks immediately because that image is reliable.
    * **Forecast**: the filter is rolled forward without any new measurement. The covariance
      grows with the horizon, which gives an uncertainty ellipse for every predicted point.

Conventions: positions in pixels, time in minutes, velocities in pixels per minute.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Chi-square quantile for 2 degrees of freedom at 95 %: the uncertainty ellipse drawn
# around a forecast contains the true position about 95 % of the time IF the filter is
# well calibrated (this is checked, not assumed, in the evaluation).
CHI2_95_2DOF = 5.991


@dataclass(frozen=True)
class TrackerParams:
    """Tuning of the tracker (see `tracking.kalman` in `configs/default.yaml`)."""

    # White-noise acceleration intensity q (pixels^2 / minute^3). Larger = the filter
    # trusts its straight-line model less and adapts faster to turns, at the price of
    # noisier velocity estimates.
    process_noise: float = 0.01
    measurement_noise_px: float = 2.0  # standard deviation of a detected centre
    gate_chi2: float = 13.8  # Mahalanobis^2 gate, 2 dof (13.8 = 99.9 %)
    min_hits: int = 2  # detections needed to confirm a track
    max_misses: int = 4  # consecutive missed frames before a confirmed track is deleted
    initial_speed_sigma: float = 2.0  # prior uncertainty on a new track's speed (px/min)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TrackerParams:
        section = config["tracking"]["kalman"]
        return cls(
            process_noise=float(section["process_noise"]),
            measurement_noise_px=float(section["measurement_noise_px"]),
            gate_chi2=float(section["gate_chi2"]),
            min_hits=int(section["min_hits"]),
            max_misses=int(section["max_misses"]),
            initial_speed_sigma=float(section["initial_speed_sigma"]),
        )


def transition_matrices(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """State transition F and process noise Q of the constant-velocity model over `dt`.

    State order is [x, y, vx, vy]. Q is the exact covariance added by a continuous
    white-noise acceleration of intensity q during `dt` (the textbook CWNA model).
    """
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    Q = np.zeros((4, 4))
    Q[0, 0] = Q[1, 1] = q * dt**3 / 3.0
    Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = q * dt**2 / 2.0
    Q[2, 2] = Q[3, 3] = q * dt
    return F, Q


_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # we only measure the position


@dataclass
class Track:
    """One followed ship: a Kalman state plus bookkeeping."""

    id: int
    x: np.ndarray  # (4,) state [x, y, vx, vy]
    P: np.ndarray  # (4, 4) state covariance
    confirmed: bool = False
    hits: int = 1
    misses: int = 0
    # (t, x, y) of the filtered position after every frame (used to draw the trail).
    history: list[tuple[float, float, float]] = field(default_factory=list)
    # (t, x, y) of the raw detections associated to this track (used by the baselines).
    measurements: list[tuple[float, float, float]] = field(default_factory=list)

    def predict(self, dt: float, q: float) -> None:
        """Move the state forward in time without any measurement."""
        F, Q = transition_matrices(dt, q)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation_covariance(self, r: float) -> np.ndarray:
        """Covariance S of the predicted measurement: P projected on (x, y) plus R."""
        return _H @ self.P @ _H.T + np.eye(2) * r**2

    def update(self, z: np.ndarray, r: float) -> None:
        """Correct the state with a detected centre z = (x, y)."""
        R = np.eye(2) * r**2
        S = _H @ self.P @ _H.T + R
        K = self.P @ _H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ (z - _H @ self.x)
        # Joseph form keeps P symmetric positive-definite despite rounding errors.
        I_KH = np.eye(4) - K @ _H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))

    @property
    def heading(self) -> float:
        return float(np.arctan2(self.x[3], self.x[2]))


class MultiShipTracker:
    """Keeps a set of tracks and updates them frame after frame."""

    def __init__(self, params: TrackerParams | None = None) -> None:
        self.params = params or TrackerParams()
        self.tracks: list[Track] = []
        self._time: float | None = None
        self._next_id = 1

    # ------------------------------------------------------------------ association
    def _associate(self, centers: np.ndarray) -> list[tuple[int, int]]:
        """Greedy global assignment under a Mahalanobis gate. Returns (track, detection)."""
        if not self.tracks or len(centers) == 0:
            return []
        r = self.params.measurement_noise_px
        costs = np.full((len(self.tracks), len(centers)), np.inf)
        for i, track in enumerate(self.tracks):
            S_inv = np.linalg.inv(track.innovation_covariance(r))
            residuals = centers - (_H @ track.x)  # (N, 2)
            d2 = np.einsum("nj,jk,nk->n", residuals, S_inv, residuals)
            costs[i] = np.where(d2 <= self.params.gate_chi2, d2, np.inf)

        pairs: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        # Cheapest pairs first: the most certain associations are settled before the
        # ambiguous ones, which is a good approximation of the optimal assignment when
        # ships are well separated.
        for flat in np.argsort(costs, axis=None):
            i, j = np.unravel_index(flat, costs.shape)
            if not np.isfinite(costs[i, j]):
                break
            if i in used_tracks or j in used_dets:
                continue
            pairs.append((int(i), int(j)))
            used_tracks.add(int(i))
            used_dets.add(int(j))
        return pairs

    # ----------------------------------------------------------------------- update
    def update(
        self, t: float, centers: np.ndarray, reference: bool = False
    ) -> list[tuple[int, int]]:
        """Process the detections of the frame taken at time `t` (minutes).

        `centers` is (N, 2). `reference=True` marks the sharp photo: its detections are
        trusted, so they start confirmed tracks without waiting for a second sighting.
        Returns the (track id, detection index) pairs that were associated.
        """
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
        p = self.params

        if self._time is not None and t > self._time:
            for track in self.tracks:
                track.predict(t - self._time, p.process_noise)
        self._time = t

        pairs = self._associate(centers)
        matched_tracks = {i for i, _ in pairs}
        matched_dets = {j for _, j in pairs}

        for i, j in pairs:
            track = self.tracks[i]
            track.update(centers[j], p.measurement_noise_px)
            track.hits += 1
            track.misses = 0
            track.measurements.append((t, float(centers[j, 0]), float(centers[j, 1])))
            if track.hits >= p.min_hits:
                track.confirmed = True

        # Remember which track id got which detection BEFORE the list is modified below
        # (the indices in `pairs` refer to the list as it is now).
        associated = [(self.tracks[i].id, j) for i, j in pairs]

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.misses += 1

        for j in range(len(centers)):
            if j in matched_dets:
                continue
            x0 = np.array([centers[j, 0], centers[j, 1], 0.0, 0.0])
            P0 = np.diag(
                [
                    p.measurement_noise_px**2,
                    p.measurement_noise_px**2,
                    p.initial_speed_sigma**2,
                    p.initial_speed_sigma**2,
                ]
            )
            new = Track(id=self._next_id, x=x0, P=P0, confirmed=reference)
            new.measurements.append((t, float(centers[j, 0]), float(centers[j, 1])))
            self._next_id += 1
            self.tracks.append(new)

        # Track deletion: a tentative track must be confirmed right away on the next frame
        # (otherwise it was probably a false alarm); a confirmed track may coast a while.
        survivors = []
        for track in self.tracks:
            limit = p.max_misses if track.confirmed else 0
            if track.misses <= limit:
                track.history.append((t, float(track.x[0]), float(track.x[1])))
                survivors.append(track)
        self.tracks = survivors
        return associated

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]


def forecast_track(
    track: Track, horizons_minutes: Sequence[float], q: float
) -> tuple[np.ndarray, np.ndarray]:
    """Predict where a track will be, `horizons_minutes` after its latest state.

    Returns (means (K, 2), covariances (K, 2, 2)). The filter is propagated in one jump for
    each horizon with the *exact* CWNA model, so the covariance reflects the growth of
    uncertainty with time: this is what turns a point prediction into a usable forecast.
    """
    means = np.zeros((len(horizons_minutes), 2))
    covs = np.zeros((len(horizons_minutes), 2, 2))
    for k, tau in enumerate(horizons_minutes):
        F, Q = transition_matrices(float(tau), q)
        x = F @ track.x
        P = F @ track.P @ F.T + Q
        means[k] = x[:2]
        covs[k] = P[:2, :2]
    return means, covs


def ellipse_from_covariance(
    cov: np.ndarray, chi2: float = CHI2_95_2DOF
) -> tuple[float, float, float]:
    """Semi-axes (a >= b) and orientation (radians) of the `chi2` confidence ellipse."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    a = float(np.sqrt(chi2 * eigenvalues[1]))
    b = float(np.sqrt(chi2 * eigenvalues[0]))
    angle = float(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    return a, b, angle


def inside_ellipse(point: np.ndarray, mean: np.ndarray, cov: np.ndarray, chi2: float) -> bool:
    """True if `point` lies inside the confidence ellipse (Mahalanobis test)."""
    d = point - mean
    return bool(d @ np.linalg.solve(cov, d) <= chi2)
