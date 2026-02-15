"""
Kalman Filter for bounding-box tracking in image space.

State vector:  [cx, cy, aspect_ratio, height, vx, vy, va, vh]
Measurement:   [cx, cy, aspect_ratio, height]

This is the standard formulation used in SORT / DeepSORT / ByteTrack.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


class KalmanFilter:
    """A simple Kalman filter for tracking bounding boxes in image space."""

    # Motion and observation uncertainty weights
    _std_weight_position = 1.0 / 20
    _std_weight_velocity = 1.0 / 160

    def __init__(self):
        ndim, dt = 4, 1.0

        # State transition matrix (constant velocity model)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation matrix (we only observe cx, cy, a, h)
        self._update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement: np.ndarray):
        """Create a track from an unassociated measurement.

        Parameters
        ----------
        measurement : ndarray (4,)
            [cx, cy, aspect_ratio, height]

        Returns
        -------
        mean : ndarray (8,)
        covariance : ndarray (8, 8)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run the Kalman filter prediction step.

        Returns
        -------
        mean : ndarray (8,)
        covariance : ndarray (8, 8)
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state to measurement space.

        Returns
        -------
        mean : ndarray (4,)
        covariance : ndarray (4, 4)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run the Kalman filter update step.

        Returns
        -------
        mean : ndarray (8,)
        covariance : ndarray (8, 8)
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray,
        measurements: np.ndarray, only_position: bool = False
    ) -> np.ndarray:
        """Compute Mahalanobis distance between state and measurements.

        Parameters
        ----------
        measurements : ndarray (N, 4)

        Returns
        -------
        distances : ndarray (N,)
        """
        proj_mean, proj_cov = self.project(mean, covariance)
        if only_position:
            proj_mean, proj_cov = proj_mean[:2], proj_cov[:2, :2]
            measurements = measurements[:, :2]

        chol = np.linalg.cholesky(proj_cov)
        d = measurements - proj_mean
        z = scipy.linalg.solve_triangular(
            chol, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        return np.sum(z * z, axis=0)
