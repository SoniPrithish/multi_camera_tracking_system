"""
Kalman Filter implementation for object tracking.
Implements a constant velocity motion model for bounding box tracking.
"""

import numpy as np
from typing import Tuple, Optional


class KalmanFilter:
    """
    Kalman filter for tracking bounding boxes in image space.
    
    State vector: [x, y, a, h, vx, vy, va, vh]
    - (x, y): center position
    - a: aspect ratio (w/h)
    - h: height
    - (vx, vy, va, vh): velocities
    
    Measurement vector: [x, y, a, h]
    """
    
    def __init__(self):
        # State dimension and measurement dimension
        self.ndim = 4
        self.dt = 1.0
        
        # State transition matrix (constant velocity model)
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        
        # Observation matrix (we observe position, not velocity)
        self._observation_mat = np.eye(self.ndim, 2 * self.ndim)
        
        # Motion and observation uncertainty weights
        # These are position-dependent for better scaling
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        
        Args:
            measurement: Bounding box [x, y, a, h] (center_x, center_y, aspect_ratio, height)
            
        Returns:
            mean: Initial state vector
            covariance: Initial state covariance matrix
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])
        
        # Initial covariance
        std = [
            2 * self._std_weight_position * measurement[3],  # x
            2 * self._std_weight_position * measurement[3],  # y
            1e-2,  # aspect ratio
            2 * self._std_weight_position * measurement[3],  # h
            10 * self._std_weight_velocity * measurement[3],  # vx
            10 * self._std_weight_velocity * measurement[3],  # vy
            1e-5,  # va
            10 * self._std_weight_velocity * measurement[3],  # vh
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        
        Args:
            mean: Current state mean
            covariance: Current state covariance
            
        Returns:
            Predicted mean and covariance
        """
        # Motion noise covariance
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
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        
        # Predict
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        
        return mean, covariance
    
    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state to measurement space.
        
        Args:
            mean: State mean
            covariance: State covariance
            
        Returns:
            Projected mean and covariance in measurement space
        """
        # Observation noise
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        observation_cov = np.diag(np.square(std))
        
        mean = self._observation_mat @ mean
        covariance = self._observation_mat @ covariance @ self._observation_mat.T + observation_cov
        
        return mean, covariance
    
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter update step.
        
        Args:
            mean: Predicted state mean
            covariance: Predicted state covariance
            measurement: Observed measurement [x, y, a, h]
            
        Returns:
            Updated mean and covariance
        """
        # Project to measurement space
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Kalman gain
        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(
            chol_factor,
            np.linalg.solve(chol_factor, (covariance @ self._observation_mat.T).T).T
        ).T
        
        # Innovation
        innovation = measurement - projected_mean
        
        # Update
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        
        return new_mean, new_covariance
    
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False
    ) -> np.ndarray:
        """
        Compute gating distance (Mahalanobis distance).
        
        Args:
            mean: State mean
            covariance: State covariance
            measurements: Array of measurements (N, 4)
            only_position: If True, only use x, y for distance
            
        Returns:
            Array of squared Mahalanobis distances
        """
        mean, covariance = self.project(mean, covariance)
        
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        # Cholesky decomposition for efficient computation
        chol_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = np.linalg.solve(chol_factor, d.T).T
        
        return np.sum(z ** 2, axis=1)


def bbox_to_xyah(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Convert bounding box from (x, y, w, h) to (cx, cy, aspect_ratio, h).
    
    Args:
        bbox: (x, y, w, h) format
        
    Returns:
        (cx, cy, a, h) format
    """
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    a = w / h if h > 0 else 0
    return np.array([cx, cy, a, h])


def xyah_to_bbox(xyah: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert from (cx, cy, aspect_ratio, h) to (x, y, w, h).
    
    Args:
        xyah: (cx, cy, a, h) format
        
    Returns:
        (x, y, w, h) format
    """
    cx, cy, a, h = xyah
    w = a * h
    x = cx - w / 2
    y = cy - h / 2
    return (x, y, w, h)

