"""Kalman filter models for different state spaces."""

import numpy as np
from .filter import KalmanFilter

class SixDModel:
    """Kalman filter with 6-state model: [x, y, vx, vy, ax, ay]."""
    
    def __init__(self, dt, q_diag=None, r_diag=None):
        """
        Initialize 6D Kalman filter.
        
        Args:
            dt: Time step
            q_diag: List of 6 values for Q matrix diagonal
            r_diag: List of 2 values for R matrix diagonal
        """
        self.dt = float(dt)
        
        # State transition matrix: [x, y, vx, vy, ax, ay]
        self.F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)
        
        # Measurement matrix (we measure x, y only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=float)
        
        # Process covariance
        if q_diag is not None:
            self.Q = np.diag(np.array(q_diag, dtype=float))
        else:
            self.Q = np.eye(6) * 1e-2
        
        # Measurement covariance
        if r_diag is not None:
            self.R = np.diag(np.array(r_diag, dtype=float))
        else:
            self.R = np.eye(2) * 1e-1
        
        # State covariance
        self.P = np.eye(6) * 500.0
        
        # State vector
        self.x = np.zeros((6, 1), dtype=float)
        self.initialized = False
    
    def initialize(self, meas):
        """Initialize state from first measurement."""
        m = np.asarray(meas).reshape(2,)
        self.x[0, 0] = float(m[0])
        self.x[1, 0] = float(m[1])
        self.x[2, 0] = 0.0  # vx
        self.x[3, 0] = 0.0  # vy
        self.x[4, 0] = 0.0  # ax
        self.x[5, 0] = 0.0  # ay
        self.initialized = True
    
    def predict(self):
        """Predict next state."""
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
    
    def update(self, meas):
        """Update state with measurement."""
        z = np.asarray(meas).reshape(2, 1)
        if not self.initialized:
            self.initialize(z)
            return
        
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        y = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y)
        self.P = (np.eye(6) - K.dot(self.H)).dot(self.P)
    
    def get_state(self):
        """Get full state vector."""
        return self.x.copy()
    
    def get_position(self):
        """Get position [x, y]."""
        return self.x[:2].reshape(2,)
