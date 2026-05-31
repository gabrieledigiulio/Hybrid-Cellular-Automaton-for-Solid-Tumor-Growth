import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class Microenvironment:
    def __init__(self, width, height, oxygen_bg=1.0, glucose_bg=1.0, h_ions_bg=0.0, D_c=0.1, D_g=0.1, D_h=0.1):
        self.width = width
        self.height = height
        self.D_c = D_c
        self.D_g = D_g
        self.D_h = D_h
        self.oxygen_bg = oxygen_bg # Save the dynamic value
        self.glucose_bg = glucose_bg
        self.h_ions_bg = h_ions_bg
        
        # Initialize fields using dynamic background values
        self.oxygen = np.ones((width, height), dtype=float) * oxygen_bg
        self.glucose = np.ones((width, height), dtype=float) * glucose_bg
        self.h_ions = np.ones((width, height), dtype=float) * h_ions_bg

    # Enforce boundary conditions: edges maintain user-specified concentrations
    def enforce_boundary_conditions(self):
        self.oxygen[0, :] = self.oxygen[-1, :] = self.oxygen_bg
        self.oxygen[:, 0] = self.oxygen[:, -1] = self.oxygen_bg
        
        self.glucose[0, :] = self.glucose[-1, :] = self.glucose_bg
        self.glucose[:, 0] = self.glucose[:, -1] = self.glucose_bg
        
        self.h_ions[0, :] = self.h_ions[-1, :] = self.h_ions_bg
        self.h_ions[:, 0] = self.h_ions[:, -1] = self.h_ions_bg

    # Compute 2D discrete finite difference Laplacian
    def _laplacian_2d(self, matrix):
        laplacian = np.zeros_like(matrix)
        laplacian[1:-1, 1:-1] = (
            matrix[2:, 1:-1] + matrix[:-2, 1:-1] +
            matrix[1:-1, 2:] + matrix[1:-1, :-2] -
            4 * matrix[1:-1, 1:-1]
        )
        return laplacian

    # Diffuse resources using numerical substepping
    def diffuse(self, dt=1.0):
        n_substeps = 4
        dt_sub = dt / n_substeps
        for _ in range(n_substeps):
            self.oxygen += self.D_c * self._laplacian_2d(self.oxygen) * dt_sub
            self.glucose += self.D_g * self._laplacian_2d(self.glucose) * dt_sub
            self.h_ions += self.D_h * self._laplacian_2d(self.h_ions) * dt_sub
            
            self.oxygen = np.clip(self.oxygen, 0.0, 1.0)
            self.glucose = np.clip(self.glucose, 0.0, 1.0)
            self.h_ions = np.clip(self.h_ions, 0.0, float('inf'))
            
            self.enforce_boundary_conditions()

    # Decrease metabolites at specific grid cell upon consumption
    def consume_metabolites(self, x, y, o2_consumed, glu_consumed, h_produced):
        self.oxygen[x, y] -= o2_consumed
        self.glucose[x, y] -= glu_consumed
        self.h_ions[x, y] += h_produced
        if self.oxygen[x, y] < 0: self.oxygen[x, y] = 0.0
        if self.glucose[x, y] < 0: self.glucose[x, y] = 0.0