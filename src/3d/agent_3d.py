import os
import sys
import numpy as np
from mesa import Agent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Represents a single 3D tumor cell agent with a neural genome
class TumorCell3D(Agent):
    def __init__(self, model, parent_weights=None):
        super().__init__(model)

        # Genetic inheritance or base initialization
        if parent_weights is not None:
            self.w, self.W, self.theta, self.phi = self.mutate(*parent_weights)
        else:
            self.w, self.W, self.theta, self.phi = self.init_base_phenotype()

        self.state = "PROLIFERATING"
        self.age = 0.0
        self.proliferation_age = max(np.random.normal(16.0, 8.0), 4.0)
        self.target_response = 0.675
        self.k_modulation = 6.0
        self.outputs = None
        self.signature = self._compute_signature()

    # Initialize base phenotype weights and thresholds
    def init_base_phenotype(self):
        w = np.array([
            [1.0,  0.0,  0.0,  0.0],
            [0.5,  0.0,  0.0,  0.0],
            [0.0, -2.0,  0.0,  0.0],
            [0.0,  0.0, -2.0,  0.5],
            [1.0,  0.0,  0.0,  0.0]
        ])
        W = np.array([
            [-0.5,  1.0, -0.5,  0.0,  0.0],
            [ 0.0,  0.55,-0.5,  0.0,  0.0],
            [ 0.0,  0.0,  2.0,  2.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0,  1.0]
        ])
        theta = np.array([0.55, 0.0, 0.7, -0.25, 0.0])
        phi = np.array([0.0, 0.0, 0.0, 0.0, 0.75])
        return w, W, theta, phi

    # Mutate neural weights and thresholds based on model mutation probability
    def mutate(self, p_w, p_W, p_theta, p_phi):
        prob = self.model.mutation_prob
        sigma = self.model.mutation_std

        w = self._mutate_array(p_w, prob, sigma)
        W = self._mutate_array(p_W, prob, sigma)
        theta = self._mutate_array(p_theta, prob, sigma)
        phi = self._mutate_array(p_phi, prob, sigma)

        return w, W, theta, phi

    # Helper function to mutate array with Poisson-distributed mutations
    def _mutate_array(self, arr, prob, sigma):
        n = arr.size
        n_mutations = np.random.poisson(prob * n)
        if n_mutations > 0:
            flat = arr.flatten().copy()
            indices = np.random.choice(n, size=min(n_mutations, n), replace=False)
            flat[indices] += np.random.normal(0, sigma, len(indices))
            return flat.reshape(arr.shape)
        return np.copy(arr)

    # Sigmoid activation function
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-2.0 * x))

    # Feed-forward response based on neural network weights
    def calculate_response(self, inputs):
        V = self._sigmoid(np.dot(self.w, inputs) - self.theta)
        O = self._sigmoid(np.dot(self.W, V) - self.phi)
        return O

    # Cell behavioral step based on environmental inputs (Von Neumann 6-neighbors)
    def step(self):
        if self.state in ["APOPTOTIC", "NECROTIC"]:
            return

        x, y, z = self.pos
        neighbors = len(self.model.grid.get_neighbors(self.pos, moore=False, include_center=False))
        o2 = self.model.env.oxygen[x, y, z]
        glu = self.model.env.glucose[x, y, z]
        h_ion = self.model.env.h_ions[x, y, z]

        inputs = np.array([float(neighbors) / 6.0, o2, glu, h_ion])
        outputs = self.calculate_response(inputs)
        self.outputs = outputs

        life_cycle_outputs = outputs[0:3]
        max_idx = np.argmax(life_cycle_outputs)
        max_response = life_cycle_outputs[max_idx]

        F = max(self.k_modulation * (max_response - self.target_response) + 1.0, 0.25)
        self.age += F

        if max_idx == 2:
            self.state = "APOPTOTIC"
            return
        elif max_idx == 1:
            self.state = "QUIESCENT"
            self.model.consume_resources(self, F / 5.0)
        elif max_idx == 0:
            self.state = "PROLIFERATING"
            self.model.consume_resources(self, F)
            if self.age >= self.proliferation_age:
                self.model.divide_cell(self)
                self.age = 0.0

    # Unique genetic signature identifier for the cell
    def _compute_signature(self, decimals=3):
        w_f = np.round(self.w, decimals=decimals).flatten()
        W_f = np.round(self.W, decimals=decimals).flatten()
        t_f = np.round(self.theta, decimals=decimals).flatten()
        p_f = np.round(self.phi, decimals=decimals).flatten()
        return tuple(np.concatenate([w_f, W_f, t_f, p_f]).tolist())
