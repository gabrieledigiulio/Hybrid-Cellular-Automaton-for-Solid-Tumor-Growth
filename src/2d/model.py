import os
import sys
import numpy as np
from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import biology and physics components
from agent import TumorCell
from environment import Microenvironment
# Import centralized configurations
import config

# Count aerobic cells (metabolic output index 3 <= 0.5)
def count_aerobic(m):
    """
    Model reporter to count cells in aerobic metabolic state.
    """
    return sum(1 for a in m.agents if a.state in ["PROLIFERATING", "QUIESCENT"] and not (a.outputs is not None and len(a.outputs) > 3 and a.outputs[3] > 0.5))

# Count anaerobic cells (metabolic output index 3 > 0.5)
def count_anaerobic(m):
    """
    Model reporter to count cells in anaerobic metabolic state.
    """
    return sum(1 for a in m.agents if a.state in ["PROLIFERATING", "QUIESCENT"] and (a.outputs is not None and len(a.outputs) > 3 and a.outputs[3] > 0.5))

class TumorModel(Model):
    """
    Mesa Model representing the tumor growth in a 2D microenvironment.
    
    Attributes:
        grid: SingleGrid instance for spatial management.
        env: Microenvironment instance for diffusing substances.
        datacollector: Mesa DataCollector for logging metrics.
    """
    def __init__(self, width=config.WIDTH_2D, height=config.HEIGHT_2D, 
                 oxygen_bg=config.OXYGEN_BG,
                 glucose_bg=config.GLUCOSE_BG,
                 h_ions_bg=config.H_IONS_BG,
                 mutation_prob=config.MUTATION_PROB,
                 mutation_std=config.MUTATION_STD,
                 base_o2_rate=config.BASE_O2_RATE,
                 D_c=config.D_C,
                 D_g=config.D_G,
                 D_h=config.D_H,
                 initial_cells=config.INITIAL_CELLS,
                 seed=config.SEED):
        """
        Initializes the tumor model with physical and biological parameters.
        """
        super().__init__(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # Extract parameter values if they arrive as dictionary inputs from Tornado visualization server
        self.oxygen_bg = oxygen_bg["value"] if isinstance(oxygen_bg, dict) else oxygen_bg
        self.glucose_bg = glucose_bg["value"] if isinstance(glucose_bg, dict) else glucose_bg
        self.h_ions_bg = h_ions_bg["value"] if isinstance(h_ions_bg, dict) else h_ions_bg
        self.mutation_prob = mutation_prob["value"] if isinstance(mutation_prob, dict) else mutation_prob
        self.mutation_std = mutation_std["value"] if isinstance(mutation_std, dict) else mutation_std
        self.base_o2_rate = base_o2_rate["value"] if isinstance(base_o2_rate, dict) else base_o2_rate
        self.D_c = D_c["value"] if isinstance(D_c, dict) else D_c
        self.D_g = D_g["value"] if isinstance(D_g, dict) else D_g
        self.D_h = D_h["value"] if isinstance(D_h, dict) else D_h
            
        self.grid = SingleGrid(width, height, torus=False)
        # Initialize microenvironment with physical parameters
        self.env = Microenvironment(
            width,
            height,
            oxygen_bg=self.oxygen_bg,
            glucose_bg=self.glucose_bg,
            h_ions_bg=self.h_ions_bg,
            D_c=self.D_c,
            D_g=self.D_g,
            D_h=self.D_h,
        )
        
        self.datacollector = DataCollector(
            model_reporters={
                "Total Cells": lambda m: len(m.agents),
                "Proliferating": lambda m: sum(1 for a in m.agents if a.state == "PROLIFERATING"),
                "Quiescent": lambda m: sum(1 for a in m.agents if a.state == "QUIESCENT"),
                "Apoptotic": lambda m: sum(1 for a in m.agents if a.state == "APOPTOTIC"),
                "Necrotic": lambda m: sum(1 for a in m.agents if a.state == "NECROTIC"),
                "Shannon Index": lambda m: m.shannon_index(),
                "Invasive Distance": lambda m: m.invasive_distance(),
                "Aerobic": count_aerobic,
                "Anaerobic": count_anaerobic
            }
        )
        
        # Place the initial tumor cells clustered around the grid center
        cx, cy = width // 2, height // 2
        self.center = (cx, cy)
        positions = [(cx, cy), (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
        placed = 0
        for pos in positions:
            x, y = pos
            if 0 <= x < width and 0 <= y < height and self.grid.is_cell_empty(pos):
                self.create_cell(pos, parent_weights=None)
                placed += 1
            if placed >= initial_cells:
                break
        self.running = True

    # Instantiate and place a cell on the grid
    def create_cell(self, pos, parent_weights=None):
        """
        Creates a new TumorCell and places it on the grid.
        """
        cell = TumorCell(self, parent_weights)
        self.grid.place_agent(cell, pos)

    # Handle mitotic division: select free neighbor cell or force quiescence
    def divide_cell(self, parent_cell):
        """
        Orchestrates cell division by finding an empty adjacent spot.
        """
        neighborhood = self.grid.get_neighborhood(parent_cell.pos, moore=False, include_center=False)
        empty_spots = [pos for pos in neighborhood if self.grid.is_cell_empty(pos)]
        if empty_spots:
            chosen_spot = self.random.choice(empty_spots)
            parent_genetics = (parent_cell.w, parent_cell.W, parent_cell.theta, parent_cell.phi)
            self.create_cell(chosen_spot, parent_genetics)
        else:
            parent_cell.state = "QUIESCENT"

    # Remove cell agent from scheduler and grid
    def remove_cell(self, cell):
        """
        Removes an agent from the simulation.
        """
        self.grid.remove_agent(cell)
        cell.remove()

    # Calculate resource consumption and transition to Necrotic state if resources are below threshold
    def consume_resources(self, cell, f_factor):
        """
        Manages metabolite consumption from the environment based on cell metabolic state.
        """
        x, y = cell.pos

        metabolic_output = 0.0
        if cell.outputs is not None and len(cell.outputs) > 3:
            metabolic_output = cell.outputs[3]
        anaerobic = metabolic_output > 0.5

        if anaerobic:
            req_o2 = 0.0
            req_glu = (18.0 / 6.0) * self.base_o2_rate * f_factor
            produced_h = (self.base_o2_rate * 0.4) * f_factor
        else:
            req_o2 = self.base_o2_rate * f_factor
            req_glu = (1.0 / 6.0) * self.base_o2_rate * f_factor
            produced_h = 0.0

        actual_o2 = min(req_o2, self.env.oxygen[x, y])
        actual_glu = min(req_glu, self.env.glucose[x, y])
        if (req_o2 > 0.0 and actual_o2 < 0.15 * req_o2) or (req_glu > 0.0 and actual_glu < 0.15 * req_glu):
            cell.state = "NECROTIC"
            return
        self.env.consume_metabolites(x, y, actual_o2, actual_glu, produced_h)

    # Perform a model simulation step
    def step(self):
        """
        Advances the simulation by one time step.
        """
        self.env.diffuse(dt=1.0)
        to_remove = [cell for cell in self.agents if cell.state == "APOPTOTIC"]
        for cell in to_remove:
            self.remove_cell(cell)

        living_cells = [cell for cell in self.agents if cell.state != "NECROTIC"]
        self.random.shuffle(living_cells)
        for cell in living_cells:
            cell.step()
        self.datacollector.collect(self)
        living_cells = [cell for cell in self.agents if cell.state != "NECROTIC"]
        if len(living_cells) == 0:
            self.running = False

    # Calculate current maximum spatial distance of tumor cells from the center
    def invasive_distance(self):
        """
        Returns the maximum distance of any cell from the tumor center.
        """
        if not self.agents:
            return 0.0
        cx, cy = self.center
        return max(
            float(np.hypot(cell.pos[0] - cx, cell.pos[1] - cy))
            for cell in self.agents
        )

    # Calculate genetic diversity index using Shannon entropy
    def shannon_index(self):
        """
        Calculates the Shannon Diversity Index based on unique genetic signatures.
        """
        if not self.agents:
            return 0.0
        counts = {}
        for cell in self.agents:
            counts[cell.signature] = counts.get(cell.signature, 0) + 1
        total = float(len(self.agents))
        probs = [count / total for count in counts.values()]
        return float(-sum(p * np.log(p) for p in probs if p > 0.0))
