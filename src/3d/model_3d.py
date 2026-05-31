import os
import sys
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

from grid_3d import SingleGrid3D
from environment_3d import Microenvironment3D
from agent_3d import TumorCell3D


# Count 3D aerobic cells (metabolic output index 3 <= 0.5)
def count_aerobic_3d(m):
    return sum(1 for a in m.agents if a.state in ["PROLIFERATING", "QUIESCENT"] and not (a.outputs is not None and len(a.outputs) > 3 and a.outputs[3] > 0.5))

# Count 3D anaerobic cells (metabolic output index 3 > 0.5)
def count_anaerobic_3d(m):
    return sum(1 for a in m.agents if a.state in ["PROLIFERATING", "QUIESCENT"] and (a.outputs is not None and len(a.outputs) > 3 and a.outputs[3] > 0.5))

class TumorModel3D(Model):
    def __init__(self, width=config.WIDTH_3D, height=config.HEIGHT_3D, depth=config.DEPTH_3D,
                 oxygen_bg=config.OXYGEN_BG, glucose_bg=config.GLUCOSE_BG,
                 h_ions_bg=config.H_IONS_BG, mutation_prob=config.MUTATION_PROB,
                 mutation_std=config.MUTATION_STD,
                 base_o2_rate=config.BASE_O2_RATE,
                 D_c=config.D_C, D_g=config.D_G, D_h=config.D_H,
                 initial_cells=config.INITIAL_CELLS,
                 seed=config.SEED):
        super().__init__(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.oxygen_bg = oxygen_bg["value"] if isinstance(oxygen_bg, dict) else oxygen_bg
        self.glucose_bg = glucose_bg["value"] if isinstance(glucose_bg, dict) else glucose_bg
        self.h_ions_bg = h_ions_bg["value"] if isinstance(h_ions_bg, dict) else h_ions_bg
        self.mutation_prob = mutation_prob["value"] if isinstance(mutation_prob, dict) else mutation_prob
        self.mutation_std = mutation_std["value"] if isinstance(mutation_std, dict) else mutation_std
        self.base_o2_rate = base_o2_rate["value"] if isinstance(base_o2_rate, dict) else base_o2_rate
        self.D_c = D_c["value"] if isinstance(D_c, dict) else D_c
        self.D_g = D_g["value"] if isinstance(D_g, dict) else D_g
        self.D_h = D_h["value"] if isinstance(D_h, dict) else D_h

        self.grid = SingleGrid3D(width, height, depth)
        self.env = Microenvironment3D(
            width,
            height,
            depth,
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
                "Invasive Distance": lambda m: m.invasive_distance(),
                "Shannon Index": lambda m: m.shannon_index(),
                "Aerobic": count_aerobic_3d,
                "Anaerobic": count_anaerobic_3d
            }
        )

        cx, cy, cz = width // 2, height // 2, depth // 2
        self.center = (cx, cy, cz)
        positions = [
            (cx, cy, cz),
            (cx - 1, cy, cz),
            (cx + 1, cy, cz),
            (cx, cy - 1, cz),
            (cx, cy + 1, cz),
            (cx, cy, cz - 1),
            (cx, cy, cz + 1),
        ]
        placed = 0
        for pos in positions:
            x, y, z = pos
            if 0 <= x < width and 0 <= y < height and 0 <= z < depth and self.grid.is_cell_empty(pos):
                self.create_cell(pos, parent_weights=None)
                placed += 1
            if placed >= initial_cells:
                break
        self.running = True

    # Instantiate and place a cell on the 3D grid
    def create_cell(self, pos, parent_weights=None):
        cell = TumorCell3D(self, parent_weights)
        self.grid.place_agent(cell, pos)

    # Handle mitotic division: select free neighbor cell or force quiescence
    def divide_cell(self, parent_cell):
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
        self.grid.remove_agent(cell)
        cell.remove()

    def consume_resources(self, cell, f_factor):
        x, y, z = cell.pos
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

        actual_o2 = min(req_o2, self.env.oxygen[x, y, z])
        actual_glu = min(req_glu, self.env.glucose[x, y, z])
        
        # Correct logic derived from 2D: necrosis if the environment cannot supply what the cell requires
        if (req_o2 > 0.0 and actual_o2 < 0.15 * req_o2) or (req_glu > 0.0 and actual_glu < 0.15 * req_glu):
            cell.state = "NECROTIC"
            return
            
        self.env.consume_metabolites(x, y, z, actual_o2, actual_glu, produced_h)

    # Perform a model simulation step in 3D
    def step(self):
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

    # Calculate current maximum spatial distance of tumor cells from the center in 3D
    def invasive_distance(self):
        if not self.agents:
            return 0.0
        cx, cy, cz = self.center
        max_sq = max(
            ((cell.pos[0] - cx)**2 + (cell.pos[1] - cy)**2 + (cell.pos[2] - cz)**2)
            for cell in self.agents
        )
        return float(np.sqrt(max_sq))

    # Calculate genetic diversity index using Shannon entropy in 3D
    def shannon_index(self):
        if not self.agents:
            return 0.0
        counts = {}
        for cell in self.agents:
            counts[cell.signature] = counts.get(cell.signature, 0) + 1
        total = float(len(self.agents))
        probs = [count / total for count in counts.values()]
        return float(-sum(p * np.log(p) for p in probs if p > 0.0))
