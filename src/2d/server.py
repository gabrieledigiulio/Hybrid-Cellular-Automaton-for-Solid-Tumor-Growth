import os
import sys

from mesa_viz_tornado.modules import CanvasGrid, ChartModule
from mesa_viz_tornado.ModularVisualization import ModularServer
from mesa_viz_tornado.UserParam import NumberInput, Slider
from model import TumorModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def agent_portrayal(agent):
    """
    Defines how agents are rendered in the 2D grid visualization.
    """
    portrayal = {"Shape": "rect", "Filled": "true", "w": 1, "h": 1, "Layer": 0}
    if agent.state == "PROLIFERATING":
        portrayal["Color"] = "red"
    elif agent.state == "QUIESCENT":
        portrayal["Color"] = "green"
    elif agent.state == "APOPTOTIC":
        portrayal["Color"] = "purple"
    elif agent.state == "NECROTIC":
        portrayal["Color"] = "blue"
    return portrayal

grid = CanvasGrid(agent_portrayal, config.WIDTH_2D, config.HEIGHT_2D, 1200, 1200)

chart = ChartModule([
    {"Label": "Total Cells",   "Color": "Black"},
    {"Label": "Proliferating", "Color": "Red"},
    {"Label": "Quiescent",     "Color": "Green"},
    {"Label": "Apoptotic",     "Color": "Purple"},
    {"Label": "Necrotic",      "Color": "Blue"},
])

metrics_chart = ChartModule([
    {"Label": "Shannon Index", "Color": "Orange"},
    {"Label": "Invasive Distance", "Color": "Teal"}
])

metabolism_chart = ChartModule([
    {"Label": "Aerobic", "Color": "Gold"},
    {"Label": "Anaerobic", "Color": "Violet"}
])

model_params = {
    "oxygen_bg": Slider(
        "Tissue Oxygen (c0)",
        value=config.OXYGEN_BG,
        min_value=0.1,
        max_value=1.0,
        step=0.1
    ),
    "glucose_bg": Slider(
        "Tissue Glucose (g0)",
        value=config.GLUCOSE_BG,
        min_value=0.1,
        max_value=1.0,
        step=0.1
    ),
    "D_g": Slider(
        "Glucose Diffusion (D_g)",
        value=config.D_G,
        min_value=0.01,
        max_value=1.0,
        step=0.01
    ),
    "base_o2_rate": Slider(
        "Base O2 Consumption (rc)",
        value=config.BASE_O2_RATE,
        min_value=0.01,
        max_value=1.0,
        step=0.01
    ),
    "mutation_prob": Slider(
        "Mutation Rate (p)",
        value=config.MUTATION_PROB,
        min_value=0.0,
        max_value=0.1,
        step=0.01
    ),
    "initial_cells": Slider(
        "Initial Cells",
        value=config.INITIAL_CELLS,
        min_value=1,
        max_value=10,
        step=1
    ),
    "seed": NumberInput(
        "Seed",
        value=config.SEED
    ),
}

server = ModularServer(
    TumorModel,
    [grid, chart, metrics_chart, metabolism_chart],
    "Tumor Growth Simulation",
    model_params
)

if __name__ == "__main__":
    server.port = 8521
    server.launch()
