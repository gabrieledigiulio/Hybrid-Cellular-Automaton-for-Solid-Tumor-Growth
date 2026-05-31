import os
import sys
import importlib.util
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_2D_PATH = os.path.join(BASE_DIR, "2d", "model.py")
MODEL_3D_PATH = os.path.join(BASE_DIR, "3d", "model_3d.py")
sys.path.insert(0, os.path.join(BASE_DIR, "2d"))
sys.path.insert(0, os.path.join(BASE_DIR, "3d"))
sys.path.insert(0, BASE_DIR)
import config


# Helper function to dynamically load modules
def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the 2D and 3D Mesa models
TumorModel = _load_module("model_2d", MODEL_2D_PATH).TumorModel
TumorModel3D = _load_module("model_3d", MODEL_3D_PATH).TumorModel3D

warnings.filterwarnings(
    "ignore",
    message="The use of the `seed` keyword argument is deprecated, use `rng` instead.*",
    category=FutureWarning,
)

RUN_MODE = "both"  # "2d", "3d", "both"
SAVE_REPLAY = True
STEPS_2D = 1000
STEPS_3D = 1000
WIDTH_2D = config.WIDTH_2D
HEIGHT_2D = config.HEIGHT_2D
WIDTH_3D = config.WIDTH_3D
HEIGHT_3D = config.HEIGHT_3D
DEPTH_3D = config.DEPTH_3D
SAMPLE_EVERY_2D = 10
SAMPLE_EVERY_3D = 10


# Encode cell states to integer codes (1: Proliferating, 2: Quiescent, 3: Necrotic/Apoptotic, 0: Empty)
def encode_state(state):
    if state == "PROLIFERATING":
        return 1
    if state == "QUIESCENT":
        return 2
    if state in ("NECROTIC", "APOPTOTIC"):
        return 3
    return 0


# Generate a 2D grid matrix of cell states
def build_cell_matrix_2d(model, width, height):
    matrix = np.zeros((width, height), dtype=np.int8)
    for cell in model.agents:
        x, y = cell.pos
        matrix[x, y] = encode_state(cell.state)
    return matrix


# Generate a 3D grid matrix of cell states
def build_cell_matrix_3d(model, width, height, depth):
    matrix = np.zeros((width, height, depth), dtype=np.int8)
    for cell in model.agents:
        x, y, z = cell.pos
        matrix[x, y, z] = encode_state(cell.state)
    return matrix


def save_plots(df, out_dir, name):
    if df.empty:
        return
    steps = np.arange(len(df))

    counts_fig = plt.figure(figsize=(8, 4))
    plt.plot(steps, df.get("Total Cells", []), label="Total", color="black")
    plt.plot(steps, df.get("Proliferating", []), label="Proliferating", color="red")
    plt.plot(steps, df.get("Quiescent", []), label="Quiescent", color="green")
    plt.plot(steps, df.get("Apoptotic", []), label="Apoptotic", color="purple")
    plt.plot(steps, df.get("Necrotic", []), label="Necrotic", color="blue")
    plt.xlabel("Step")
    plt.ylabel("Cells")
    plt.legend(loc="upper left")
    plt.tight_layout()
    counts_path = os.path.join(out_dir, f"{name}_counts.png")
    counts_fig.savefig(counts_path, dpi=150)
    plt.close(counts_fig)

    metrics_fig = plt.figure(figsize=(8, 3))
    if "Shannon Index" in df:
        plt.plot(steps, df["Shannon Index"], label="Shannon", color="orange")
    if "Invasive Distance" in df:
        plt.plot(steps, df["Invasive Distance"], label="Invasive", color="teal")
    plt.xlabel("Step")
    plt.ylabel("Metric")
    plt.legend(loc="upper left")
    plt.tight_layout()
    metrics_path = os.path.join(out_dir, f"{name}_metrics.png")
    metrics_fig.savefig(metrics_path, dpi=150)
    plt.close(metrics_fig)

    metabolism_fig = plt.figure(figsize=(8, 3))
    if "Aerobic" in df:
        plt.plot(steps, df["Aerobic"], label="Aerobic", color="orange")
    if "Anaerobic" in df:
        plt.plot(steps, df["Anaerobic"], label="Anaerobic", color="purple")
    plt.xlabel("Step")
    plt.ylabel("Cells")
    plt.legend(loc="upper left")
    plt.tight_layout()
    metabolism_path = os.path.join(out_dir, f"{name}_metabolism.png")
    metabolism_fig.savefig(metabolism_path, dpi=150)
    plt.close(metabolism_fig)


def run_experiment(exp, out_dir, log_every=100):
    history_cells = []
    history_steps = []
    sample_every = exp.get("sample_every", 1)
    if exp["dimensions"] == 2:
        model = TumorModel(
            width=exp["width"],
            height=exp["height"],
            **exp["params"]
        )
        for step in range(1, exp["steps"] + 1):
            model.step()
            if step % sample_every == 0 or step == exp["steps"]:
                history_cells.append(build_cell_matrix_2d(model, exp["width"], exp["height"]))
                history_steps.append(step)
            if log_every and step % log_every == 0:
                print(f"[{exp['name']}] step {step}/{exp['steps']}")
    else:
        model = TumorModel3D(
            width=exp["width"],
            height=exp["height"],
            depth=exp["depth"],
            **exp["params"]
        )
        for step in range(1, exp["steps"] + 1):
            model.step()
            if step % sample_every == 0 or step == exp["steps"]:
                history_cells.append(build_cell_matrix_3d(model, exp["width"], exp["height"], exp["depth"]))
                history_steps.append(step)
            if log_every and step % log_every == 0:
                print(f"[{exp['name']}] step {step}/{exp['steps']}")

    npz_path = os.path.join(out_dir, f"data_{exp['name']}.npz")
    np.savez_compressed(
        npz_path,
        cells=np.array(history_cells, dtype=np.int8),
        steps=np.array(history_steps, dtype=np.int32)
    )

    df = model.datacollector.get_model_vars_dataframe()
    if not df.empty:
        save_plots(df, out_dir, exp["name"])

    last_row = df.iloc[-1].to_dict() if not df.empty else {}
    summary = {
        "Experiment": exp["name"],
        "Dimensions": exp["dimensions"],
        "Width": exp["width"],
        "Height": exp["height"],
        "Depth": exp.get("depth", None),
        "base_o2_rate": exp["params"].get("base_o2_rate", config.BASE_O2_RATE),
        "mutation_prob": exp["params"].get("mutation_prob", config.MUTATION_PROB),
        "glucose_bg": exp["params"].get("glucose_bg", config.GLUCOSE_BG),
        "D_g": exp["params"].get("D_g", config.D_G),
    }
    summary.update(last_row)
    return summary


def main():
    experiments = [
        {
            "name": "2D_Normoxia",
            "dimensions": 2,
            "width": WIDTH_2D,
            "height": HEIGHT_2D,
            "steps": STEPS_2D,
            "sample_every": SAMPLE_EVERY_2D,
            "params": {
                "oxygen_bg": 1.0,
                "base_o2_rate": 0.01,
                "mutation_prob": 0.01,
            },
        },
        {
            "name": "2D_Medium_Hypoxia",
            "dimensions": 2,
            "width": WIDTH_2D,
            "height": HEIGHT_2D,
            "steps": STEPS_2D,
            "sample_every": SAMPLE_EVERY_2D,
            "params": {
                "oxygen_bg": 1.0,
                "base_o2_rate": 0.21,
                "mutation_prob": 0.03,
            },
        },
        {
            "name": "2D_Severe_Hypoxia",
            "dimensions": 2,
            "width": WIDTH_2D,
            "height": HEIGHT_2D,
            "steps": STEPS_2D,
            "sample_every": SAMPLE_EVERY_2D,
            "params": {
                "oxygen_bg": 1.0,
                "base_o2_rate": 0.34,
                "mutation_prob": 0.03,
            },
        },
        {
            "name": "3D_Normoxia",
            "dimensions": 3,
            "width": WIDTH_3D,
            "height": HEIGHT_3D,
            "depth": DEPTH_3D,
            "steps": STEPS_3D,
            "sample_every": SAMPLE_EVERY_3D,
            "params": {
                "oxygen_bg": 1.0,
                "base_o2_rate": 0.1,
                "mutation_prob": 0.01,
            },
        },
        {
            "name": "3D_Medium_Hypoxia",
            "dimensions": 3,
            "width": WIDTH_3D,
            "height": HEIGHT_3D,
            "depth": DEPTH_3D,
            "steps": STEPS_3D,
            "sample_every": SAMPLE_EVERY_3D,
            "params": {
                "oxygen_bg": 1.0,
                "base_o2_rate": 0.48,
                "mutation_prob": 0.02,
            },
        },
        {
            "name": "3D_Severe_Hypoxia",
            "dimensions": 3,
            "width": WIDTH_3D,
            "height": HEIGHT_3D,
            "depth": DEPTH_3D,
            "steps": STEPS_3D,
            "sample_every": SAMPLE_EVERY_3D,
            "params": {
                "oxygen_bg": 1.0,
                "base_o2_rate": 1.1,
                "mutation_prob": 0.03,
            },
        },
        {
            "name": "2D_Hypoxia_with_Modified_Glucose_Diffusion",
            "dimensions": 2,
            "width": WIDTH_2D,
            "height": HEIGHT_2D,
            "steps": STEPS_2D,
            "sample_every": SAMPLE_EVERY_2D,
            "params": {
                "oxygen_bg": 1.0,
                "base_o2_rate": 0.14,
                "mutation_prob": 0.04,
                "glucose_bg": 1.0,
                "D_g": 1.0,
            },
        },
        {
            "name": "3D_Hypoxia_with_Modified_Glucose_Diffusion",
            "dimensions": 3,
            "width": WIDTH_3D,
            "height": HEIGHT_3D,
            "depth": DEPTH_3D,
            "steps": STEPS_3D,
            "sample_every": SAMPLE_EVERY_3D,
            "params": {
                "oxygen_bg": 1.0,
                "base_o2_rate": 0.95,
                "mutation_prob": 0.04,
                "glucose_bg": 1.0,
                "D_g": 0.6,
            },
        },
    ]

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    results_2d = os.path.join(results_dir, "2d")
    results_3d = os.path.join(results_dir, "3d")
    os.makedirs(results_2d, exist_ok=True)
    os.makedirs(results_3d, exist_ok=True)

    if RUN_MODE == "2d":
        experiments = [exp for exp in experiments if exp["dimensions"] == 2]
    elif RUN_MODE == "3d":
        experiments = [exp for exp in experiments if exp["dimensions"] == 3]

    summaries = []
    for exp in experiments:
        base_results = results_2d if exp["dimensions"] == 2 else results_3d
        out_dir = os.path.join(base_results, exp["name"])
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== {exp['name']} ({exp['dimensions']}D) ===")
        summaries.append(run_experiment(exp, out_dir, log_every=100))

    df = pd.DataFrame(summaries)
    out_path = os.path.join(results_dir, "summary_table.csv")
    df.to_csv(out_path, index=False)
    print(df)

    if SAVE_REPLAY:
        replay_2d_path = os.path.join(BASE_DIR, "video_generation", "replay_2d.py")
        replay_3d_path = os.path.join(BASE_DIR, "video_generation", "replay_3d.py")
        for exp in experiments:
            if exp["dimensions"] == 2:
                data_path = os.path.join(results_2d, exp["name"], f"data_{exp['name']}.npz")
                print(f"Replay 2D: {data_path}")
                subprocess.run([sys.executable, replay_2d_path, data_path, "--no-show"], check=True)
        for exp in experiments:
            if exp["dimensions"] == 3:
                data_path = os.path.join(results_3d, exp["name"], f"data_{exp['name']}.npz")
                print(f"Replay 3D: {data_path}")
                subprocess.run([sys.executable, replay_3d_path, data_path, "--no-show"], check=True)


if __name__ == "__main__":
    main()
