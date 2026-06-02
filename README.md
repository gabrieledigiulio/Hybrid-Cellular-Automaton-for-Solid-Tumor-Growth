# Evolutionary Hybrid Cellular Automaton for Solid Tumor Growth

Agent-based simulation of solid tumor growth based on an **Evolutionary Hybrid Cellular Automaton (HCA)** model, implemented with the [Mesa](https://mesa.readthedocs.io/) framework in 2D and 3D.

Based on the paper: *Anderson, A.R.A. (2005). A hybrid mathematical model of solid tumour invasion: the importance of cell adhesion.* — Journal of Theoretical Biology.

---

## Scientific Model

The simulation combines two modeling paradigms:

### 1. Continuous PDEs (Physical Environment)
- **Diffusion** of Oxygen (O₂), Glucose and Hydrogen Ions (H⁺) via discrete finite difference Laplacian.
- **4-neighbor** stencil (Von Neumann) for 2D and **6-neighbor** stencil for 3D.
- **Dirichlet boundary conditions**: the grid edges maintain the arterial physiological concentrations.

### 2. Discrete Cellular Automaton (Biological Agents)
- Each tumor cell is an **autonomous agent** occupying a grid voxel.
- **Neural Genome**: each cell possesses a feed-forward neural network that maps the local microenvironment (nutrients, waste, crowding) to behavioral decisions:
  - **Proliferation** → cell division
  - **Quiescence** → dormant state
  - **Apoptosis** → programmed cell death
  - **Metabolism** → aerobic/anaerobic switch
- **Evolution**: mitosis introduces Poisson-distributed mutations on neural weights, allowing natural selection on the tumor population.


### Mathematical Context

The simulation is driven by a minimal set of governing equations that bridge physics and biology:

**1. Reaction-Diffusion (Microenvironment)**
The physical fields for Oxygen ($c$), Glucose ($g$), and Hydrogen ions ($h$) update over time via partial differential equations (PDEs):
- **Oxygen**: $\frac{\partial c}{\partial t} = D_c \Delta c - f_c(\vec{x}, t)$
- **Glucose**: $\frac{\partial g}{\partial t} = D_g \Delta g - f_g(\vec{x}, t)$
- **Acidity**: $\frac{\partial h}{\partial t} = D_h \Delta h + f_h(\vec{x}, t)$

**2. Neural Genome (Cellular Decision)**
Each agent computes its behavioral phenotype using a feed-forward network with a modified sigmoid transfer function, $\sigma(x) = \frac{1}{1 + e^{-2x}}$. For an environmental input vector $\xi$:
- **Hidden Layer**: $V_j = \sigma(\sum_k w_{jk} \xi_k - \theta_j)$
- **Output Layer**: $O_i = \sigma(\sum_j W_{ij} V_j - \phi_i)$

**3. Metabolism Modulation**
A cell's metabolic energy demand $F$ scales dynamically based on the strength of its dominant neural response $R$:
- **Demand Function**: $F = \max(k(R - T_r) + 1, 0.25)$

**4. Evolutionary Diversity (Shannon Index)**
Clonal heterogeneity is measured mathematically to track tumor evolution:
- **Diversity**: $H = -\sum (p_i \ln p_i)$

---

## Project Structure

```
project/
├── results/                       # Output directory containing 2d/ and 3d/ simulation results
├── src/                           # Source directory
│   ├── config.py                  # Centralized biological/physical parameters
│   ├── run_all.py                 # Orchestrator: runs all experiments + generates videos
│   │
│   ├── 2d/                        # 2D Model (Mesa)
│   │   ├── model.py               # TumorModel — 2D model orchestrator
│   │   ├── agent.py               # TumorCell — agent with neural genome
│   │   ├── environment.py         # Microenvironment — 2D PDE diffusion
│   │   └── server.py              # Interactive web server (Mesa Tornado)
│   │
│   ├── 3d/                        # 3D Model (Mesa)
│   │   ├── model_3d.py            # TumorModel3D — 3D model orchestrator
│   │   ├── agent_3d.py            # TumorCell3D — 3D agent
│   │   ├── environment_3d.py      # Microenvironment3D — 3D PDE diffusion
│   │   ├── grid_3d.py             # SingleGrid3D — custom 3D grid
│   │   └── server_3d.py           # Interactive 3D web server (Solara + Plotly)
│   │
│   └── video_generation/          # Video creation tools
│       ├── replay_2d.py           # Generates MP4/GIF videos from saved 2D data (.npz)
│       ├── replay_3d.py           # Generates MP4/GIF videos from saved 3D data (.npz)
│       └── videos/                # Videos generated from replays
│
├── README.md                      # Documentation
└── experiment_configurations.txt  # Active experiment configurations info
```

---

## Codebase Files Reference

| File | Description |
|---|---|
| `config.py` | Centralized configuration for all biological and physical parameters of the simulation. |
| `run_all.py` | Global execution orchestrator to run all experiment groups in both 2D and 3D and export data. |
| `2d/agent.py` | `TumorCell` representing 2D cells driven by a neural network genome (with Mesa). |
| `2d/environment.py` | 2D continuous microenvironment resolving PDE diffusion for O2, Glucose, and H+ ions. |
| `2d/model.py` | `TumorModel` orchestrating the 2D spatial simulation (with Mesa). |
| `2d/server.py` | Interactive web visualizer and parameter control dashboard for the 2D simulation (Mesa Tornado). |
| `3d/agent_3d.py` | `TumorCell3D` representing 3D cells driven by a neural network genome (with Mesa). |
| `3d/environment_3d.py` | 3D continuous microenvironment resolving PDE diffusion for O2, Glucose, and H+ ions. |
| `3d/grid_3d.py` | Custom `SingleGrid3D` implementation defining a 3D grid layout (not using Mesa). |
| `3d/model_3d.py` | `TumorModel3D` orchestrating the 3D spatial simulation (with Mesa, but without its grid). |
| `3d/server_3d.py` | Interactive 3D web visualizer dashboard using Solara and Plotly. |
| `video_generation/replay_2d.py` | Visualizer script that loads compressed 2D simulation arrays to generate MP4 or GIF videos. |
| `video_generation/replay_3d.py` | Visualizer script that loads compressed 3D simulation arrays to generate MP4 or GIF videos. |
| `experiment_configurations.txt` | Documentation list detailing physical and biological parameters of all active simulation experiments. |

---

## Installation

### Install dependencies

```bash
pip install mesa mesa-viz-tornado solara numpy pandas matplotlib plotly kaleido imageio imageio-ffmpeg
```

---

## Usage

### Run all experiments + generate videos

```bash
python src/run_all.py
```

This script:
1. Runs **8 experiments** (2D Normoxia, 2D Medium Hypoxia, 2D Severe Hypoxia, 3D Normoxia, 3D Medium Hypoxia, 3D Severe Hypoxia, 2D Hypoxia with Modified Glucose Diffusion, 3D Hypoxia with Modified Glucose Diffusion)
2. Saves data in `results/2d/` and `results/3d/` under specific experiment folders (`.npz` files, PNG plots)
3. Automatically generates the **videos** in `src/video_generation/videos/`
4. Produces a **summary table** in `results/summary_table.csv`

#### Configuration of `src/run_all.py`
The main parameters are configurable directly in the file:
```python
RUN_MODE = "both"    # "2d", "3d", "both" — which experiments to run
SAVE_REPLAY = True   # True/False — automatically generate videos
STEPS_2D = 1000      # Number of steps for 2D experiments
STEPS_3D = 300       # Number of steps for 3D experiments
```

---

### Interactive 2D Server (Mesa Tornado)

```bash
python src/2d/server.py
```

Opens a web interface on **http://localhost:8521** with:
- Interactive 2D grid with real-time visualization of cells
- Population charts (Proliferating, Quiescent, Apoptotic, Necrotic)
- Metrics charts (Shannon Index, Invasive Distance)
- **Sliders** to adjust parameters in real time:
  - Tissue Oxygen (c₀)
  - Tissue Glucose (g₀)
  - Glucose Diffusion (D_g)
  - Base O₂ Consumption (rc)
  - Mutation Rate (p)
  - Initial Cells
  - Seed

---

### Interactive 3D Server (Solara + Plotly)

```bash
solara run src/3d/server_3d.py
```

Opens a web interface with:
- Interactive 3D visualization of the tumor (Plotly Scatter3D — rotatable with the mouse)
- Real-time population and metrics charts
- **Sidebar** with adjustable parameters:
  - Tissue Oxygen, Oxygen Diffusion, Tissue Glucose, Glucose Diffusion, Base O₂ Consumption
  - Mutation Rate, Initial Cells, Seed
  - Speed and Chart Detail
- **Start/Stop** and **Reset** buttons

---

### Video Replay from Saved Data

If you already have `.npz` files and only want to regenerate the videos:

```bash
# 2D Video
python src/video_generation/replay_2d.py results/2d/2D_Normoxia/data_2D_Normoxia.npz

# 3D Video
python src/video_generation/replay_3d.py results/3d/3D_Normoxia/data_3D_Normoxia.npz

# With custom parameters
python src/video_generation/replay_3d.py results/3d/3D_Normoxia/data_3D_Normoxia.npz --frame-step 3 --orbit-speed 2.0 --dpi 200
```

#### `replay_3d.py` Parameters

| Parameter | Default | Description |
|---|---|---|
| `--frame-step` | 5 | Use every N-th frame (lower = smoother) |
| `--interval` | 80 | Interval between frames in ms |
| `--orbit-speed` | 1.5 | Camera rotation degrees per frame (0 = static) |
| `--dpi` | 150 | Video resolution |
| `--no-show` | — | Skip the interactive window |

---

## Configuration

All biological and physical parameters are centralized in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `WIDTH_2D`, `HEIGHT_2D` | 400 | 2D grid dimensions |
| `WIDTH_3D`, `HEIGHT_3D`, `DEPTH_3D` | 100 | 3D grid dimensions |
| `OXYGEN_BG` | 1.0 | Background oxygen concentration (c₀) |
| `GLUCOSE_BG` | 1.0 | Background glucose concentration (g₀) |
| `D_C` | 0.1 | Oxygen diffusion coefficient |
| `MUTATION_PROB` | 0.01 | Mutation probability per gene (p) |
| `MUTATION_STD` | 0.25 | Mutation strength (σ) |
| `BASE_O2_RATE` | 0.03 | Base oxygen consumption rate |
| `SEED` | 43 | Seed for reproducibility |
| `INITIAL_CELLS` | 1 | Initial number of cells |

---

*Developed for the Complex Systems course.*
