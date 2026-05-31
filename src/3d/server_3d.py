import os
import sys
import solara
import time
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

from model_3d import TumorModel3D, count_aerobic_3d, count_anaerobic_3d

step_counter = solara.reactive(0)
oxygen_val = solara.reactive(config.OXYGEN_BG)
glucose_val = solara.reactive(config.GLUCOSE_BG)
D_g_val = solara.reactive(config.D_G)
base_o2_rate_val = solara.reactive(config.BASE_O2_RATE)
D_c_val = solara.reactive(config.D_C)
mutation_val = solara.reactive(config.MUTATION_PROB)
initial_cells_val = solara.reactive(config.INITIAL_CELLS)
seed_val = solara.reactive(config.SEED)
speed_val = solara.reactive(5)
detail_val = solara.reactive(5)
fps_val = solara.reactive(10)
running = solara.reactive(False)
steps_per_tick = solara.reactive(5)
sample_every = solara.reactive(5)

step_count = 0

steps_hist = solara.reactive([])
total_hist = solara.reactive([])
prol_hist = solara.reactive([])
quies_hist = solara.reactive([])
apop_hist = solara.reactive([])
necr_hist = solara.reactive([])
inv_hist = solara.reactive([])
shannon_hist = solara.reactive([])
aerobic_hist = solara.reactive([])
anaerobic_hist = solara.reactive([])

def _make_model():
    depth = config.DEPTH_3D
    return TumorModel3D(
        width=config.WIDTH_3D,
        height=config.HEIGHT_3D,
        depth=depth,
        oxygen_bg=oxygen_val.value,
        glucose_bg=glucose_val.value,
        D_g=D_g_val.value,
        base_o2_rate=base_o2_rate_val.value,
        D_c=D_c_val.value,
        mutation_prob=mutation_val.value,
        initial_cells=initial_cells_val.value,
        seed=seed_val.value,
    )

model = _make_model()


def do_step():
    global model
    global step_count
    model.step()
    step_count += 1
    if step_count % max(sample_every.value, 1) == 0:
        step_counter.value = step_count
        steps_hist.value = steps_hist.value + [step_counter.value]
        total_hist.value = total_hist.value + [len(model.agents)]
        prol_hist.value = prol_hist.value + [sum(1 for c in model.agents if c.state == "PROLIFERATING")]
        quies_hist.value = quies_hist.value + [sum(1 for c in model.agents if c.state == "QUIESCENT")]
        apop_hist.value = apop_hist.value + [sum(1 for c in model.agents if c.state == "APOPTOTIC")]
        necr_hist.value = necr_hist.value + [sum(1 for c in model.agents if c.state == "NECROTIC")]
        inv_hist.value = inv_hist.value + [model.invasive_distance()]
        shannon_hist.value = shannon_hist.value + [model.shannon_index()]
        aerobic_hist.value = aerobic_hist.value + [count_aerobic_3d(model)]
        anaerobic_hist.value = anaerobic_hist.value + [count_anaerobic_3d(model)]


def reset_model():
    global model
    global step_count
    model = _make_model()
    step_counter.value = 0
    step_count = 0
    steps_hist.value = []
    total_hist.value = []
    prol_hist.value = []
    quies_hist.value = []
    apop_hist.value = []
    necr_hist.value = []
    inv_hist.value = []
    shannon_hist.value = []
    aerobic_hist.value = []
    anaerobic_hist.value = []


def build_figure():
    xs, ys, zs, colors = [], [], [], []
    for cell in list(model.agents):
        if cell.state == "APOPTOTIC":
            continue
        x, y, z = cell.pos
        xs.append(x)
        ys.append(y)
        zs.append(z)
        if cell.state == "PROLIFERATING":
            colors.append("red")
        elif cell.state == "QUIESCENT":
            colors.append("green")
        elif cell.state == "NECROTIC":
            colors.append("blue")

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(size=3, color=colors),
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(range=[0, config.WIDTH_3D]),
            yaxis=dict(range=[0, config.HEIGHT_3D]),
            zaxis=dict(range=[0, config.DEPTH_3D]),
            aspectmode="cube",
        ),
        showlegend=False,
    )
    return fig


def build_counts_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps_hist.value, y=total_hist.value, name="Total", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=steps_hist.value, y=prol_hist.value, name="Proliferating", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=steps_hist.value, y=quies_hist.value, name="Quiescent", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=steps_hist.value, y=apop_hist.value, name="Apoptotic", line=dict(color="purple")))
    fig.add_trace(go.Scatter(x=steps_hist.value, y=necr_hist.value, name="Necrotic", line=dict(color="blue")))
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=10), height=400, legend=dict(orientation="h"))
    fig.update_xaxes(title_text="Step")
    return fig


def build_metrics_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps_hist.value, y=shannon_hist.value, name="Shannon", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=steps_hist.value, y=inv_hist.value, name="Invasive Distance", line=dict(color="teal")))
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=10), height=220, legend=dict(orientation="h"))
    fig.update_xaxes(title_text="Step")
    return fig


def build_metabolism_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps_hist.value, y=aerobic_hist.value, name="Aerobic", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=steps_hist.value, y=anaerobic_hist.value, name="Anaerobic", line=dict(color="violet")))
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=10), height=220, legend=dict(orientation="h"))
    fig.update_xaxes(title_text="Step")
    return fig


@solara.component
def Page():
    running_ref = solara.use_ref(running.value)
    fps_ref = solara.use_ref(fps_val.value)
    steps_ref = solara.use_ref(steps_per_tick.value)

    def _clamp(value, min_val, max_val):
        return max(min_val, min(max_val, value))

    def reset_on_param_change():
        oxygen_val.set(_clamp(oxygen_val.value, 0.1, 1.0))
        glucose_val.set(_clamp(glucose_val.value, 0.1, 1.0))
        D_g_val.set(_clamp(D_g_val.value, 0.01, 1.0))
        D_c_val.set(_clamp(D_c_val.value, 0.01, 1.0))
        base_o2_rate_val.set(_clamp(base_o2_rate_val.value, 0.01, 5.0))
        mutation_val.set(_clamp(mutation_val.value, 0.0, 0.1))
        initial_cells_val.set(int(_clamp(initial_cells_val.value, 1, 10)))
        seed_val.set(int(_clamp(seed_val.value, 0, 99999)))
        running.set(False)
        reset_model()

    def sync_speed():
        speed = int(_clamp(speed_val.value, 1, 50))
        speed_val.set(speed)
        fps_val.set(int(_clamp(speed, 1, 60)))
        steps_per_tick.set(int(_clamp(speed, 1, 50)))

    def sync_detail():
        detail = int(_clamp(detail_val.value, 1, 50))
        detail_val.set(detail)
        sample_every.set(int(_clamp(51 - detail, 1, 50)))

    solara.use_effect(sync_speed, [speed_val.value])
    solara.use_effect(sync_detail, [detail_val.value])

    solara.use_effect(
        reset_on_param_change,
        [
            oxygen_val.value,
            glucose_val.value,
            D_g_val.value,
            base_o2_rate_val.value,
            D_c_val.value,
            mutation_val.value,
            initial_cells_val.value,
            seed_val.value,
        ],
    )

    def sync_refs():
        running_ref.current = running.value
        fps_ref.current = fps_val.value
        steps_ref.current = steps_per_tick.value

    solara.use_effect(sync_refs, [running.value, fps_val.value, steps_per_tick.value])

    def run_loop(cancel):
        while not cancel.is_set():
            if running_ref.current:
                for _ in range(max(int(steps_ref.current), 1)):
                    do_step()
                time.sleep(1.0 / max(fps_ref.current, 1))
            else:
                time.sleep(0.1)

    solara.use_thread(run_loop, dependencies=[])

    with solara.Sidebar():
        solara.InputFloat("Tissue Oxygen (c0) [0.1-1.0]", oxygen_val)
        solara.InputFloat("Oxygen Diffusion (D_c) [0.01-1.0]", D_c_val)
        solara.InputFloat("Tissue Glucose (g0) [0.1-1.0]", glucose_val)
        solara.InputFloat("Glucose Diffusion (D_g) [0.01-1.0]", D_g_val)
        solara.InputFloat("Base O2 Consumption (rc) [0.01-5.0]", base_o2_rate_val)
        solara.InputFloat("Mutation Rate (p) [0.0-0.1]", mutation_val)
        solara.InputInt("Initial Cells [1-10]", initial_cells_val)
        solara.InputInt("Seed [0-99999]", seed_val)
        solara.InputInt("Speed [1-50]", speed_val)
        solara.InputInt("Chart Detail [1-50]", detail_val)
        solara.Text(f"Step (sampled): {step_counter.value}")
        solara.Button("Start" if not running.value else "Stop", on_click=lambda: running.set(not running.value))
        solara.Button("Reset", on_click=reset_model)

    fig = build_figure()
    solara.FigurePlotly(fig)
    solara.FigurePlotly(build_counts_figure())
    solara.FigurePlotly(build_metrics_figure())
    solara.FigurePlotly(build_metabolism_figure())
