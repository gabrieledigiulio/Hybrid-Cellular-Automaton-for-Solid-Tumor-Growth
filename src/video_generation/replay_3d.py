import argparse
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_REPLAY_PATH = os.path.join(ROOT_DIR, "results", "3d", "3D_Normoxia", "data_3D_Normoxia.npz")

# Colors per state (1=Proliferating, 2=Quiescent, 3=Necrotic)
COLORS = {
    1: "#DC3232",   # red — proliferating
    2: "#32CD32",   # green — quiescent
    3: "#4169E1",   # blue   — necrotic
}
ALPHAS = {1: 0.9, 2: 0.7, 3: 0.35}
SIZES  = {1: 12,  2: 8,   3: 5}
LABELS = {1: "Proliferating", 2: "Quiescent", 3: "Necrotic"}


def main():
    parser = argparse.ArgumentParser(description="Replay 3D simulation from .npz (Matplotlib)")
    parser.add_argument(
        "path", nargs="?", default=DEFAULT_REPLAY_PATH,
        help="Path to .npz file produced by run_all.py",
    )
    parser.add_argument("--interval", type=int, default=80,
                        help="Frame interval in ms (lower = faster)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Use every Nth simulation step")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip interactive window")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI")
    parser.add_argument("--orbit-speed", type=float, default=1.5,
                        help="Camera rotation degrees per frame (0 = static)")
    args = parser.parse_args()

    data = np.load(args.path)
    cells = np.asarray(data["cells"])
    steps = data["steps"] if "steps" in data else np.arange(len(cells))
    total = len(cells)
    if total < 2:
        raise ValueError("Not enough frames.")

    frame_step = max(1, min(args.frame_step, total))
    indices = list(range(0, total, frame_step))
    x_max, y_max, z_max = cells[0].shape

    # ── Setup figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8), facecolor="#0F0F19")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0F0F19")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_zlim(0, z_max)
    ax.set_xlabel("X", color="white", fontsize=10)
    ax.set_ylabel("Y", color="white", fontsize=10)
    ax.set_zlabel("Z", color="white", fontsize=10)
    ax.tick_params(colors="grey", labelsize=7)
    # Panel backgrounds
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.3, 0.3, 0.4, 0.3))
    ax.yaxis.pane.set_edgecolor((0.3, 0.3, 0.4, 0.3))
    ax.zaxis.pane.set_edgecolor((0.3, 0.3, 0.4, 0.3))
    ax.grid(True, alpha=0.15)

    title = ax.set_title("", color="white", fontsize=13, fontweight="bold", pad=15)

    # Scatter handles — one per state, drawn in the correct order
    scatters = {}
    for state_id in [3, 2, 1]:  # necrotic behind, proliferating in front
        sc = ax.scatter([], [], [], c=COLORS[state_id], s=SIZES[state_id],
                        alpha=ALPHAS[state_id], label=LABELS[state_id],
                        edgecolors="none", depthshade=True)
        scatters[state_id] = sc

    ax.legend(loc="upper right", fontsize=9, framealpha=0.6,
              facecolor="#1a1a2e", edgecolor="grey", labelcolor="white")

    base_elev = 25
    base_azim = 45

    total_simulation_steps = steps[-1] if len(steps) > 0 else total

    def update(frame_num):
        idx = indices[frame_num]
        frame = cells[idx]
        step_val = steps[idx]

        for state_id in [3, 2, 1]:
            coords = np.argwhere(frame == state_id)
            sc = scatters[state_id]
            if len(coords) > 0:
                sc._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
            else:
                sc._offsets3d = ([], [], [])

        # Camera rotation
        azim = base_azim + frame_num * args.orbit_speed
        ax.view_init(elev=base_elev, azim=azim)

        n_total = int(np.sum(frame > 0))
        n_pro = int(np.sum(frame == 1))
        n_qui = int(np.sum(frame == 2))
        n_nec = int(np.sum(frame == 3))
        title.set_text(
            f"Tumor Growth 3D — Step {step_val}/{total_simulation_steps}\n"
            f"Cells: {n_total}  |  P: {n_pro}  Q: {n_qui}  N: {n_nec}"
        )
        return list(scatters.values()) + [title]

    anim = FuncAnimation(fig, update, frames=len(indices),
                         interval=args.interval, blit=False)

    # ── Save ────────────────────────────────────────────────────────
    videos_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.path))[0]
    fps = max(1, int(1000 / max(args.interval, 1)))

    if shutil.which("ffmpeg"):
        out_path = os.path.join(videos_dir, f"{base_name}.mp4")
        anim.save(out_path, writer="ffmpeg", fps=fps, dpi=args.dpi,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
    else:
        out_path = os.path.join(videos_dir, f"{base_name}.gif")
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=args.dpi,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})

    file_kb = os.path.getsize(out_path) / 1024
    print(f"Saved: {out_path} ({file_kb:.0f} KB, {len(indices)} frames)")

    if not args.no_show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
