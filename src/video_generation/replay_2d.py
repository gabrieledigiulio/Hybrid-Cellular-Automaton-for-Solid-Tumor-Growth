import argparse
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_REPLAY_PATH = os.path.join(ROOT_DIR, "results", "2d", "2D_Normoxia", "data_2D_Normoxia.npz")


def main():
    parser = argparse.ArgumentParser(description="Replay 2D simulation from .npz")
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_REPLAY_PATH,
        help="Path to .npz file produced by run_all.py",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=40,
        help="Frame interval in ms (lower = faster)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive window (useful when called from run_all.py)",
    )
    args = parser.parse_args()

    data = np.load(args.path)
    cells = data["cells"]
    steps = data["steps"] if "steps" in data else np.arange(len(cells))

    cmap = ListedColormap(["white", "red", "green", "blue"])

    fig, ax = plt.subplots()
    im = ax.imshow(cells[0].T, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    initial_step = steps[0] if len(steps) > 0 else 0
    ax.set_title(f"Step {initial_step}")
    ax.axis("off")

    def update(frame_idx):
        im.set_data(cells[frame_idx].T)
        ax.set_title(f"Step {steps[frame_idx]}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(cells), interval=args.interval, blit=True)

    videos_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.path))[0]

    fps = max(1, int(1000 / max(args.interval, 1)))
    if shutil.which("ffmpeg"):
        out_path = os.path.join(videos_dir, f"{base_name}.mp4")
        anim.save(out_path, fps=fps, dpi=150)
    else:
        out_path = os.path.join(videos_dir, f"{base_name}.gif")
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=150)

    print(f"Saved: {out_path}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
