import argparse
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection


def get_roots(
    r: float,
    theta: float,
    n: int,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    r_n: float = r ** (1 / n)
    roots: list[complex] = []

    if verbose:
        print(f"\ncalculated {n} roots of z = {r:.3f} * e^(i * {theta:.3f}):")

    for k in range(n):
        angle: float = (theta + 2 * k * np.pi) / n
        root: complex = r_n * np.exp(1j * angle)
        roots.append(root)

        if verbose:
            print(
                f"k = {k}: \t{root.real:+.3f} {root.imag:+.3f}i "
                f"\t(r = {r_n:.3f}, angle = {angle:.3f} rad)"
            )

    return np.array(roots), r_n


def init_plot(fig: Figure, ax: Axes) -> Dict[str, Any]:
    ax.set_aspect("equal")
    ax.axis("off")

    circle: Circle = Circle(
        (0, 0),
        1,
        color="gray",
        fill=False,
        linewidth=1,
        alpha=0.5,
    )
    ax.add_patch(circle)

    arrow_style: dict[str, Any] = dict(
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1,
        color="black",
    )

    re_axis: FancyArrowPatch = FancyArrowPatch(
        (-1, 0), (1, 0), **arrow_style
    )
    im_axis: FancyArrowPatch = FancyArrowPatch(
        (0, -1), (0, 1), **arrow_style
    )

    ax.add_patch(re_axis)
    ax.add_patch(im_axis)

    re_label: Text = ax.text(1, 0, "Re", va="center", fontsize=12)
    im_label: Text = ax.text(0, 1, "Im", ha="center", fontsize=12)

    polygon_line: Line2D
    polygon_line, = ax.plot([], [], "r-", linewidth=1.5, zorder=1)

    scatter: PathCollection = ax.scatter(
        [], [], color="blue", s=50, zorder=2
    )

    labels: list[Text] = []
    for _ in range(12):
        txt: Text = ax.text(
            0,
            0,
            "",
            color="blue",
            fontsize=12,
            ha="center",
            va="center",
            fontweight="bold",
        )
        labels.append(txt)

    origin_label: Text = ax.text(
        0, 0, "O", fontsize=12, style="italic", fontweight="bold"
    )

    title: Text = ax.set_title("")

    return {
        "circle": circle,
        "polygon": polygon_line,
        "scatter": scatter,
        "labels": labels,
        "origin": origin_label,
        "re_label": re_label,
        "im_label": im_label,
        "re_axis": re_axis,
        "im_axis": im_axis,
        "title": title,
    }


def update_plot_artists(
    r: float,
    theta: float,
    n: int,
    artists: Dict[str, Any],
    ax: Axes
) -> None:
    roots, r_n = get_roots(r, theta, n, verbose=False)

    limit: float = r_n * 1.5

    ax.set_xlim(-limit * 1.1, limit * 1.1)
    ax.set_ylim(-limit * 1.1, limit * 1.1)

    artists["circle"].set_radius(r_n)

    artists["re_axis"].set_positions((-limit, 0), (limit, 0))
    artists["im_axis"].set_positions((0, -limit), (0, limit))

    artists["re_label"].set_position((limit * 1.05, 0))
    artists["im_label"].set_position((0, limit * 1.05))

    real_parts: np.ndarray = np.real(roots)
    imag_parts: np.ndarray = np.imag(roots)

    polygon_reals: np.ndarray = np.append(real_parts, real_parts[0])
    polygon_imags: np.ndarray = np.append(imag_parts, imag_parts[0])
    artists["polygon"].set_data(polygon_reals, polygon_imags)

    artists["scatter"].set_offsets(
        np.column_stack([real_parts, imag_parts])
    )

    for k in range(12):
        if k < n:
            angle: float = (theta + 2 * k * np.pi) / n
            offset_r: float = r_n * 1.15
            x: float = offset_r * np.cos(angle)
            y: float = offset_r * np.sin(angle)

            artists["labels"][k].set_position((x, y))
            artists["labels"][k].set_text(f"k = {k}")
            artists["labels"][k].set_visible(True)
        else:
            artists["labels"][k].set_visible(False)

    artists["origin"].set_position((r_n * 0.1, -r_n * 0.1))

    artists["title"].set_text(
        rf"The {n} roots of $z = {r:g} e^{{i {theta / np.pi:.3g} \pi}}$"
    )


def plot_complex_roots(
    r: float,
    theta: float,
    n: int,
    save: bool = True
) -> Figure:
    roots, r_n = get_roots(r, theta, n, verbose=save)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6, 6))

    artists = init_plot(fig, ax)
    update_plot_artists(r, theta, n, artists, ax)

    if save:
        output_filename: str = "complex_roots.png"
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to '{output_filename}'")
        plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and plot the n-th roots of a complex number."
    )
    parser.add_argument("--r", type=float, default=4.0)
    parser.add_argument("--theta", type=float, default=np.pi / 3.0)
    parser.add_argument("--n", type=int, default=3)

    args = parser.parse_args()
    plot_complex_roots(args.r, args.theta, args.n)
