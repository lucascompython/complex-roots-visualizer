import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def get_roots(r: float, theta: float, n: int, verbose: bool = True) -> Tuple[np.ndarray, float]:
    """
    calculates the n-th roots of a complex number given by r * e^(i * theta).
    """
    r_n = r ** (1 / n)
    roots = []

    if verbose:
        print(f"\ncalculated {n} roots of z = {r:.3f} * e^(i * {theta:.3f}):")
    for k in range(n):
        angle = (theta + 2 * k * np.pi) / n
        root = r_n * np.exp(1j * angle)
        roots.append(root)
        if verbose:
            print(
                f"k = {k}: \t{root.real:+.3f} {root.imag:+.3f}i \t(r = {r_n:.3f}, angle = {angle:.3f} rad)"
            )

    return np.array(roots), r_n


def plot_complex_roots(r: float, theta: float, n: int, save: bool = True, fig=None, ax=None):
    roots, r_n = get_roots(r, theta, n, verbose=save)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        ax.clear()

    circle = plt.Circle(
        (0, 0), r_n, color="gray", fill=False, linestyle="-", linewidth=1, alpha=0.5
    )
    ax.add_patch(circle)

    # axes limits based on radius
    limit = r_n * 1.5

    # Re axis (horizontal)
    ax.annotate(
        "",
        xy=(limit, 0),
        xytext=(-limit, 0),
        arrowprops=dict(arrowstyle="->", color="k", linewidth=1),
    )
    ax.text(limit + 0.05 * limit, 0, "Re", va="center", fontsize=12)

    # Im axis (vertical)
    ax.annotate(
        "",
        xy=(0, limit),
        xytext=(0, -limit),
        arrowprops=dict(arrowstyle="->", color="k", linewidth=1),
    )
    ax.text(0, limit + 0.05 * limit, "Im", ha="center", fontsize=12)

    # plot roots
    real_parts = np.real(roots)
    imag_parts = np.imag(roots)

    # connect roots to form a polygon
    polygon_reals = np.append(real_parts, real_parts[0])
    polygon_imags = np.append(imag_parts, imag_parts[0])
    ax.plot(polygon_reals, polygon_imags, "r-", linewidth=1.5, zorder=1)

    # scatter roots on top
    ax.scatter(real_parts, imag_parts, color="blue", s=50, zorder=2)

    for k in range(n):
        offset_r = r_n * 1.15
        angle = (theta + 2 * k * np.pi) / n
        text_x = offset_r * np.cos(angle)
        text_y = offset_r * np.sin(angle)

        ax.text(
            text_x,
            text_y,
            f"k = {k}",
            color="blue",
            fontsize=12,
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.text(r_n * 0.1, -r_n * 0.1, "O", fontsize=12, style="italic", fontweight="bold")

    ax.set_aspect("equal")
    ax.set_xlim(-limit * 1.1, limit * 1.1)
    ax.set_ylim(-limit * 1.1, limit * 1.1)
    ax.axis("off")

    plt.title(rf"The {n} roots of $z = {r:g} e^{{i {theta / np.pi:.3g} \pi}}$")
    plt.tight_layout()

    if save:
        output_filename = "complex_roots.png"
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to '{output_filename}'")

        import matplotlib

        if matplotlib.get_backend().lower() != "agg":
            plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and plot the n-th roots of a complex number."
    )
    parser.add_argument(
        "--r", type=float, default=4.0, help="Magnitude (r) of the complex number"
    )
    parser.add_argument(
        "--theta", type=float, default=np.pi / 3.0, help="Angle (theta) in radians"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="The degree of the root (e.g. 3 for cubic roots)",
    )

    args = parser.parse_args()
    plot_complex_roots(args.r, args.theta, args.n)
