"""PEM-Flow style adapted for the course figure; see docs/REUSE.md.

The apply/panel_label structure, Arial typography, TrueType PDF export,
white canvas and restrained axes come from the existing pemflow_style.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_PED = "#0072B2"
C_NEUTRAL = "#4D4D4D"
C_GRID = "#DDDDDD"


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.hashsalt": "traj-course-week01-v1",
    })


def panel_label(ax, label, x=-0.14, y=1.075):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left", color="#152A3A")


def save(fig, base, dpi=300):
    """Publishable PNG plus editable paths/text in PDF and SVG."""
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=dpi)
    fig.savefig(base.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".svg"), metadata={"Date": None})
    svg = base.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
