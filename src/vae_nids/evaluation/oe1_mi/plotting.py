"""Estilo común de figuras de OE1-IM (paleta de referencia del skill dataviz:
categóricos en orden fijo, secuencial de un solo tono azul)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # azul, naranja, aqua
MUTED = "#8a8984"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })
