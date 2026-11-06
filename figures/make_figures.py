"""Block diagrams for the manuscript (Fig. 1 and Fig. 2).

Springer artwork rules followed: no titles inside figures, sans-serif lettering (Arial, or the
metric-compatible Liberation Sans) at 7.5-8.5 pt, lines >= 0.3 pt, meaning carried by line style
and grey shade rather than colour, width 174 mm, outputs as EPS (fonts embedded), 600 dpi TIFF and PNG.
Run: python figures/make_figures.py   (writes Fig1.* and Fig2.* next to this script)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "font.size": 8, "ps.fonttype": 42, "pdf.fonttype": 42,
})
OUT = os.path.dirname(os.path.abspath(__file__))
W_IN = 174 / 25.4
FS, FB = 7.6, 8.2  # body and bold sizes (pt)


def rect(ax, x, y, w, h, fc="white", ls="-", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec="black", lw=lw, ls=ls))


def block(ax, x, y, w, h, title, lines=(), fc="white", ls="-", lw=0.8):
    rect(ax, x, y, w, h, fc, ls, lw)
    ax.text(x + 2, y + h - 2.6, title, weight="bold", va="top", size=FB)
    for i, ln in enumerate(lines):
        ax.text(x + 2, y + h - 7.4 - 3.6 * i, ln, va="top", size=FS)


def line(ax, pts, ls="-", lw=0.8, head=True):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="black", lw=lw, ls=ls, solid_capstyle="butt")
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        dist = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1e-9)
        start = (x1 - (x1 - x0) / dist * 2.0, y1 - (y1 - y0) / dist * 2.0)
        ax.annotate("", xy=pts[-1], xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=lw, ls="-", mutation_scale=8, shrinkA=0, shrinkB=0))


def save(fig, name):
    for ext, kw in (("eps", {}), ("png", {"dpi": 600}),
                    ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.03, **kw)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 1 architecture
def fig1():
    fig, ax = plt.subplots(figsize=(W_IN, 3.5))
    ax.set_xlim(0, 174); ax.set_ylim(0, 92); ax.axis("off")
    rect(ax, 0.5, 1, 61, 90, fc="#f0f0f0", lw=0.9)
    ax.text(3, 88.5, "Battery electric mining vehicle k", weight="bold", va="top", size=FB)
    ax.text(3, 84.6, "(loader or haul truck), k = 1 … K", va="top", size=FS)
    rect(ax, 66, 1, 40, 90, fc="white", ls="--", lw=0.9)
    ax.text(86, 88.5, "Underground network", weight="bold", ha="center", va="top", size=FB)
    ax.text(86, 84.6, "charging bay gateway", ha="center", va="top", size=FS)
    rect(ax, 110.5, 1, 63, 90, fc="#dcdcdc", lw=0.9)
    ax.text(113, 88.5, "Surface cloud", weight="bold", va="top", size=FB)
    ax.text(113, 84.6, "fleet operator", va="top", size=FS)
    bw, bh, vx, cx = 56, 19, 3, 114
    block(ax, vx, 58, bw, bh, "BMS sensors", ["voltage V, current I, temperature T", "logged per charge and discharge"])
    block(ax, vx, 32, bw, bh, "Edge feature extraction", ["12 health indicators + SoH history,", "federated standardisation"])
    block(ax, vx, 6, bw, bh, "Local Bayesian LSTM", ["variational dropout, heteroscedastic", "head, chemistry adapter"])
    line(ax, [(31, 58), (31, 51)]); line(ax, [(31, 32), (31, 25)])
    block(ax, cx, 6, bw, bh, "Drift aware aggregation", ["weights from volume, NLL reliability", "and MMD drift, chemistry clusters"])
    block(ax, cx, 32, bw, bh, "Fleet digital twin", ["SoH and RUL distributions,", "split conformal recalibration"])
    block(ax, cx, 58, bw, bh, "Calibrated risk shift planner", ["inspection, charge or swap mode,", "end of life routing (Algorithm 2)"])
    line(ax, [(142, 25), (142, 32)]); line(ax, [(142, 51), (142, 58)])
    line(ax, [(59, 19), (114, 19)], ls="--", lw=1.1)
    ax.text(86, 24.8, "sparse 8 bit Δθ,", ha="center", va="bottom", size=FS)
    ax.text(86, 21.0, "drift score (uplink)", ha="center", va="bottom", size=FS)
    line(ax, [(114, 11), (59, 11)], ls="-", lw=1.1)
    ax.text(86, 9.0, "global model (downlink)", ha="center", va="top", size=FS)
    line(ax, [(114, 67.5), (59, 67.5)], ls=":", lw=1.3)
    ax.text(86, 65.5, "shift plan", ha="center", va="top", size=FS)
    ax.text(86, 61.8, "(actions)", ha="center", va="top", size=FS)
    save(fig, "Fig1")


# ------------------------------------------------------------------ Fig. 2 Algorithm 2 decision flow
def diamond(ax, cx, cy, w, h, lines):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)], closed=True,
                         fc="white", ec="black", lw=0.8))
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(cx, cy + (n - 1) * 1.8 - 3.6 * i, ln, ha="center", va="center", size=FS)


def cbox(ax, cx, cy, w, h, lines, fc="white", ls="-", lw=0.8, bold_first=False):
    rect(ax, cx - w / 2, cy - h / 2, w, h, fc, ls, lw)
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(cx, cy + (n - 1) * 1.8 - 3.6 * i, ln, ha="center", va="center", size=FS,
                weight="bold" if bold_first and i == 0 else "normal")


def lab(ax, x, y, s, ha="left"):
    ax.text(x, y, s, size=FS, ha=ha, va="center")


def fig2():
    fig, ax = plt.subplots(figsize=(W_IN, 4.9))
    ax.set_xlim(0, 174); ax.set_ylim(0, 132); ax.axis("off")
    A, B, dw, dh = 34, 118, 52, 16
    y_in, y_d1, y_cal, y_d2, y_d3, y_d4, y_fc = 123, 104, 85, 67, 49, 25, 6
    cbox(ax, A, y_in, 64, 14, ["Inputs for vehicle k", "RUL samples, SoH posterior, epistemic",
                               "variance, drift flag, cycles per shift $u_k$"], fc="#e6e6e6", bold_first=True)
    diamond(ax, A, y_d1, dw, dh, ["Drift flag raised?"])
    cbox(ax, B, y_d1, 66, 11, ["Reset calibration window and", "recompute conformal factor $\\hat{q}$"], ls="--")
    cbox(ax, A, y_cal, 64, 14, ["Calibrate intervals with $\\hat{q}$;", "$R_k$ = RUL / $u_k$ (shifts);",
                                "$p_k$ = Pr($R_k \\leq H$)"])
    diamond(ax, A, y_d2, dw, dh, ["Mean SoH ≤ EoL", "threshold?"])
    cbox(ax, B, y_d2, 66, 11, ["Retire pack: second life if lower SoH", "bound is high, else recycle by chemistry"],
         fc="#d0d0d0")
    diamond(ax, A, y_d3, dw, dh, ["$p_k > p_{crit}$ ?"])
    cbox(ax, B, y_d3, 66, 11, ["Queue for inspection, rank by $p_k$,", "fill B bay slots, derate the rest"], lw=1.6)
    diamond(ax, A, y_d4, dw + 16, dh + 8, ["Lower SoH bound below", "derate level, or", "epistemic variance high?"])
    cbox(ax, B, y_d4, 66, 11, ["Battery swap or slow charge,", "cap C rate, SoC window 20–80%"], ls=":", lw=1.3)
    cbox(ax, A, y_fc, 40, 8, ["Fast charge allowed"])
    line(ax, [(A, y_in - 7), (A, y_d1 + dh / 2)])
    line(ax, [(A, y_d1 - dh / 2), (A, y_cal + 7)]); lab(ax, A - 1.5, 93, "no", ha="right")
    line(ax, [(A, y_cal - 7), (A, y_d2 + dh / 2)])
    line(ax, [(A, y_d2 - dh / 2), (A, y_d3 + dh / 2)]); lab(ax, A - 1.5, 58, "no", ha="right")
    line(ax, [(A, y_d3 - dh / 2), (A, y_d4 + (dh + 8) / 2)]); lab(ax, A - 1.5, 37, "no", ha="right")
    line(ax, [(A, y_d4 - (dh + 8) / 2), (A, y_fc + 4)]); lab(ax, A - 1.5, 12, "no", ha="right")
    for y, half in ((y_d1, dw / 2), (y_d2, dw / 2), (y_d3, dw / 2), (y_d4, (dw + 16) / 2)):
        line(ax, [(A + half, y), (B - 33, y)]); lab(ax, A + half + 1.5, y + 2.4, "yes")
    line(ax, [(B, y_d1 - 5.5), (B, 95.5), (A + 0.6, 95.5)])
    line(ax, [(B, y_d3 - 5.5), (B, 38.5), (A + 0.6, 38.5)])
    ax.text(118, 4.5, "H: planning horizon in shifts", size=7, ha="center", va="center")
    ax.text(118, 0.8, "B: inspection bays per shift", size=7, ha="center", va="center")
    save(fig, "Fig2")


if __name__ == "__main__":
    fig1()
    fig2()
    print("written to", OUT)
