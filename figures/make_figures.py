"""Block diagrams for the manuscript (Figs. 1 to 4).

Springer artwork rules followed: no titles inside figures, sans-serif lettering (Arial, or the
metric-compatible Liberation Sans) at 7.5-8.5 pt, lines >= 0.3 pt, meaning carried by line style
and grey shade rather than colour, width 174 mm, outputs as EPS (fonts embedded), 600 dpi TIFF and PNG.
Run: python figures/make_figures.py   (writes Fig1.* to Fig4.* next to this script)
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


def block(ax, x, y, w, h, title, lines=(), fc="white", ls="-", lw=0.8, gap=3.6):
    rect(ax, x, y, w, h, fc, ls, lw)
    ax.text(x + 2, y + h - 2.6, title, weight="bold", va="top", size=FB)
    for i, ln in enumerate(lines):
        ax.text(x + 2, y + h - 7.4 - gap * i, ln, va="top", size=FS)


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
    block(ax, cx, 58, bw, bh, "Calibrated risk shift planner", ["inspection, charge or swap mode,", "end of life routing (Algorithm 3)"])
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


# ------------------------------------------------------------------ Fig. 4 Algorithm 3 decision flow (shift planner)
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
    save(fig, "Fig4")


# ------------------------------------------------------------------ Fig. 2 physics-guided Bayesian LSTM (edge model)
def sum_node(ax, x, y, r=3.2):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, fc="white", ec="black", lw=0.9))
    ax.plot([x - r * 0.6, x + r * 0.6], [y, y], color="black", lw=0.9)
    ax.plot([x, x], [y - r * 0.6, y + r * 0.6], color="black", lw=0.9)


def fig3():
    fig, ax = plt.subplots(figsize=(W_IN, 3.3))
    ax.set_xlim(0, 174); ax.set_ylim(0, 86); ax.axis("off")
    # learned branch (top)
    block(ax, 1, 52, 36, 22, "Input window", ["SoH history and 12", "health indicators,", "last L = 10 cycles"])
    block(ax, 41, 52, 34, 22, "Variational LSTM", ["2 layers, 64 units,", "one dropout mask", "per sequence"])
    block(ax, 79, 52, 35, 22, "Chemistry adapter", ["16 unit bottleneck,", "residual, one per", "chemistry c"])
    block(ax, 118, 52, 32, 22, "Residual head", ["mean $\\mu_{res}$ and", "log variance s"], gap=4.4)
    line(ax, [(37, 63), (41, 63)]); line(ax, [(75, 63), (79, 63)]); line(ax, [(114, 63), (118, 63)])
    # physics branch (bottom)
    block(ax, 1, 8, 36, 24, "Operating state", ["temperature T,", "C rate c and", "throughput Ah"], fc="#e6e6e6")
    block(ax, 41, 8, 73, 24, "Arrhenius throughput prior", ["rate k = B exp(\u2212(Ea \u2212 Ec c) / RT)",
                                                             "fade grows with k and with Ah to the power z",
                                                             "B, Ea, z pooled across the federation"], fc="#e6e6e6")
    line(ax, [(37, 20), (41, 20)])
    # federated parameter exchange
    rect(ax, 119, 8, 33, 24, fc="white", ls="--", lw=0.9)
    ax.text(135.5, 28.5, "Server pooling", ha="center", va="top", weight="bold", size=FB)
    ax.text(135.5, 23.5, "local (log k, z, T) up,", ha="center", va="top", size=FS)
    ax.text(135.5, 19.5, "(log B, $E_a$, z) down;", ha="center", va="top", size=FS)
    ax.text(135.5, 15.5, "$E_a$ fitted across", ha="center", va="top", size=FS)
    ax.text(135.5, 11.5, "temperatures", ha="center", va="top", size=FS)
    line(ax, [(119, 23), (114, 23)], ls="--"); line(ax, [(114, 17), (119, 17)], ls="--")
    # sum and output
    sum_node(ax, 163, 42)
    line(ax, [(150, 63), (163, 63), (163, 45.2)])
    line(ax, [(76.5, 32), (76.5, 42), (159.8, 42)])
    ax.text(164.5, 54, "residual", ha="left", va="center", size=FS, rotation=90)
    ax.text(100, 43.5, "physics prior", ha="center", va="bottom", size=FS)
    line(ax, [(163, 38.8), (163, 34)])
    ax.text(163, 32.5, "SoH mean", ha="center", va="top", size=FS)
    ax.text(163, 28.8, "= prior +", ha="center", va="top", size=FS)
    ax.text(163, 25.1, "$\\mu_{res}$; T = 50", ha="center", va="top", size=FS)
    ax.text(163, 21.4, "MC passes", ha="center", va="top", size=FS)
    ax.text(163, 17.7, "give $\\sigma^2_{epi}$, $\\sigma^2_{ale}$", ha="center", va="top", size=FS)
    ax.text(1, 82, "Learned branch", weight="bold", va="top", size=FB)
    ax.text(1, 40, "Physics branch", weight="bold", va="top", size=FB)
    save(fig, "Fig2")


# ------------------------------------------------------------------ Fig. 3 drift-weighted federated conformal calibration (Algorithm 2)
def fig4():
    fig, ax = plt.subplots(figsize=(W_IN, 3.4))
    ax.set_xlim(0, 174); ax.set_ylim(0, 92); ax.axis("off")
    rect(ax, 0.5, 1, 46, 90, fc="#f0f0f0", lw=0.9)
    ax.text(3, 88.5, "Target vehicle k", weight="bold", va="top", size=FB)
    ax.text(3, 84.6, "(hotter face, new duty)", va="top", size=FS)
    rect(ax, 52, 1, 66, 90, fc="#dcdcdc", lw=0.9)
    ax.text(54.5, 88.5, "Server (secure sums only)", weight="bold", va="top", size=FB)
    rect(ax, 123.5, 1, 50, 90, fc="white", ls="--", lw=0.9)
    ax.text(126, 88.5, "Calibration vehicles i", weight="bold", va="top", size=FB)
    ax.text(126, 84.6, "(held-out recent cycles)", va="top", size=FS)
    block(ax, 3, 58, 41, 20, "(1) Target embedding", ["$\\mu_Q$: mean RFF of", "recent windows"], gap=4.4)
    block(ax, 126, 58, 45, 20, "(2) Local statistics", ["scores $e_i$, RFF $\\psi_i$;", "send $\\sum \\psi_i$ and count"], gap=4.4)
    block(ax, 55, 58, 60, 20, "(3) Shift direction", ["$\\mu_P$: pooled calibration mean;", "$\\Delta = \\mu_Q - \\mu_P$ broadcast"], gap=4.4)
    block(ax, 55, 32, 60, 21, "(4) Exponential tilt", ["$w_i = \\exp(\\eta \\langle \\psi_i, \\Delta \\rangle)$; $\\eta$ by bisection",
                                                         "on secure sums; effective", "sample size at least 20%"], gap=4.4)
    block(ax, 126, 32, 45, 21, "Weighted histogram", ["of scores $e_i$ with", "weights $w_i$, summed", "securely"])
    block(ax, 55, 6, 60, 21, "(5) Weighted quantile", ["$\\hat{q}_k$ at level 1 \u2212 a with test", "point mass $w_k$ (weighted", "conformal prediction)"], gap=4.4)
    block(ax, 3, 6, 41, 21, "Calibrated interval", ["$\\hat{\\mu} \\pm \\hat{q}_k \\hat{\\sigma}$, passed", "to Algorithm 3"], fc="#e6e6e6", gap=4.4)
    line(ax, [(44, 68), (55, 68)]); line(ax, [(126, 68), (115, 68)])
    line(ax, [(85, 58), (85, 53)])
    line(ax, [(115, 42.5), (126, 42.5)], ls="--"); line(ax, [(148.5, 32), (148.5, 16.5), (115, 16.5)])
    line(ax, [(55, 16.5), (44, 16.5)])
    ax.text(120.5, 44.5, "$\\eta$", ha="center", va="bottom", size=FS)
    save(fig, "Fig3")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("written to", OUT)
