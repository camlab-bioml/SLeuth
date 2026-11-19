#!/usr/bin/env python3
"""Redraw fig_preprocessing.pdf and fig_model.pdf for the revised paper.

Same canvas helpers and colours as paper/figures/make_figures.py (git 7ee75a5)
and the submitted figures.
Run with the system python (matplotlib 3.9.4, the version that drew the
submitted PDFs):  FIG_OUT=<dir> python3 make_figures.py
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(os.environ.get("FIG_OUT") or Path(__file__).resolve().parent)
plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42})

BLUE_FILL, BLUE_EDGE = "#dfe6f3", "#3b5998"
TAN_FILL, TAN_EDGE = "#f9efe0", "#c38d3d"
GREY_FILL, GREY_EDGE = "#ededed", "#9d9d9d"
WHITE_EDGE = "#b7b7b7"
NOTE, ARROW, ACCENT = "#6e6e6e", "#6e6e6e", "#3b5998"
STYLE = {"blue": (BLUE_FILL, BLUE_EDGE), "tan": (TAN_FILL, TAN_EDGE),
         "grey": (GREY_FILL, GREY_EDGE), "white": ("white", WHITE_EDGE)}
PX_PER_IN = 1600 / 5.78
FONT = 0.74


class Canvas:
    def __init__(self, height):
        self.h = height
        self.fig = plt.figure(figsize=(1600 / PX_PER_IN, height / PX_PER_IN))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1600)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")

    def y(self, y):
        return self.h - y

    def box(self, x0, y0, x1, y1, text="", kind="blue", size=11.5, lw=1.3,
            dashed=False, spacing=1.15, color="black", edge=None):
        fill, ec = STYLE[kind] if not dashed else ("none", BLUE_EDGE)
        p = FancyBboxPatch((x0, self.y(y1)), x1 - x0, y1 - y0,
                           boxstyle="round,pad=0,rounding_size=7",
                           fc=fill, ec=edge or ec, lw=lw,
                           ls=(0, (3, 2.2)) if dashed else "-")
        self.ax.add_patch(p)
        if text:
            self.ax.text((x0 + x1) / 2, self.y((y0 + y1) / 2), text,
                         ha="center", va="center", fontsize=size * FONT,
                         linespacing=spacing, color=color)

    def note(self, x, y, text, ha="left", color=NOTE, size=10.5, italic=True,
             weight="normal"):
        self.ax.text(x, self.y(y), text, ha=ha, va="center", color=color,
                     fontsize=size * FONT, style="italic" if italic else "normal",
                     linespacing=1.2, weight=weight)

    def arrow(self, p0, p1, rad=0.0, color=ARROW):
        a = FancyArrowPatch((p0[0], self.y(p0[1])), (p1[0], self.y(p1[1])),
                            arrowstyle="-|>", mutation_scale=9, lw=1.2,
                            color=color, shrinkA=0, shrinkB=1,
                            connectionstyle=f"arc3,rad={rad}")
        self.ax.add_patch(a)

    def line(self, pts, color=ARROW, lw=1.2):
        xs, ys = zip(*pts)
        self.ax.plot(xs, [self.y(v) for v in ys], color=color, lw=lw,
                     solid_capstyle="butt")

    def plus(self, x, y, r=21):
        self.ax.add_patch(Circle((x, self.y(y)), r, fc="white", ec=ACCENT, lw=1.4))
        self.ax.text(x, self.y(y) + 1, "+", ha="center", va="center",
                     fontsize=13 * FONT, color=ACCENT)

    def rule(self, y, x0=5, x1=1595):
        self.line([(x0, y), (x1, y)], color="#d4d4d4", lw=1.0)

    def save(self, name):
        self.fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
        self.fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.01,
                         dpi=150)
        plt.close(self.fig)


def fig_preprocessing():
    c = Canvas(540)
    c.note(5, 28, "pretrained sources", size=11)
    cells = [("expression", "protein seq"), ("text", "PPI"), ("GO", "KG")]
    for r, (a, b) in enumerate(cells):
        y0 = 78 + r * 82
        c.box(5, y0, 158, y0 + 69, a, "grey", size=10.3)
        c.box(167, y0, 320, y0 + 69, b, "grey", size=10.3)
    c.line([(332, 78), (350, 78), (350, 311), (332, 311)], lw=1.1)
    c.arrow((350, 140), (387, 112))
    c.arrow((350, 260), (387, 372))

    c.note(387, 45, "single-modality stream", size=11)
    c.box(387, 75, 1254, 153, "one embedding, native dimension   ·   no "
          "reduction", "grey", size=12)

    c.note(387, 318, "multimodal stream", size=11)
    c.box(387, 343, 672, 428, "6 winners per split\nor all 16",
          "grey", size=11)
    steps = [(705, 840, "impute"), (870, 1015, "normalise"), (1045, 1185, "concat")]
    prev = 672
    for x0, x1, t in steps:
        c.arrow((prev, 385), (x0, 385))
        c.box(x0, 343, x1, 428, t, "white", size=10.5)
        prev = x1
    c.note(1045, 452, "no PCA", size=10.5)

    c.plus(1340, 265)
    c.arrow((1254, 114), (1328, 247), rad=-0.35)
    c.arrow((1185, 385), (1328, 283), rad=-0.25)
    c.note(1300, 350, "cell-line embedding\n(SLeuth-CL only)", color=ACCENT, size=10.5)
    c.arrow((1361, 265), (1410, 265))
    c.box(1410, 222, 1592, 310, "shared\nencoder", "blue", size=12.5, lw=1.6)

    c.rule(482)
    c.note(5, 510, "frozen", size=11)
    c.note(1592, 510, "trainable", ha="right", color=ACCENT, size=11)
    c.save("fig_preprocessing")


def fig_model():
    c = Canvas(1060)
    # ---- A: shared encoder
    c.note(5, 18, "A", italic=False, color="black", size=14, weight="bold")
    c.note(545, 30, r"identical encoder $f_{\theta}$", ha="center", color=ACCENT,
           size=11.5)
    c.box(357, 52, 735, 446, dashed=True, lw=1.3)
    for y0, g, h in ((76, r"gene $i$", r"$h_i,\ z_i$"), (276, r"gene $j$", r"$h_j,\ z_j$")):
        c.box(9, y0, 228, y0 + 92, g, "grey", size=13)
        c.box(387, y0, 706, y0 + 92, "encoder\n" + r"$f_{\theta}$", "blue", size=13)
        c.box(891, y0, 1080, y0 + 92, h, "tan", size=14)
        c.arrow((228, y0 + 46), (387, y0 + 46))
        c.arrow((706, y0 + 46), (891, y0 + 46))
    c.line([(1148, 40), (1148, 440)], color="#d4d4d4", lw=1.0)
    c.note(1217, 30, r"encoder $f_{\theta}$", size=11.5)
    c.box(1217, 60, 1596, 160, "entry\n" + r"$\mathbb{R}^{d\,(+24)}\rightarrow"
          r"\mathbb{R}^{32}$", "blue", size=12.5, spacing=1.45)
    c.box(1217, 197, 1596, 295, "residual block\n×5", "blue", size=12.5)
    c.box(1217, 335, 1596, 425, r"$W_{-1}$" + "\n" + r"$32\times32$, plus $c_H$",
          "blue", size=12.5)
    c.arrow((1406, 160), (1406, 197))
    c.arrow((1406, 295), (1406, 335))
    c.note(1217, 468, "+24 only in SLeuth-CL;\nno layer normalization", size=10)
    c.rule(508)

    # ---- B: kernel
    c.note(5, 526, "B", italic=False, color="black", size=14, weight="bold")
    c.box(9, 556, 139, 642, r"$h_i$", "tan", size=13)
    c.box(9, 684, 139, 770, r"$h_j$", "tan", size=13)
    c.note(291, 532, r"$z = W_{-1}h + c_H$", size=11.5)
    c.box(291, 553, 626, 636, r"$\langle z_i,\, z_j\rangle$", "blue", size=13)
    c.box(291, 675, 626, 760, r"$\varepsilon\,\langle h_i,\, h_j\rangle$", "white",
          size=13)
    for (ya, yb) in ((599, 594), (599, 717), (727, 594), (727, 717)):
        c.arrow((139, ya), (291, yb))
    c.plus(700, 655)
    c.arrow((626, 594), (684, 641), rad=-0.3)
    c.arrow((626, 717), (684, 669), rad=0.3)
    c.note(781, 655, "=", ha="center", italic=False, color="black", size=18)
    c.box(867, 600, 1596, 712,
          r"$k_H(i,j)=\langle \phi_H(h_i),\, \phi_H(h_j)\rangle$" + "\n"
          + r"$\phi_H(h) = (W_{-1}h + c_H,\ \sqrt{\varepsilon}\,h)$",
          "blue", size=12.5)
    c.line([(1231, 712), (1231, 800), (228, 800), (228, 905)])
    c.arrow((228, 905), (228, 918))
    c.rule(840)

    # ---- C: score per head
    c.note(5, 862, "C", italic=False, color="black", size=14, weight="bold")
    xs = [(109, 300, r"$\times\, \tau$", "blue", "black"),
          (330, 800, r"$+\,g(h_i)+g(h_j)+p(z_i,z_j)$", "blue", "black"),
          (830, 990, r"$+\, b_H$", "blue", "black"),
          (1020, 1180, r"$\sigma(\cdot)$", "white", "black"),
          (1210, 1596, r"$P(\mathrm{SL}\,|\,i, j, H)$", "tan", "black")]
    for x0, x1, t, kind, col in xs:
        c.box(x0, 918, x1, 1003, t, kind, size=13, color=col)
    for (a, b) in ((300, 330), (800, 830), (990, 1020), (1180, 1210)):
        c.arrow((a, 960), (b, 960))
    c.note(330, 1035, "per-gene and pair terms, both zero at start", size=10)
    c.note(1596, 1035, "SLeuth: one head (SL in any cell line)", ha="right",
           size=10)
    c.save("fig_model")


if __name__ == "__main__":
    fig_preprocessing()
    fig_model()
    print(f"wrote fig_preprocessing.pdf and fig_model.pdf to {OUT}")
