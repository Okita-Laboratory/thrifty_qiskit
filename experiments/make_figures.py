"""
make_figures.py -- generate the figures for both papers as PDF.

Figures produced (in ./figs/):
  fig_reduction_stages.pdf   theory  : circuit count and quantum share by stage
  fig_two_axes.pdf           theory  : A/B/C groups on (surrogatability, measurability)
  fig_conversion_rate.pdf    theory  : conversion rate of the eight measured cases
  fig_speedup_v2.pdf         theory  : cost of the shadow sampler, naive vs v2
  fig_layers.pdf             impl    : the three design conventions as a layer diagram
  fig_test_types.pdf         impl    : test counts by module and type
  fig_abs_design.pdf         abstract: overall tool design (thrifty / causal)
  fig_abs_speedup.pdf        abstract: circuit-count and classical-side speedup
  fig_abs_causal.pdf         abstract: partial identification and orientation
  fig_structure_cost.pdf     theory  : reconstruction cost vs feature dimension
  fig_hw_measured.pdf        both    : measured wall clock on ibm_fez

Run:  python3 -m experiments.make_figures
Needs: matplotlib.  A CJK font is selected automatically if available.

Each figure is written as .pdf, .eps and .png.
  * pdflatex / lualatex : use the .pdf as is.
  * platex/uplatex + dvipdfmx : use the .pdf, but run
        extractbb figs/*.pdf
    once to create the .xbb files, and load graphicx with the
    dvipdfmx driver option.
  * latex + dvips : the .eps is picked up automatically.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ***********************************************
# Japanese font
# ***********************************************
for fam in ("Noto Sans CJK JP", "IPAexGothic", "IPAPGothic", "Noto Sans CJK TC",
            "Noto Sans CJK KR", "TakaoPGothic", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue
plt.rcParams.update({"font.size": 9, "axes.grid": False, "pdf.fonttype": 42})
OUT = "figs"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    """Write PDF, EPS and PNG so that any TeX driver works.
    PDF   : pdflatex / lualatex (and dvipdfmx after `extractbb`)
    EPS   : latex + dvips
    PNG   : quick preview
    """
    stem = os.path.splitext(name)[0]
    for ext in ("pdf", "eps", "png"):
        path = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, dpi=200)
        print("wrote", path)
    plt.close(fig)


# ***********************************************
# 1. Reduction stages (theory)
# ***********************************************
def fig_reduction_stages():
    labels = ["素朴", "添字操作化", "+ Nystrom\n(m=30)", "厳密再構成"]
    circuits = [1e7, 2e4, 6e3, 3]
    t_q = [c * 1.21 + 14.4 for c in circuits]   # measured on ibm_fez, 4096 shots
    t_c = [0.0, 2.5, 2.5, 2.5]                  # classical part, seconds

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = np.arange(len(labels))

    ax1.bar(x, circuits, color="#4C72B0")
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("量子回路数")
    for xi, c in zip(x, circuits):
        ax1.text(xi, c * 2.0, ("3" if c < 10 else f"{c:.0e}"), ha="center", fontsize=8)
    ax1.set_ylim(1, 5e8)

    share = [tq / (tq + tc) * 100 for tq, tc in zip(t_q, t_c)]
    ax2.bar(x, t_q, color="#4C72B0", label="量子")
    ax2.bar(x, t_c, bottom=t_q, color="#DD8452", label="古典")
    ax2.set_yscale("log")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("壁時計時間 [s]（実測）")
    ax2.legend(fontsize=8, loc="upper right")
    for xi, tq, tc, sh in zip(x, t_q, t_c, share):
        ax2.text(xi, (tq + tc) * 1.6, f"量子{sh:.0f}%", ha="center", fontsize=7.5)
    ax2.set_ylim(5, 1e9)
    fig.tight_layout()
    save(fig, "fig_reduction_stages.pdf")


# ***********************************************
# 2. Two axes: surrogatability vs measurability (theory)
# ***********************************************
def fig_two_axes():
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    # background regions
    ax.axhspan(0, 0.5, xmin=0.0, xmax=0.5, color="#E8F0F8", zorder=0)
    ax.text(0.13, 0.40, "A 群\n古典で十分", ha="center", va="center", fontsize=9, color="#2C5C8A")
    ax.text(0.72, 0.92, "C 群：量子優位性の候補", ha="center", va="center", fontsize=9, color="#8A3B2C")
    ax.text(0.62, 0.13, "測定回数の壁\n（カーネル集中）", ha="center", va="center",
            fontsize=8, color="#666666")
    ax.text(0.20, 0.70, "B 群", ha="center", va="center", fontsize=9, color="#4A7A3A")

    pts = {
        "Z 特徴写像": (0.08, 0.10), "1 次元短時間発展": (0.20, 0.13),
        "浅い生成モデル": (0.34, 0.30),
        "ZZ 特徴写像": (0.72, 0.28), "projected kernel": (0.45, 0.62),
        "QAOA (p 小)": (0.35, 0.55),
        "強相関 2 次元": (0.80, 0.78), "Shor": (0.90, 0.85),
        "ランダム回路": (0.86, 0.60),
    }
    offs = {"Shor": (-30, 4), "強相関 2 次元": (-52, -12), "ランダム回路": (-56, 4),
            "ZZ 特徴写像": (-44, 6), "Z 特徴写像": (7, 0),
            "1 次元短時間発展": (7, -1), "浅い生成モデル": (-58, 3),
            "projected kernel": (-64, 4), "QAOA (p 小)": (-48, -12)}
    for name, (x, y) in pts.items():
        ax.plot(x, y, "o", ms=5, color="#333333")
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=offs.get(name, (5, 3)), fontsize=7.5)
    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=0.8)
    ax.set_xlabel("古典サロゲート化の困難さ →")
    ax.set_ylabel("必要な測定回数 →")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, "fig_two_axes.pdf")


# ***********************************************
# 3. Conversion rate of the eight measured cases (theory)
# ***********************************************
def fig_conversion_rate():
    names = ["permutation\n添字操作", "厳密再構成", "古典サロゲート", "shadows+ML",
             "NQS/MPS", "Rotosolve", "Nystrom", "適応ショット",
             "分離再構成\n(d=10)", "ブロック分解\n(d=8, 2対)", "PQK 写像\n(d=3)"]
    rate = [99.8, 99.98, 99.9, 98.0, 100.0, 85.0, 70.0, 67.0, 99.999, 99.98, 89.2]
    exact = [True, True, True, False, True, True, False, False, True, True, True]
    order = np.argsort(rate)[::-1]
    names = [names[i] for i in order]; rate = [rate[i] for i in order]
    exact = [exact[i] for i in order]
    colors = ["#4C72B0" if e else "#DD8452" for e in exact]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    y = np.arange(len(names))
    ax.barh(y, rate, color=colors)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("古典に変換した割合 [%]")
    ax.set_xlim(0, 112)
    for yi, r in zip(y, rate):
        ax.text(r + 1.5, yi, f"{r:.4g}", va="center", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", label="精度を保つ"),
                       Patch(color="#DD8452", label="近似を伴う")],
              fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_conversion_rate.pdf")


def fig_hw_measured():
    """Measured on ibm_fez, 190 pairs, 4096 shots (hw_benchmark)."""
    pts = [(1, 21.02), (5, 13.46), (20, 41.74), (60, 84.75), (190, 244.74)]
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    slope, intercept = np.polyfit(x, y, 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))

    xs = np.linspace(0, 200, 50)
    a1.plot(xs, slope * xs + intercept, "-", color="#4C72B0", lw=1.2,
            label=f"回帰 {intercept:.1f} + {slope:.2f}$n$ s")
    a1.plot(x, y, "o", ms=5, color="#C44E52", label="実測")
    a1.set_xlabel("1 ジョブ内の回路数", fontsize=8)
    a1.set_ylabel("壁時計時間 [s]", fontsize=8)
    a1.legend(fontsize=7, loc="upper left")
    a1.set_title("(a) ジョブの固定費と限界費用", fontsize=8.5)

    labs = ["素朴\n$10^7$", "添字\n操作化\n$2{\\times}10^4$",
            "+Nystrom\n$6{\\times}10^3$", "厳密\n再構成\n3"]
    circ = np.array([1e7, 2e4, 6e3, 3.0])
    t = circ * slope + intercept
    xb = np.arange(4)
    a2.bar(xb, t, color=["#DD8452", "#C0A080", "#8FA8C8", "#4C72B0"])
    a2.set_yscale("log")
    a2.set_xticks(xb); a2.set_xticklabels(labs, fontsize=6.5)
    a2.set_ylabel("壁時計時間 [s]", fontsize=8)
    for xi, ti in zip(xb, t):
        lab = (f"{ti/86400:.0f} 日" if ti > 86400 else
               f"{ti/3600:.1f} 時間" if ti > 3600 else f"{ti:.0f} s")
        a2.text(xi, ti * 1.7, lab, ha="center", fontsize=7)
    a2.set_ylim(3, 1e9)
    a2.set_title("(b) 実測値による所要時間", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_hw_measured.pdf")


def fig_structure_cost():
    """Circuit cost of the exact reconstruction as structure is exploited."""
    d = np.arange(1, 11)
    reps = 2
    m = 2 * reps + 1
    tensor = m ** d.astype(float)
    separable = 2 * reps * d + 1
    direct = np.full_like(d, 200 * 199 // 2, dtype=float)
    block2 = np.array([float(sum(m ** len(b) for b in
                                 ([[2 * i, 2 * i + 1] for i in range(dd // 2)]
                                  + [[dd - 1]] * (dd % 2))) - (((dd + 1) // 2) - 1))
                       for dd in d])

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(d, tensor, "o-", ms=4, label="完全テンソル $(2L{+}1)^d$", color="#DD8452")
    ax.plot(d, block2, "^-", ms=4, label="ブロック分解（2 特徴ずつ結合）", color="#55A868")
    ax.plot(d, separable, "s-", ms=4, label="分離 $2Ld{+}1$", color="#4C72B0")
    ax.plot(d, direct, "--", lw=1, color="#888888", label="直接評価 $n(n{-}1)/2$ ($n{=}200$)")
    ax.set_yscale("log"); ax.set_xlabel("特徴次元 $d$"); ax.set_ylabel("量子回路数")
    ax.set_title("$L=2$ での再構成コスト", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_structure_cost.pdf")


# ***********************************************
# 4. Classical-side speedup of the shadow sampler (theory)
# ***********************************************
def fig_speedup_v2():
    n = np.arange(4, 13)
    M = 16000
    # naive: per snapshot build a 2^n x 2^n unitary and multiply -> ~ M * 2^{3n}
    naive = M * (2.0 ** n) ** 3
    # tensor contraction of the rotation: ~ M * n * 2^{2n}
    tensor = M * n * (2.0 ** n) ** 2
    # factorised: factors of <= k qubits, ~ (#factors) * 3^k * 2^{2k}, k = 4
    k = 4
    fact = np.full_like(n, 1.0, dtype=float) * (n / 2) * 3 ** k * (2.0 ** k) ** 2

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.plot(n, naive, "o-", label="v1: スナップショットごとに $2^n$ 行列", color="#DD8452")
    ax.plot(n, tensor, "s-", label="テンソル縮約", color="#55A868")
    ax.plot(n, fact, "^-", label="v2: 因子分解 ($k\\leq4$)", color="#4C72B0")
    ax.set_yscale("log")
    ax.set_xlabel("全 qubit 数 $n$")
    ax.set_ylabel("演算量（相対，$M=16000$）")
    ax.axvline(9, color="#AAAAAA", ls=":", lw=0.8)
    ax.text(9.1, naive[-1] * 0.3, "本稿の\n$n=9$", fontsize=7.5, color="#666666")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_speedup_v2.pdf")


# ***********************************************
# 5. Layer diagram of the three conventions (implementation)
# ***********************************************
def fig_layers():
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    fc=fc, ec="#444444", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    box(0.3, 6.0, 9.4, 1.0, "リサンプリング  perm_test / bootstrap / k-fold\n"
                            "（添字のみを扱う： K[ix(idx, idx)]）", "#EAF1F8")
    box(0.3, 4.4, 9.4, 1.0, "統計量  hsic / mmd2 / kernel_ridge_residual\n"
                            "（Gram 行列のみを受け取る）", "#EAF1F8")
    box(0.3, 2.8, 9.4, 1.0, "GramCache（データ配列を受け取る唯一の層）\n"
                            "n_kernel_calls をカウント", "#FDECD9")
    box(0.3, 1.2, 4.4, 1.0, "量子カーネル\nBatchedFidelityKernel", "#F5E2E2")
    box(5.2, 1.2, 4.5, 1.0, "削減つきカーネル\n差分グリッド / Nystrom", "#F5E2E2")
    box(0.3, 0.0, 9.4, 0.85, "qiskit Sampler → 量子回路", "#E6E6E6", fs=8.5)

    for y0, y1 in ((6.0, 5.4), (4.4, 3.8), (2.8, 2.2)):
        ax.add_patch(FancyArrowPatch((5.0, y0), (5.0, y1), arrowstyle="-|>",
                                     mutation_scale=11, color="#555555", lw=1.0))
    ax.add_patch(FancyArrowPatch((2.5, 1.2), (2.5, 0.85), arrowstyle="-|>",
                                 mutation_scale=11, color="#555555", lw=1.0))
    ax.add_patch(FancyArrowPatch((7.4, 1.2), (7.4, 0.85), arrowstyle="-|>",
                                 mutation_scale=11, color="#555555", lw=1.0))
    ax.text(9.85, 5.3, "回路数に\n影響しない", fontsize=7.5, color="#2C5C8A",
            ha="right", va="center")
    ax.text(9.85, 1.7, "回路数を\n決める", fontsize=7.5, color="#8A3B2C",
            ha="right", va="center")
    fig.tight_layout()
    save(fig, "fig_layers.pdf")


# ***********************************************
# 6. Tests by module and type (implementation)
# ***********************************************
def fig_test_types():
    mods = ["gram_resampling", "quantum_gram_kernels", "thrifty_exact",
            "thrifty_budget", "thrifty_psd", "thrifty_scaling", "thrifty_primal",
            "thrifty_structure", "thrifty_downstream",
            "surrogate", "shadows", "nqs", "vqe_surrogate_opt", "cdr_mitigation",
            "l2_mediator", "l2_qpc_cmi", "l2_orientation", "l3_witness"]
    # (equivalence, budget, property, statistical)
    counts = np.array([
        [5, 2, 3, 3], [5, 3, 3, 1],
        [6, 4, 6, 3], [5, 2, 8, 3], [4, 1, 9, 3], [6, 5, 5, 2],
        [5, 2, 6, 2], [6, 3, 7, 2], [5, 4, 7, 2],
        [3, 1, 2, 2], [1, 1, 3, 2], [2, 0, 4, 3], [2, 2, 3, 2], [2, 1, 4, 3],
        [2, 0, 4, 2], [2, 0, 5, 2], [1, 0, 5, 2], [2, 0, 4, 2],
    ])
    labels = ["同値性", "予算", "性質", "統計"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    y = np.arange(len(mods))
    left = np.zeros(len(mods))
    for k in range(4):
        ax.barh(y, counts[:, k], left=left, color=colors[k], label=labels[k])
        left += counts[:, k]
    ax.set_yticks(y); ax.set_yticklabels(mods, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("テスト数")
    ax.legend(fontsize=7.5, ncol=4, loc="lower right", frameon=False)
    ax.set_xlim(0, 26)
    fig.tight_layout()
    save(fig, "fig_test_types.pdf")


# ======================================================================
# 7-9. Compact figures for the one-page extended abstract
# ======================================================================
def fig_abs_speedup():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.1))
    lab = ["素朴", "添字\n操作化", "+Nystrom", "厳密\n再構成"]
    circ = [1e7, 2e4, 6e3, 3]
    x = np.arange(4)
    a1.bar(x, circ, color="#4C72B0"); a1.set_yscale("log")
    a1.set_xticks(x); a1.set_xticklabels(lab, fontsize=7)
    a1.set_ylabel("量子回路数", fontsize=8); a1.set_ylim(1, 5e8)
    for xi, c in zip(x, circ):
        a1.text(xi, c * 2.2, ("3" if c < 10 else f"{c:.0e}"), ha="center", fontsize=6.5)
    a1.set_title("(a) 回路数の削減（$n=200$）", fontsize=8)

    n = np.arange(4, 13); M = 16000
    a2.plot(n, M * (2.0 ** n) ** 3, "o-", ms=3, label="素朴", color="#DD8452")
    a2.plot(n, M * n * (2.0 ** n) ** 2, "s-", ms=3, label="テンソル縮約", color="#55A868")
    a2.plot(n, (n / 2) * 3 ** 4 * (2.0 ** 4) ** 2, "^-", ms=3, label="因子分解", color="#4C72B0")
    a2.set_yscale("log"); a2.set_xlabel("qubit 数 $n$", fontsize=8)
    a2.set_ylabel("演算量（相対）", fontsize=8); a2.legend(fontsize=6.5, loc="upper left")
    a2.set_title("(b) 古典側：shadow 標本化", fontsize=8)
    fig.tight_layout(); save(fig, "fig_abs_speedup.pdf")


def fig_abs_causal():
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.6, 2.2))
    sc = ["S0\n観測のみ", "S1\n+準備\nランダム化", "S2\nΛ古典", "S3\n+N_Y既知", "S4\nZ制御\n頂点"]
    lo = [0.045, 0.299, 0.152, 0.265, 0.299]
    hi = [0.869, 0.299, 0.927, 0.678, 0.299]
    truth = [0.299, 0.299, 0.566, 0.299, 0.299]
    y = np.arange(5)
    for yi, (l, h) in enumerate(zip(lo, hi)):
        b1.plot([l, h], [yi, yi], lw=6, color="#4C72B0", solid_capstyle="butt", alpha=.75)
    b1.plot(truth, y, "D", color="#C44E52", ms=5, label="真値")
    b1.set_yticks(y); b1.set_yticklabels(sc, fontsize=6.5); b1.invert_yaxis()
    b1.set_xlabel("$P(y{=}0 \\mid \\mathrm{do}(X{=}\\tau))$ の区間", fontsize=8)
    b1.legend(fontsize=7, loc="lower right"); b1.set_xlim(0, 1)
    b1.set_title("(a) 量子媒介の部分識別", fontsize=8)

    shots = [200, 1000, 5000]; cor = [22, 51, 71]; und = [70, 41, 21]; wr = [1, 1, 1]
    xb = np.arange(3); w = .6
    b2.bar(xb, cor, w, color="#4C72B0", label="正解")
    b2.bar(xb, und, w, bottom=cor, color="#CCCCCC", label="未決定")
    b2.bar(xb, wr, w, bottom=np.array(cor) + np.array(und), color="#C44E52", label="誤向き")
    b2.set_xticks(xb); b2.set_xticklabels([str(s) for s in shots], fontsize=7.5)
    b2.set_xlabel("shots（基底対あたり）", fontsize=8)
    b2.set_ylabel("辺数（全 93）", fontsize=8)
    b2.legend(fontsize=7, loc="lower right"); b2.set_ylim(0, 110)
    b2.set_title("(b) 観測のみによる向き付け", fontsize=8)
    fig.tight_layout(); save(fig, "fig_abs_causal.pdf")


def fig_abs_design():
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.set_xlim(-0.3, 13.8); ax.set_ylim(-0.25, 5.35); ax.axis("off")

    def box(x, y, w, h, t, fc, fs=7.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    fc=fc, ec="#444444", lw=.7))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)

    ax.text(3.1, 4.95, "thrifty-qiskit", fontsize=8.5, ha="center",
            color="#2C5C8A", weight="bold")
    ax.text(10.2, 4.95, "causal-qiskit", fontsize=8.5, ha="center",
            color="#8A3B2C", weight="bold")
    ax.plot([6.55, 6.55], [1.35, 4.75], ls=":", color="#999999", lw=.8)

    # left column
    box(0.2, 3.55, 5.8, 0.95, "リサンプリング  perm / bootstrap / k-fold\n（添字のみ： K[ix(idx,idx)]）", "#EAF1F8")
    box(0.2, 2.45, 5.8, 0.95, "統計量  HSIC / MMD / KCIT 残差\n（Gram 行列のみ）", "#EAF1F8")
    box(0.2, 1.35, 5.8, 0.95, "量子カーネル\n（バッチ化 / 差分グリッド / Nystrom）", "#F5E2E2", fs=7.0)
    # right column
    box(7.0, 3.55, 6.3, 0.95, "L1：量子カーネル因果発見\n（サロゲート・回路数削減）", "#F3E8F0")
    box(7.0, 2.45, 6.3, 0.95, "L2：量子因果モデル\n部分識別 SDP / 骨格 / 向き付け", "#F3E8F0")
    box(7.0, 1.35, 6.3, 0.95, "L3：不定因果順序\n因果ロバストネス・witness SDP", "#F3E8F0")

    for x in (3.1, 10.15):
        for y0, y1 in ((3.55, 3.40), (2.45, 2.30), (1.35, 1.20)):
            ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                         mutation_scale=8, color="#555555", lw=.9))

    # shared bottom layer
    box(0.2, 0.05, 13.1, 1.00,
        "EvalCache：データ・デバイスに触れる唯一の層（呼び出し回数を計数）\n"
        "GramCache（Gram 行列）  ｜  縮約状態のエントロピー  ｜  シミュレータ実行",
        "#FDECD9", fs=7.2)
    save(fig, "fig_abs_design.pdf")


if __name__ == "__main__":
    fig_reduction_stages()
    fig_two_axes()
    fig_conversion_rate()
    fig_structure_cost()
    fig_hw_measured()
    fig_speedup_v2()
    fig_layers()
    fig_test_types()
    fig_abs_speedup()
    fig_abs_causal()
    fig_abs_design()
    print("\nall figures written to", OUT)
