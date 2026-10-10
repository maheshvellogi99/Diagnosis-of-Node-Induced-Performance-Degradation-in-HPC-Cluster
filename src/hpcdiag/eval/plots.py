"""Figures for Layer-1 runtime evaluation (static PNG for reports)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

# Reference palette (dataviz skill, references/palette.md): sequential blue 100 -> 700,
# categorical slot 1, light surface and text inks.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES_1 = "#2a78d6"
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def pred_vs_actual_by_bin(y: np.ndarray, yhat: np.ndarray, table: pd.DataFrame,
                          edges_h: list[float], title: str, out: Path) -> Path:
    """Left: hexbin of predicted vs actual (log-log, hours; both clipped to >= 1 s for display),
    actual-runtime bin edges marked. Right: median r = log(max(yhat,1)/y) per bin (y > 30 s)."""
    yc = np.maximum(np.asarray(y, float), 1.0) / 3600
    pc = np.maximum(np.asarray(yhat, float), 1.0) / 3600
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)

    hb = a1.hexbin(np.log10(yc), np.log10(pc), gridsize=70, cmap=cmap, norm=LogNorm(),
                   mincnt=1, linewidths=0)
    lo = min(np.log10(yc).min(), np.log10(pc).min())
    hi = max(np.log10(yc).max(), np.log10(pc).max())
    a1.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls="--")
    a1.text(hi, hi, " y = x", color=INK_2, fontsize=8, va="bottom", ha="right")
    for e in edges_h[1:]:
        a1.axvline(np.log10(e), color=GRID, lw=1)
    ticks = [-3, -2, -1, 0, 1, 2]
    lab = ["3.6 s", "36 s", "6 min", "1 h", "10 h", "100 h"]
    a1.set_xticks(ticks, lab)
    a1.set_yticks(ticks, lab)
    a1.set_xlabel("actual run time (log scale); vertical lines = bin edges 1 h, 6 h, 1 d, 2 d")
    a1.set_ylabel("predicted run time (log scale)")
    _style(a1)
    cb = fig.colorbar(hb, ax=a1, pad=0.01)
    cb.set_label("test jobs per cell", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    cb.outline.set_visible(False)

    t = table.reset_index(drop=True)
    ypos = np.arange(len(t))[::-1]
    a2.barh(ypos, t["median_r"], color=SERIES_1, height=0.55)
    a2.axvline(0, color=INK_2, lw=1)
    for yp, v, n in zip(ypos, t["median_r"], t["n_r"], strict=True):
        a2.text(v, yp, f" {v:+.2f} " if np.isfinite(v) else " n/a",
                va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=9)
        a2.text(1.0, yp, f"n={n:,}", transform=a2.get_yaxis_transform(), va="center",
                ha="left", color=INK_2, fontsize=8)
    a2.set_yticks(ypos, t["bin"])
    a2.set_xlabel("median log(max(pred,1 s) / actual), jobs > 30 s  (< 0 = underestimate)")
    span = np.nanmax(np.abs(t["median_r"])) if np.isfinite(t["median_r"]).any() else 1
    a2.set_xlim(-1.35 * span, 1.35 * span)
    _style(a2)

    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def rmse_over_time(split_t: pd.Series, rmse_h: pd.Series, band: tuple[float, float], ref: float,
                   title: str, out: Path) -> Path:
    """Per-split RMSE (hours) against split date, with the Gate 1 band shaded."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor(SURFACE)
    ax.axhspan(band[0], band[1], color=GRID, alpha=0.45, lw=0)
    ax.axhline(ref, color=INK_2, lw=1, ls="--")
    ax.text(0.995, 0.97, f"dashed = reference {ref:g} h; shaded = Gate 1 band "
            f"[{band[0]:.3f}, {band[1]:.3f}] h", transform=ax.transAxes, color=INK_2,
            fontsize=8, va="top", ha="right")
    ax.plot(split_t, rmse_h, color=SERIES_1, lw=1.5, marker="o", ms=3)
    ax.set_ylabel("RMSE per split (h)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
    _style(ax)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def pvalue_hists(p: np.ndarray, users: np.ndarray, k: int, title: str, out: Path) -> Path:
    """Histogram of p-values (density, 20 bins): pooled + the k users with most jobs.
    Uniform p-values give bars at height 1 (dashed line)."""
    p = np.asarray(p, float)
    u = pd.Series(np.asarray(users, dtype=object))
    top = u.value_counts().head(k)
    panels = [("all jobs", np.ones(len(p), bool))] + [
        (f"user {name}", (u == name).to_numpy()) for name in top.index]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.1 * nrow), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    bins = np.linspace(0, 1, 21)
    for ax, (name, m) in zip(axes.ravel(), panels, strict=False):
        ax.hist(p[m], bins=bins, density=True, color=SERIES_1, edgecolor=SURFACE, linewidth=1)
        ax.axhline(1.0, color=INK_2, lw=1, ls="--")
        ax.set_title(f"{name} (n={int(m.sum()):,})", color=INK, fontsize=9, loc="left")
        ax.set_xlim(0, 1)
        _style(ax)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("p-value")
    for ax in axes[:, 0]:
        ax.set_ylabel("density (uniform = 1)")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def pvalue_panels(panels: list[tuple[str, np.ndarray]], title: str, out: Path) -> Path:
    """One density histogram (20 bins) per (name, p-values) panel; uniform = dashed line at 1."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    bins = np.linspace(0, 1, 21)
    for ax, (name, p) in zip(axes[0], panels, strict=True):
        p = np.asarray(p, float)
        p = p[np.isfinite(p)]
        ax.hist(p, bins=bins, density=True, color=SERIES_1, edgecolor=SURFACE, linewidth=1)
        ax.axhline(1.0, color=INK_2, lw=1, ls="--")
        ax.set_title(f"{name} (n={len(p):,})", color=INK, fontsize=9, loc="left")
        ax.set_xlim(0, 1)
        ax.set_xlabel("p-value")
        _style(ax)
    axes[0][0].set_ylabel("density (uniform = 1)")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out
