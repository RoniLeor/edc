"""Render publication-style vector figures from the bundled benchmark measurements."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
output: Path = Path("assets")
output.mkdir(exist_ok=True)
colors: dict[str, str] = {"alm": "#2878A5", "gdi": "#73958A", "elastic": "#C16438"}
labels: dict[str, str] = {"alm": "PC-ALM", "gdi": "Ordinary GDI", "elastic": "Elastic GDI"}
manifest: dict = json.loads(Path("benchmarks/manifest.json").read_text())
runs: list[dict] = [json.loads(Path(item["path"]).read_text()) for item in manifest["runs"]]
fig: Any
axes: Any
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
epochs: int
dataset: str
row: int
col: int
method: str
for row, epochs in enumerate((5, 10)):
    for col, dataset in enumerate(("mnist", "fashion")):
        ax: Any = axes[row, col]
        for method in ("alm", "elastic"):
            values: list[list[float]] = []
            steps: int
            for steps in (16, 32, 64):
                selected: list[float] = [
                    100 * r["test"]["accuracy"]
                    for r in runs
                    if r["config"]["dataset"] == dataset
                    and r["config"]["epochs"] == epochs
                    and r["config"]["steps"] == steps
                    and ("elastic" if r["config"].get("elastic", 0) > 0 else r["config"]["method"])
                    == method
                ]
                assert len(selected) == 3
                values.append(selected)
            data: Any = np.asarray(values)
            ax.errorbar(
                x=range(3),
                y=data.mean(axis=1),
                yerr=data.std(axis=1, ddof=1),
                color=colors[method],
                label=labels[method],
                marker="o",
                capsize=3,
            )
        ax.set_title(
            f"{'MNIST' if dataset == 'mnist' else 'Fashion-MNIST'} · {epochs} epochs", loc="left"
        )
        ax.set_xticks(ticks=range(3), labels=["16", "32", "64"])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(65, 100)
        ax.grid(axis="y", alpha=0.16)
        if row == 1:
            ax.set_xlabel("Settling steps per batch")
        if row == 0 and col == 0:
            ax.legend(frameon=False, loc="lower right")
fig.suptitle(
    "Learning with fewer settling steps", fontsize=17, fontweight="bold", x=0.07, ha="left"
)
fig.text(
    0.07,
    0.015,
    "Mean ± sample SD, three seeds. Same architecture, MSE, split and optimizer. "
    "Checkpoints selected by validation.",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
for extension in ("png", "svg", "pdf"):
    fig.savefig(output / f"accuracy.{extension}", dpi=200)
plt.close(fig)

alignment: dict = json.loads(Path("benchmarks/gradients.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, dataset in zip(axes, ("mnist", "fashion"), strict=True):
    for method in ("alm", "gdi", "elastic"):
        values = [
            [
                m["global_cosine"]
                for r in alignment["rows"]
                if r["dataset"] == dataset
                for m in r["measurements"]
                if m["method"] == method and m["steps"] == steps
            ]
            for steps in (16, 32, 64)
        ]
        data = np.asarray(values)
        assert data.shape == (3, 3)
        ax.errorbar(
            x=range(3),
            y=data.mean(axis=1),
            yerr=data.std(axis=1, ddof=1),
            color=colors[method],
            label=labels[method],
            marker="o",
            capsize=3,
        )
    ax.axhline(y=1, color="#A4A7AC", linestyle=":", linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(ticks=range(3), labels=["16", "32", "64"])
    ax.set_title("MNIST" if dataset == "mnist" else "Fashion-MNIST", loc="left")
    ax.set_xlabel("Settling steps")
    ax.grid(axis="y", alpha=0.16)
axes[0].set_ylabel("Full weight-gradient cosine to BP")
axes[1].legend(frameon=False, loc="lower right")
fig.suptitle(
    "Gradient alignment at identical weights", fontsize=17, fontweight="bold", x=0.07, ha="left"
)
fig.text(
    0.07,
    0.015,
    "Three fixed BP checkpoints per dataset; four disjoint training query batches each. "
    "Error bars: sample SD across checkpoints.",
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.9))
for extension in ("png", "svg", "pdf"):
    fig.savefig(output / f"gradients.{extension}", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, dataset in zip(axes, ("mnist", "fashion"), strict=True):
    for method in ("alm", "gdi", "elastic"):
        data = np.asarray(
            [
                m["layer_cosines"]
                for r in alignment["rows"]
                if r["dataset"] == dataset
                for m in r["measurements"]
                if m["method"] == method and m["steps"] == 16
            ]
        )
        mean: Any = data.mean(axis=0)
        sd: Any = data.std(axis=0, ddof=1)
        ax.plot(range(1, 33), mean, color=colors[method], label=labels[method])
        ax.fill_between(range(1, 33), mean - sd, mean + sd, color=colors[method], alpha=0.12)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(1, 32)
    ax.set_xlabel("Weight layer (32 = readout)")
    ax.set_title("MNIST" if dataset == "mnist" else "Fashion-MNIST", loc="left")
    ax.grid(axis="y", alpha=0.16)
axes[0].set_ylabel("Per-layer gradient cosine to BP")
axes[1].legend(frameon=False, loc="lower right")
fig.suptitle(
    "Where credit reaches · 16 settling steps", fontsize=17, fontweight="bold", x=0.07, ha="left"
)
fig.text(
    0.07,
    0.015,
    "Same frozen checkpoints and queries as the full-gradient comparison. "
    "Shading: sample SD across three checkpoints.",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.9))
for extension in ("png", "svg", "pdf"):
    fig.savefig(output / f"layers.{extension}", dpi=200)
plt.close(fig)

fig = plt.figure(figsize=(12, 5.5), facecolor="white")
ax = fig.add_axes((0.035, 0.27, 0.71, 0.58))
ax.set_xlim(0, 11)
ax.set_ylim(-0.3, 3.7)
ax.axis("off")
fig.text(0.035, 0.94, "Elastic Dual Coupling", fontsize=20, fontweight="bold")
fig.text(0.035, 0.885, "(a) Separate multipliers, softly connected across depth", fontsize=11)
x: float
name: str
left: float
right: float
rad: float
extension: str
label: str
offset: float
positions: list[float] = [1.2, 2.5, 5.2, 7.9, 9.2]
names: list[str] = ["1", "2", "16", "30", "31"]
for x, name in zip(positions, names, strict=True):
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.39, 2.43),
            0.78,
            0.54,
            boxstyle="round,pad=0.06",
            facecolor="#EDF3F6",
            edgecolor="#2878A5",
            linewidth=1.2,
        )
    )
    ax.text(x, 2.70, rf"$z_{{{name}}}$", ha="center", va="center", fontsize=13)
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.36, 1.43),
            0.72,
            0.48,
            boxstyle="round,pad=0.06",
            facecolor="#FBF0E9",
            edgecolor="#C16438",
            linewidth=1.2,
        )
    )
    ax.text(x, 1.67, rf"$\lambda_{{{name}}}$", ha="center", va="center", fontsize=12)
    ax.add_patch(
        FancyArrowPatch((x, 1.96), (x, 2.36), arrowstyle="<->", mutation_scale=9, color="#656D76")
    )
for left, right in zip([0.2] + positions, positions + [10.4], strict=True):
    start: float = left + 0.47 if left > 0.2 else left
    end: float = right - 0.47 if right < 10 else right
    ax.add_patch(
        FancyArrowPatch(
            (start, 2.7), (end, 2.7), arrowstyle="-|>", mutation_scale=10, color="#2878A5"
        )
    )
ax.text(0.05, 2.98, r"$x$", fontsize=13)
ax.text(10.15, 2.98, r"$\hat y$", fontsize=13)
for x in (3.85, 6.55):
    ax.text(x, 3.1, r"$\cdots$", ha="center", fontsize=18)
for left, right, rad in ((1.2, 9.2, 0.34), (2.5, 7.9, 0.25)):
    ax.add_patch(
        FancyArrowPatch(
            (left, 1.36),
            (right, 1.36),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="<->",
            mutation_scale=12,
            color="#C16438",
            linewidth=1.6,
        )
    )
ax.text(5.2, 1.0, "center unchanged", ha="center", fontsize=9, color="#656D76")
ax.text(
    5.2,
    -0.18,
    r"Symmetric pairs: $1\leftrightarrow31$, $2\leftrightarrow30$, $\ldots$",
    ha="center",
    fontsize=10,
)
ax.text(
    5.2,
    3.45,
    "Forward initialization → local activity / dual settling → weight update",
    ha="center",
    fontsize=9,
    color="#48515B",
)
ax = fig.add_axes((0.80, 0.41, 0.17, 0.35))
t: Any = np.arange(16)
strength: Any = 0.05 * np.maximum(1 - t / 7, 0)
ax.plot(t, strength, color="#C16438", linewidth=2)
ax.fill_between(t, 0, strength, color="#C16438", alpha=0.12)
ax.axvline(7, color="#969DA5", linestyle=":", linewidth=1)
ax.set_xticks(ticks=[0, 7, 15])
ax.set_yticks(ticks=[0, 0.025, 0.05])
ax.set_xlabel("Settling iteration $t$")
ax.set_ylabel(r"$\kappa_t$")
ax.set_ylim(-0.003, 0.055)
fig.text(0.78, 0.885, "(b) Fade, then refine independently", fontsize=10)
fig.text(
    0.80, 0.27, "Example: T = 16\nZero coupling from t = 7 onward", fontsize=9, color="#48515B"
)
fig.text(
    0.50,
    0.18,
    r"$\lambda_i^{t+1}=\lambda_i^t+\alpha\,r_i(z^{t+1})-\kappa_t(\lambda_i^t-\lambda_{H+1-i}^t)$",
    ha="center",
    fontsize=17,
)
fig.text(
    0.50,
    0.10,
    r"$\kappa_t=0.05\,\max(1-t/(\lfloor T/2\rfloor-1),0),\quad \alpha=1,\ H=31$",
    ha="center",
    fontsize=12,
)
fig.text(
    0.50,
    0.025,
    "GDI retrieves the initial multipliers from training memory. "
    "Both partners read old duals; no extra gradient evaluations.",
    ha="center",
    fontsize=9,
    color="#48515B",
)
for extension in ("png", "svg", "pdf"):
    fig.savefig(output / f"architecture.{extension}", dpi=220)
plt.close(fig)
print("Saved four figures as PNG, SVG and PDF.")

# Matplotlib emits trailing spaces in SVG path data; normalize generated text.
svg: Path
for svg in output.glob("*.svg"):
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
