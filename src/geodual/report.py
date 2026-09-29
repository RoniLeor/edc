"""Transparent aggregate tables and scientific plots for completed matched runs."""

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass(frozen=True)
class Trial:
    """Reviewable scalar outcomes extracted from one immutable run summary."""

    dataset: str
    method: str
    steps: int
    seed: int
    accuracy: float
    train_accuracy: float
    seconds: float
    first_cosine: float
    best_epoch: int


def read_trials(root: Path) -> list[Trial]:
    trials: list[Trial] = []
    seen: set[tuple[str, str, int, int]] = set()
    path: Path
    for path in sorted(root.glob("*/summary.json")):
        raw: dict[str, Any] = json.loads(path.read_text())
        config: dict[str, Any] = raw["config"]
        key: tuple[str, str, int, int] = (
            config["dataset"],
            config["method"],
            config["steps"],
            config["seed"],
        )
        if key in seen:
            raise ValueError(f"Duplicate run: {key}")
        seen.add(key)
        trials.append(
            Trial(
                dataset=key[0],
                method=key[1],
                steps=key[2],
                seed=key[3],
                accuracy=100 * raw["test"]["accuracy"],
                train_accuracy=100 * raw["train"]["accuracy"],
                seconds=raw["training_seconds_including_compile"],
                first_cosine=raw["diagnostics"]["layer_cosines"][0],
                best_epoch=raw["best_epoch"],
            )
        )
    if not trials:
        raise ValueError("No completed trials found")
    return trials


@dataclass(frozen=True)
class Report:
    """Descriptive seed statistics and paired deltas without cherry-picking."""

    trials: tuple[Trial, ...]

    def __call__(self, *, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        groups: dict[tuple[str, str, int], list[Trial]] = defaultdict(list)
        trial: Trial
        for trial in self.trials:
            groups[(trial.dataset, trial.method, trial.steps)].append(trial)
        stream: TextIO
        with (output / "trials.csv").open("w", newline="") as stream:
            writer: csv.DictWriter = csv.DictWriter(stream, fieldnames=list(asdict(self.trials[0])))
            writer.writeheader()
            writer.writerows(asdict(trial) for trial in self.trials)
        lines: list[str] = [
            "# Geometric Dual Initialization: measured results",
            "",
            "Accuracy is held-out evaluation accuracy after tuning-set checkpoint selection. "
            "Values are mean ± sample standard deviation across seeds; deltas are in pp. "
            "With one seed, the displayed zero SD does not estimate uncertainty.",
            "",
            "| Dataset | Method | T | Seeds | Test accuracy (%) | Train−test (pp) | "
            "Training seconds | First-layer cosine |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        key: tuple[str, str, int]
        runs: list[Trial]
        for key, runs in sorted(groups.items()):
            accuracies: list[float] = [trial.accuracy for trial in runs]
            deviation: float = stdev(accuracies) if len(runs) > 1 else 0
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2] if key[1] != 'bp' else '—'} | {len(runs)} | "
                f"{mean(accuracies):.2f} ± {deviation:.2f} | "
                f"{mean(trial.train_accuracy - trial.accuracy for trial in runs):.2f} | "
                f"{mean(trial.seconds for trial in runs):.2f} | "
                f"{mean(trial.first_cosine for trial in runs):.3f} |"
            )
        lines.extend(
            [
                "",
                "## Paired GDI minus zero-initialized PC-ALM",
                "",
                "| Dataset | T | Mean delta (pp) | Per-seed deltas (pp) |",
                "|---|---:|---:|---|",
            ]
        )
        for key, runs in sorted(groups.items()):
            if key[1] != "gdi":
                continue
            baseline: dict[int, float] = {
                trial.seed: trial.accuracy for trial in groups.get((key[0], "alm", key[2]), [])
            }
            if set(baseline) != {trial.seed for trial in runs}:
                raise ValueError(f"Unpaired GDI baseline: {key}")
            deltas: list[float] = [trial.accuracy - baseline[trial.seed] for trial in runs]
            lines.append(
                f"| {key[0]} | {key[2]} | {mean(deltas):+.2f} | "
                + ", ".join(f"{delta:+.2f}" for delta in deltas)
                + " |"
            )
        lines.extend(
            [
                "",
                "## Scope and cost",
                "",
                "See each run configuration and its experiment protocol for architecture, splits, "
                "and training budget. These finite-budget experiments do not establish a general "
                "generalization advantage. For ImageNet, held-out evaluation uses the official "
                "labeled validation split; tuning uses separate training-source images.",
                "",
                "Training times include JIT compilation, retrieval, all five epochs, and device "
                "synchronization. Diagnostics and evaluation are outside training time. "
                "Timings are from a shared local machine and can vary with hardware contention.",
                "",
                "Each summary records allocated bank bytes, including stored multiplier fields. "
                "Baselines carry an empty bank in the shared state structure. This measures "
                "persistent array storage, not peak process RAM.",
                "",
                "Mismatched candidates come from different labels; their score distribution is not "
                "norm-matched. Improved low-budget training does not itself prove better eventual "
                "generalization or faster training than backpropagation.",
                "",
                "Full configuration, epoch validation history, gradient cosine per layer, residual "
                "and stationarity norms, and checkpoints are in each run directory.",
            ]
        )
        (output / "results.md").write_text("\n".join(lines) + "\n")
        self.plot(groups=groups, path=output / "comparison.png")

    def compare(
        self,
        *,
        reference: tuple[Trial, ...],
        output: Path,
        current_name: str = "CE",
        reference_name: str = "MSE",
    ) -> None:
        """Pair experiment outcomes by dataset, method, budget, and seed."""
        previous: dict[tuple[str, str, int, int], Trial] = {
            (trial.dataset, trial.method, trial.steps, trial.seed): trial for trial in reference
        }
        groups: dict[tuple[str, str, int], list[tuple[Trial, Trial]]] = defaultdict(list)
        trial: Trial
        for trial in self.trials:
            identity: tuple[str, str, int, int] = (
                trial.dataset,
                trial.method,
                trial.steps,
                trial.seed,
            )
            if identity not in previous:
                raise ValueError(f"Missing squared-error reference: {identity}")
            groups[identity[:3]].append((trial, previous[identity]))
        lines: list[str] = [
            f"# {current_name} versus {reference_name}",
            "",
            "Official test accuracy after tuning-only checkpoint selection. Mean ± sample SD "
            "across matched seeds; differences are percentage points. A single seed has no "
            "estimable seed variability. Hyperparameters were held fixed, not separately tuned.",
            "",
            f"| Dataset | Method | T | Seeds | {reference_name} (%) | {current_name} (%) | "
            f"{current_name} − {reference_name} (pp) | "
            "Paired seed differences (pp) |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        key: tuple[str, str, int]
        pairs: list[tuple[Trial, Trial]]
        for key, pairs in sorted(groups.items()):
            current: list[float] = [pair[0].accuracy for pair in pairs]
            baseline: list[float] = [pair[1].accuracy for pair in pairs]
            differences: list[float] = [pair[0].accuracy - pair[1].accuracy for pair in pairs]
            current_sd: float = stdev(current) if len(current) > 1 else 0
            baseline_sd: float = stdev(baseline) if len(baseline) > 1 else 0
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2] if key[1] != 'bp' else '—'} | {len(pairs)} | "
                f"{mean(baseline):.2f} ± {baseline_sd:.2f} | "
                f"{mean(current):.2f} ± {current_sd:.2f} | {mean(differences):+.2f} | "
                + ", ".join(f"{delta:+.2f}" for delta in differences)
                + " |"
            )
        lines.extend(
            [
                "",
                "See results.md for paired GDI versus PC-ALM differences within CE. "
                "These are fixed-budget, fixed-hyperparameter comparisons; they do not establish "
                "asymptotic superiority or distribution-shift robustness. Concurrent ImageNet work "
                "affects runtime; historical timing differences are not clean speed comparisons.",
            ]
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "comparison.md").write_text("\n".join(lines) + "\n")
        datasets: list[str] = sorted({trial.dataset for trial in self.trials})
        figure: Figure
        axes: object
        figure, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(6 * len(datasets), 4.5))
        dataset: str
        axis: Axes
        method: str
        color: str
        for dataset, axis in zip(datasets, figure.axes, strict=True):
            for method, color in (("alm", "#2878b5"), ("gdi", "#d45c25")):
                budgets: list[int] = sorted(
                    key[2] for key in groups if key[:2] == (dataset, method)
                )
                loss_index: int
                label: str
                style: str
                for loss_index, label, style in ((0, current_name, "-"), (1, reference_name, "--")):
                    values: list[list[float]] = [
                        [pair[loss_index].accuracy for pair in groups[(dataset, method, budget)]]
                        for budget in budgets
                    ]
                    if values:
                        axis.errorbar(
                            x=budgets,
                            y=[mean(value) for value in values],
                            yerr=[stdev(value) if len(value) > 1 else 0 for value in values],
                            marker="o",
                            linestyle=style,
                            color=color,
                            capsize=3,
                            label=f"{method.upper()} {label}",
                        )
            axis.set(title=dataset.upper(), xlabel="Settling steps", ylabel="Test accuracy (%)")
            axis.set_xticks(
                sorted({key[2] for key in groups if key[0] == dataset and key[1] != "bp"})
            )
            axis.grid(alpha=0.2)
            axis.legend()
        figure.suptitle(f"{current_name} vs {reference_name} · matched seeds, mean ± sample SD")
        figure.tight_layout()
        figure.savefig(output / "loss.png", dpi=180)
        plt.close(figure)

    def plot(self, *, groups: dict[tuple[str, str, int], list[Trial]], path: Path) -> None:
        datasets: list[str] = sorted({key[0] for key in groups})
        figure: Figure
        axes: object
        figure, axes = plt.subplots(
            nrows=1, ncols=len(datasets), figsize=(6 * len(datasets), 4.2), squeeze=False
        )
        index: int
        dataset: str
        method: str
        color: str
        for index, dataset in enumerate(datasets):
            axis: Axes = figure.axes[index]
            for method, color in (("alm", "#2878b5"), ("gdi", "#d45c25"), ("mismatch", "#858585")):
                budgets: list[int] = sorted(
                    key[2] for key in groups if key[0] == dataset and key[1] == method
                )
                if not budgets:
                    continue
                values: list[list[float]] = [
                    [trial.accuracy for trial in groups[(dataset, method, budget)]]
                    for budget in budgets
                ]
                axis.errorbar(
                    x=budgets,
                    y=[mean(value) for value in values],
                    yerr=[stdev(value) if len(value) > 1 else 0 for value in values],
                    marker="o",
                    capsize=4,
                    label=method.upper(),
                    color=color,
                )
            bp: list[float] = [
                trial.accuracy
                for trial in self.trials
                if trial.dataset == dataset and trial.method == "bp"
            ]
            if bp:
                axis.axhline(y=mean(bp), color="#237a45", linestyle="--", label="Backprop")
            axis.set(
                title=dataset.upper(), xlabel="Settling iterations", ylabel="Test accuracy (%)"
            )
            ticks: list[int] = sorted({key[2] for key in groups if key[0] == dataset})
            axis.set_xticks(ticks)
            axis.grid(alpha=0.2)
            axis.legend()
        figure.suptitle("Geometric Dual Initialization · mean ± seed SD")
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/main"))
    parser.add_argument("--output", type=Path, default=Path("results/report"))
    parser.add_argument("--reference", type=Path, help="Saved squared-error runs for CE comparison")
    args: argparse.Namespace = parser.parse_args(argv)
    report: Report = Report(trials=tuple(read_trials(args.runs)))
    report(output=args.output)
    if args.reference is not None:
        report.compare(reference=tuple(read_trials(args.reference)), output=args.output)
