"""Report aggregation exercises persisted production summaries without training."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from geodual.report import Report, Trial, main, read_trials


@pytest.mark.parametrize("dataset", ["mnist", "imagenet"])
def test_report_and_cli(tmp_path: Path, dataset: str) -> None:
    root: Path = tmp_path / "runs"
    method: str
    seed: int
    for method in ("bp", "alm", "gdi", "mismatch"):
        for seed in range(2):
            directory: Path = root / f"{method}-{seed}"
            directory.mkdir(parents=True)
            raw: dict[str, object] = {
                "config": {"dataset": dataset, "method": method, "seed": seed, "steps": 16},
                "test": {"accuracy": 0.8 + seed * 0.01 + (0.02 if method == "gdi" else 0)},
                "train": {"accuracy": 0.9},
                "training_seconds_including_compile": 1.0,
                "diagnostics": {"layer_cosines": [0.9]},
                "best_epoch": 1,
            }
            (directory / "summary.json").write_text(json.dumps(raw))
    trials: list[Trial] = read_trials(root)
    assert len(trials) == 8
    assert "Report" in repr(Report(trials=tuple(trials)))
    output: Path = tmp_path / "report"
    main(["--runs", str(root), "--output", str(output), "--reference", str(root)])
    assert f"| {dataset} | 16 | +2.00 |" in (output / "results.md").read_text()
    assert "width-32" not in (output / "results.md").read_text()
    assert "55k/5k/10k" not in (output / "results.md").read_text()
    assert len((output / "trials.csv").read_text().splitlines()) == 9
    assert (output / "comparison.png").stat().st_size > 0
    duplicate: Path = root / "duplicate"
    duplicate.mkdir()
    (duplicate / "summary.json").write_text((root / "bp-0" / "summary.json").read_text())
    with pytest.raises(ValueError, match="Duplicate"):
        read_trials(root)


def test_report_rejects_empty_or_unpaired_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No completed"):
        read_trials(tmp_path)
    trial: Trial = Trial(
        dataset="mnist",
        method="gdi",
        steps=16,
        seed=0,
        accuracy=80,
        train_accuracy=90,
        seconds=1,
        first_cosine=1,
        best_epoch=1,
    )
    with pytest.raises(ValueError, match="Unpaired"):
        Report(trials=(trial,))(output=tmp_path)


def test_ce_comparison_pairs_seeds_and_rejects_missing_reference(tmp_path: Path) -> None:
    baseline: tuple[Trial, ...] = tuple(
        Trial(
            dataset="fashion",
            method="gdi",
            steps=16,
            seed=seed,
            accuracy=80 + seed,
            train_accuracy=90,
            seconds=1,
            first_cosine=0.9,
            best_epoch=1,
        )
        for seed in range(2)
    )
    current: tuple[Trial, ...] = tuple(
        replace(trial, accuracy=trial.accuracy + 2) for trial in baseline
    )
    report: Report = Report(trials=current)
    report.compare(reference=baseline, output=tmp_path)
    text: str = (tmp_path / "comparison.md").read_text()
    assert (tmp_path / "loss.png").stat().st_size > 0
    assert "80.50 ± 0.71" in text and "82.50 ± 0.71" in text
    assert "| +2.00 | +2.00, +2.00 |" in text
    report.compare(
        reference=baseline,
        output=tmp_path / "drift",
        current_name="Optimizer-aware GDI",
        reference_name="Ordinary GDI",
    )
    assert (
        "Optimizer-aware GDI versus Ordinary GDI"
        in (tmp_path / "drift" / "comparison.md").read_text()
    )
    with pytest.raises(ValueError, match="Missing squared-error reference"):
        report.compare(reference=baseline[:1], output=tmp_path)
