"""Offline ImageNet sampling, decoding, cache, and CLI isolation tests."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from geodual.cli import main
from geodual.data import Dataset, IntArray
from geodual.imagenet import SIZE, ImageNet
from geodual.training import Config


def test_selection_is_uniform_disjoint_and_repeatable(tmp_path: Path) -> None:
    loader: ImageNet = ImageNet(root=tmp_path, fraction=0.1, tuning_per_class=2, classes=3)
    labels: IntArray = np.repeat(np.arange(3, dtype=np.int64), repeats=20)
    train: IntArray
    tuning: IntArray
    train, tuning = loader.select(labels=labels)
    np.testing.assert_array_equal(actual=np.bincount(labels[train]), desired=[2, 2, 2])
    np.testing.assert_array_equal(actual=np.bincount(labels[tuning]), desired=[2, 2, 2])
    assert set(train).isdisjoint(tuning)
    np.testing.assert_array_equal(actual=train, desired=loader.select(labels=labels)[0])
    assert not np.array_equal(train, np.array([0, 1, 20, 21, 40, 41]))
    with pytest.raises(ValueError, match="fraction"):
        ImageNet(root=tmp_path, fraction=1).select(labels=labels)
    with pytest.raises(ValueError, match="class"):
        ImageNet(root=tmp_path, classes=4).select(labels=labels)
    with pytest.raises(ValueError, match="Insufficient"):
        ImageNet(root=tmp_path, classes=3, tuning_per_class=20).select(labels=labels)


def test_prepare_decode_and_cache(tmp_path: Path) -> None:
    source: Path = tmp_path / "fixture"
    (source / "data").mkdir(parents=True)
    payload: io.BytesIO = io.BytesIO()
    Image.new("RGB", (SIZE, SIZE), color=(255, 0, 128)).save(payload, format="PNG")
    images: list[dict[str, object]] = [{"bytes": payload.getvalue(), "path": None}] * 40
    paths: list[Path] = []
    shard: int
    for shard in range(2):
        path: Path = source / "data" / f"train-{shard}.parquet"
        pq.write_table(pa.table({"image": images[:20], "label": [shard] * 20}), where=path)
        paths.append(path)
    pq.write_table(
        pa.table({"image": images[:2], "label": [0, 1]}),
        where=source / "data" / "validation-0.parquet",
    )
    loader: ImageNet = ImageNet(root=tmp_path, classes=2, tuning_per_class=2)
    decoded: np.ndarray = loader.decode(paths=paths, indices=np.array([0, 19, 20, 39]))
    assert decoded.shape == (4, SIZE * SIZE * 3)
    np.testing.assert_array_equal(actual=decoded[0, :3], desired=[255, 0, 128])
    with pytest.raises(ValueError, match="not all decoded"):
        loader.decode(paths=paths, indices=np.array([40]))
    download: MagicMock
    with patch("geodual.imagenet.snapshot_download", return_value=str(source)) as download:
        data: Dataset = loader()
        loader.prepare()
        assert download.call_count == 1
    assert len(data.train.x) == 4 and len(data.validation.x) == 4 and len(data.test.x) == 2
    assert set(data.train.ids).isdisjoint(data.validation.ids)
    assert (data.test.ids < 0).all()
    assert data.train.x.dtype == np.float32
    np.testing.assert_allclose(
        actual=data.train.x[0, :3], desired=[1, -1, 128 / 127.5 - 1], atol=1e-7
    )
    manifest: dict = json.loads((tmp_path / "imagenet" / "manifest.json").read_text())
    assert len(manifest["source_sha256"]) == 3
    assert manifest["training_class_counts"] == [2, 2]
    with pytest.raises(ValueError, match="different sampling"):
        ImageNet(root=tmp_path, classes=2, fraction=0.2).prepare()
    payload = io.BytesIO()
    Image.new("RGB", (1, 1)).save(payload, format="PNG")
    pq.write_table(pa.table({"image": [{"bytes": payload.getvalue()}]}), where=paths[0])
    with pytest.raises(ValueError, match="image size"):
        loader.decode(paths=paths[:1], indices=np.array([0]))


def test_imagenet_cli_sets_dimensions(tmp_path: Path) -> None:
    trainer: MagicMock
    loader: MagicMock
    with (
        patch("geodual.cli.ImageNet") as loader,
        patch("geodual.cli.Trainer") as trainer,
    ):
        trainer.return_value.run.return_value = {"ok": True}
        main(["--dataset", "imagenet", "--output", str(tmp_path / "run")])
        config: Config = trainer.call_args.kwargs["config"]
        assert (config.inputs, config.outputs, config.width) == (12288, 1000, 128)
        assert (config.capacity, config.memory_age) == (4000, 64)
        assert loader.call_count == 1
        assert (tmp_path / "run" / "summary.json").exists()
