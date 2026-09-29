"""Offline data integrity and split-isolation checks."""

import gzip
import hashlib
import io
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from edc.data import Dataset, Examples, fetch, load, partition, read_idx


def test_idx_validation(tmp_path: Path) -> None:
    path: Path = tmp_path / "labels.gz"
    path.write_bytes(gzip.compress(struct.pack(">II", 2049, 3) + bytes([1, 2, 3])))
    np.testing.assert_array_equal(actual=read_idx(path), desired=[1, 2, 3])
    path.write_bytes(gzip.compress(struct.pack(">II", 2049, 2) + bytes([1])))
    with pytest.raises(ValueError, match="length"):
        read_idx(path)
    path.write_bytes(gzip.compress(b"invalid"))
    with pytest.raises(ValueError, match="IDX"):
        read_idx(path)
    path.write_bytes(gzip.compress(struct.pack(">IIII", 2051, 1, 2, 2) + bytes([1, 2, 3, 4])))
    assert read_idx(path).shape == (1, 2, 2)


def test_checksum_download_and_cache(tmp_path: Path) -> None:
    payload: bytes = b"dataset"
    checksum: str = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    request: MagicMock
    with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as request:
        path: Path = fetch(
            root=tmp_path, filename="sample.gz", base="https://example.org/", checksum=checksum
        )
        assert path.read_bytes() == payload
        assert (
            fetch(
                root=tmp_path, filename="sample.gz", base="https://example.org/", checksum=checksum
            )
            == path
        )
        assert request.call_count == 1
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        fetch(root=tmp_path, filename="sample.gz", base="https://example.org/", checksum=checksum)


def test_partition_is_stratified_disjoint_repeatable() -> None:
    images: NDArray[np.uint8] = np.arange(48, dtype=np.uint8).reshape(12, 2, 2)
    labels: NDArray[np.uint8] = np.repeat(np.arange(3, dtype=np.uint8), repeats=4)
    train: Examples
    validation: Examples
    train, validation = partition(
        images=images, labels=labels, mean=0, std=1, validation_per_class=1
    )
    assert len(train.x) == 9 and len(validation.x) == 3
    assert set(train.ids).isdisjoint(set(validation.ids))
    assert set(train.ids) | set(validation.ids) == set(range(12))
    np.testing.assert_array_equal(actual=validation.y.argmax(axis=1), desired=[0, 1, 2])
    repeated: tuple[Examples, Examples] = partition(
        images=images, labels=labels, mean=0, std=1, validation_per_class=1
    )
    np.testing.assert_array_equal(actual=train.x, desired=repeated[0].x)
    with pytest.raises(ValueError):
        partition(images=images, labels=labels, mean=0, std=0)
    with pytest.raises(ValueError):
        partition(images=images, labels=labels, mean=0, std=1, validation_per_class=4)


def test_load_uses_official_test_separately(tmp_path: Path) -> None:
    train_images: NDArray[np.uint8] = np.zeros((1002, 2, 2), dtype=np.uint8)
    train_labels: NDArray[np.uint8] = np.repeat(np.arange(2, dtype=np.uint8), repeats=501)
    test_images: NDArray[np.uint8] = np.ones((2, 2, 2), dtype=np.uint8)
    test_labels: NDArray[np.uint8] = np.arange(2, dtype=np.uint8)
    with (
        patch("edc.data.fetch", return_value=tmp_path / "unused"),
        patch(
            "edc.data.read_idx",
            side_effect=[train_images, train_labels, test_images, test_labels],
        ),
    ):
        data: Dataset = load(name="mnist", root=tmp_path)
    assert len(data.train.x) == 2 and len(data.validation.x) == 1000
    assert len(data.test.x) == 2 and (data.test.ids < 0).all()
    assert data.test.x.dtype == np.float32
