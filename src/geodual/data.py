"""Checksum-verified IDX datasets and deterministic stratified holdouts."""

import gzip
import hashlib
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int64]

FILES: tuple[str, ...] = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
SOURCES: dict[str, tuple[str, tuple[str, ...], float, float]] = {
    "mnist": (
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        (
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
            "d53e105ee54ea40749a09fcbcd1e9432",
            "9fb629c4189551a2d022fa330f9573f3",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
        0.1307,
        0.3081,
    ),
    "fashion": (
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        (
            "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
            "25c81989df183df01b3e8a0aad5dffbe",
            "bef4ecab320f06d8554ea6380940ec79",
            "bb300cfdad3c16e7a12a480ee83cd310",
        ),
        0.2860,
        0.3530,
    ),
}


@dataclass(frozen=True)
class Examples:
    """Feature matrix, one-hot targets, and original training IDs."""

    x: FloatArray
    y: FloatArray
    ids: IntArray


@dataclass(frozen=True)
class Dataset:
    """Training and validation partitions plus untouched official test data."""

    train: Examples
    validation: Examples
    test: Examples


def read_idx(path: Path) -> NDArray[np.uint8]:
    stream: gzip.GzipFile
    with gzip.open(path, "rb") as stream:
        header: bytes = stream.read(4)
        if len(header) != 4 or header[:3] != b"\x00\x00\x08" or header[3] not in (1, 3):
            raise ValueError("Expected unsigned-byte IDX labels or images")
        dimensions: int = header[3]
        shape: tuple[int, ...] = struct.unpack(">" + "I" * dimensions, stream.read(4 * dimensions))
        values: NDArray[np.uint8] = np.frombuffer(stream.read(), dtype=np.uint8)
        if values.size != int(np.prod(shape)):
            raise ValueError("IDX payload length does not match header")
        return values.reshape(shape)


def fetch(*, root: Path, filename: str, base: str, checksum: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path: Path = root / filename
    payload: bytes
    if path.exists():
        payload = path.read_bytes()
    else:
        with urllib.request.urlopen(base + filename, timeout=120) as response:
            payload = response.read()
    if hashlib.md5(payload, usedforsecurity=False).hexdigest() != checksum:
        raise ValueError(f"Dataset checksum mismatch: {path}")
    if not path.exists():
        path.write_bytes(payload)
    return path


def partition(
    *,
    images: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    mean: float,
    std: float,
    validation_per_class: int = 500,
    seed: int = 42,
) -> tuple[Examples, Examples]:
    if validation_per_class < 1 or std <= 0:
        raise ValueError("Positive holdout count and normalization scale required")
    rng: np.random.Generator = np.random.default_rng(seed)
    train: list[IntArray] = []
    validation: list[IntArray] = []
    label: np.uint8
    for label in np.unique(labels):
        indices: IntArray = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) <= validation_per_class:
            raise ValueError("Each class must retain training examples")
        validation.append(indices[:validation_per_class])
        train.append(indices[validation_per_class:])
    x: FloatArray = (
        (images.reshape(len(images), -1).astype(np.float32) / 255 - mean) / std
    ).astype(np.float32)
    y: FloatArray = np.eye(10, dtype=np.float32)[labels]
    tr: IntArray = np.sort(np.concatenate(train))
    val: IntArray = np.sort(np.concatenate(validation))
    return Examples(x=x[tr], y=y[tr], ids=tr), Examples(x=x[val], y=y[val], ids=val)


def load(*, name: str, root: Path) -> Dataset:
    base: str
    checksums: tuple[str, ...]
    mean: float
    std: float
    base, checksums, mean, std = SOURCES[name]
    paths: list[Path] = []
    filename: str
    checksum: str
    for filename, checksum in zip(FILES, checksums, strict=True):
        paths.append(fetch(root=root / name, filename=filename, base=base, checksum=checksum))
    train: Examples
    validation: Examples
    train, validation = partition(
        images=read_idx(paths[0]),
        labels=read_idx(paths[1]),
        mean=mean,
        std=std,
    )
    images: NDArray[np.uint8] = read_idx(paths[2])
    labels: NDArray[np.uint8] = read_idx(paths[3])
    test: Examples = Examples(
        x=((images.reshape(len(images), -1).astype(np.float32) / 255 - mean) / std).astype(
            np.float32,
        ),
        y=np.eye(10, dtype=np.float32)[labels],
        ids=-1 - np.arange(len(labels), dtype=np.int64),
    )
    return Dataset(train=train, validation=validation, test=test)
