"""Reproducible stratified ImageNet-1K subsets from a pinned 64-pixel repack."""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from numpy.typing import NDArray
from PIL import Image

from .data import Dataset, Examples, FloatArray, IntArray

REPOSITORY: str = "benjamin-paine/imagenet-1k-64x64"
REVISION: str = "f12246fedf1127d5b34bea760b61ef00e0f3d603"
SIZE: int = 64
CLASSES: int = 1000
DECODE_BATCH: int = 2048


@dataclass(frozen=True)
class ImageNet:
    """Cache sampled images, with disjoint train-derived tuning and official validation."""

    root: Path
    fraction: float = 0.1
    seed: int = 42
    tuning_per_class: int = 10
    classes: int = CLASSES

    def select(self, *, labels: IntArray) -> tuple[IntArray, IntArray]:
        if not 0 < self.fraction < 1 or self.tuning_per_class < 1:
            raise ValueError("Require a fraction between zero and one and positive tuning count")
        if not np.array_equal(np.unique(labels), np.arange(self.classes)):
            raise ValueError("Expected every ImageNet class exactly in the configured label range")
        rng: np.random.Generator = np.random.default_rng(self.seed)
        training: list[IntArray] = []
        tuning: list[IntArray] = []
        label: int
        for label in range(self.classes):
            indices: IntArray = rng.permutation(np.flatnonzero(labels == label))
            count: int = int(np.floor(len(indices) * self.fraction))
            if count < 1 or count + self.tuning_per_class > len(indices):
                raise ValueError("Insufficient examples per class for training and tuning")
            training.append(indices[:count])
            tuning.append(indices[count : count + self.tuning_per_class])
        return np.sort(np.concatenate(training)), np.sort(np.concatenate(tuning))

    def decode(self, *, paths: list[Path], indices: IntArray) -> NDArray[np.uint8]:
        images: NDArray[np.uint8] = np.empty((len(indices), SIZE * SIZE * 3), dtype=np.uint8)
        cursor: int = 0
        written: int = 0
        path: Path
        for path in paths:
            batch: Any
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=DECODE_BATCH, columns=["image"]
            ):
                selected: IntArray = indices[(indices >= cursor) & (indices < cursor + len(batch))]
                rows: list[dict[str, Any]] = batch.take(selected - cursor).to_pylist()
                row: dict[str, Any]
                for row in rows:
                    with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
                        if image.size != (SIZE, SIZE):
                            raise ValueError("ImageNet repack contains an unexpected image size")
                        images[written] = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(
                            -1
                        )
                    written += 1
                cursor += len(batch)
        if written != len(indices):
            raise ValueError("Selected image indices were not all decoded")
        return images

    def prepare(self) -> None:
        directory: Path = self.root / "imagenet"
        directory.mkdir(parents=True, exist_ok=True)
        manifest: Path = directory / "manifest.json"
        settings: dict[str, object] = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "fraction": self.fraction,
            "seed": self.seed,
            "tuning_per_class": self.tuning_per_class,
            "classes": self.classes,
        }
        if manifest.exists():
            saved: dict[str, Any] = json.loads(manifest.read_text())
            if saved["settings"] != settings:
                raise ValueError("Existing ImageNet cache uses different sampling settings")
            return
        source: Path = Path(
            snapshot_download(
                repo_id=REPOSITORY,
                repo_type="dataset",
                revision=REVISION,
                allow_patterns=["data/train-*.parquet", "data/validation-*.parquet", "README.md"],
                local_dir=directory / "source",
                max_workers=3,
            )
        )
        train_paths: list[Path] = sorted((source / "data").glob("train-*.parquet"))
        test_paths: list[Path] = sorted((source / "data").glob("validation-*.parquet"))
        labels: IntArray = np.concatenate(
            [pq.read_table(path, columns=["label"])["label"].to_numpy() for path in train_paths]
        ).astype(np.int64)
        train: IntArray
        tuning: IntArray
        train, tuning = self.select(labels=labels)
        test_labels: IntArray = np.concatenate(
            [pq.read_table(path, columns=["label"])["label"].to_numpy() for path in test_paths]
        ).astype(np.int64)
        if not np.array_equal(np.unique(test_labels), np.arange(self.classes)):
            raise ValueError("Official validation is missing classes")
        split: str
        indices: IntArray
        paths: list[Path]
        targets: IntArray
        counts: dict[str, int] = {}
        for split, indices, paths, targets in (
            ("train", train, train_paths, labels),
            ("tuning", tuning, train_paths, labels),
            ("evaluation", np.arange(len(test_labels), dtype=np.int64), test_paths, test_labels),
        ):
            print(f"Preparing ImageNet {split}: {len(indices)} images", flush=True)
            pixels: NDArray[np.uint8] = self.decode(paths=paths, indices=indices)
            temporary: Path = directory / f"{split}.partial.npz"
            np.savez(temporary, pixels=pixels, labels=targets[indices], ids=indices)
            temporary.replace(directory / f"{split}.npz")
            counts[split] = len(indices)
        hashes: dict[str, str] = {}
        path: Path
        for path in train_paths + test_paths:
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.write_text(
            json.dumps(
                {
                    "settings": settings,
                    "counts": counts,
                    "source_sha256": hashes,
                    "source_train_examples": len(labels),
                    "training_class_counts": np.bincount(
                        labels[train], minlength=self.classes
                    ).tolist(),
                    "evaluation_source": "official labeled validation; never used for selection",
                    "preprocessing": (
                        "Repack center crop and Lanczos resize to RGB 64x64; (pixel/255-0.5)/0.5"
                    ),
                },
                indent=2,
            )
            + "\n"
        )

    def __call__(self) -> Dataset:
        self.prepare()
        splits: list[Examples] = []
        split: str
        for split in ("train", "tuning", "evaluation"):
            with np.load(self.root / "imagenet" / f"{split}.npz", allow_pickle=False) as cache:
                x: FloatArray = cache["pixels"].astype(np.float32)
                x /= 255
                x -= 0.5
                x /= 0.5
                ids: IntArray = cache["ids"].astype(np.int64)
                if split == "evaluation":
                    ids = -1 - ids
                splits.append(
                    Examples(
                        x=x,
                        y=np.eye(self.classes, dtype=np.float32)[cache["labels"]],
                        ids=ids,
                    )
                )
        return Dataset(train=splits[0], validation=splits[1], test=splits[2])
