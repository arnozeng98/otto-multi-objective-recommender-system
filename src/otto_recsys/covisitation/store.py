from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from otto_recsys.artifacts import stable_hash


@dataclass(frozen=True, slots=True)
class MatrixStoreBuildResult:
    partitions: int
    source_aids: int
    neighbors: int


@dataclass(frozen=True, slots=True)
class MatrixStoreSuiteResult:
    stores: dict[str, MatrixStoreBuildResult]


def build_matrix_store(
    source: Path,
    destination: Path,
    *,
    overwrite: bool = False,
) -> MatrixStoreBuildResult:
    """Convert a partitioned matrix artifact to bounded, memory-mapped arrays."""
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    partitions = int(manifest["result"]["partitions"])
    temporary = destination.with_name(f"{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    source_count = 0
    neighbor_count = 0
    try:
        for partition in range(partitions):
            parquet = source / "matrix" / f"partition-{partition:03d}.parquet"
            if not parquet.exists():
                continue
            table = pq.read_table(
                parquet,
                columns=["source_aid", "target_aid", "weight", "rank"],
            )
            source_aids = table["source_aid"].to_numpy().astype(np.int32, copy=False)
            targets = table["target_aid"].to_numpy().astype(np.int32, copy=False)
            scores = table["weight"].to_numpy().astype(np.float32, copy=False)
            ranks = table["rank"].to_numpy().astype(np.uint16, copy=False)
            starts = np.flatnonzero(
                np.concatenate((np.array([True]), source_aids[1:] != source_aids[:-1]))
            )
            unique_sources = source_aids[starts]
            offsets = np.concatenate((starts, np.array([len(source_aids)]))).astype(np.int64)
            prefix = temporary / f"partition-{partition:03d}"
            np.save(prefix.with_name(f"{prefix.name}-sources.npy"), unique_sources)
            np.save(prefix.with_name(f"{prefix.name}-offsets.npy"), offsets)
            np.save(prefix.with_name(f"{prefix.name}-targets.npy"), targets)
            np.save(prefix.with_name(f"{prefix.name}-scores.npy"), scores)
            np.save(prefix.with_name(f"{prefix.name}-ranks.npy"), ranks)
            source_count += len(unique_sources)
            neighbor_count += len(targets)
        result = MatrixStoreBuildResult(
            partitions=partitions,
            source_aids=source_count,
            neighbors=neighbor_count,
        )
        store_manifest = {
            "stage": "matrix_store",
            "config_hash": stable_hash({"partitions": partitions}),
            "input": str(source),
            "result": asdict(result),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(store_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(temporary), str(destination))
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_matrix_store_suite(
    source: Path,
    destination: Path,
    *,
    overwrite: bool = False,
) -> MatrixStoreSuiteResult:
    """Convert every matrix in a co-visitation suite to memory-mapped storage."""
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    suite_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    names = sorted(suite_manifest["result"]["matrices"])
    temporary = destination.with_name(f"{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        stores = {name: build_matrix_store(source / name, temporary / name) for name in names}
        result = MatrixStoreSuiteResult(stores=stores)
        manifest = {
            "stage": "matrix_store_suite",
            "config_hash": stable_hash({"names": names}),
            "input": str(source),
            "result": asdict(result),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(temporary), str(destination))
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class MatrixStore:
    """Lazy memory-mapped random access to partitioned co-visitation neighbors."""

    def __init__(self, path: Path) -> None:
        self.path = path
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.partitions = int(manifest["result"]["partitions"])
        self._cache: dict[int, tuple[Any, Any, Any, Any, Any] | None] = {}

    def _load_partition(self, partition: int) -> tuple[Any, Any, Any, Any, Any] | None:
        if partition in self._cache:
            return self._cache[partition]
        prefix = self.path / f"partition-{partition:03d}"
        sources_path = prefix.with_name(f"{prefix.name}-sources.npy")
        if not sources_path.exists():
            self._cache[partition] = None
            return None
        arrays = (
            np.load(sources_path, mmap_mode="r"),
            np.load(prefix.with_name(f"{prefix.name}-offsets.npy"), mmap_mode="r"),
            np.load(prefix.with_name(f"{prefix.name}-targets.npy"), mmap_mode="r"),
            np.load(prefix.with_name(f"{prefix.name}-scores.npy"), mmap_mode="r"),
            np.load(prefix.with_name(f"{prefix.name}-ranks.npy"), mmap_mode="r"),
        )
        self._cache[partition] = arrays
        return arrays

    def neighbors(self, source_aid: int) -> tuple[tuple[int, float, int], ...]:
        arrays = self._load_partition(source_aid % self.partitions)
        if arrays is None:
            return ()
        sources, offsets, targets, scores, ranks = arrays
        index = int(np.searchsorted(sources, source_aid))
        if index >= len(sources) or int(sources[index]) != source_aid:
            return ()
        start = int(offsets[index])
        end = int(offsets[index + 1])
        return tuple(
            (int(targets[row]), float(scores[row]), int(ranks[row])) for row in range(start, end)
        )
