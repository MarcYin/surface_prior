"""Disk-backed cache for STAC search + tile partition results.

Repeated builds with the same ``(stac_url, collection, bbox, datetime,
max_cloud_cover)`` produce identical search responses; classifying the
same grid against the same scene set produces identical partitions.
Across separate process invocations (e.g. one process per year in a
5-year batch) these reads otherwise repeat from scratch.

Cached data is stored unsigned so URL signers with finite-TTL tokens
(Planetary Computer, CDSE) can re-sign on cache hit without producing
stale hrefs. Cache files are plain JSON written atomically; corrupt or
unreadable files behave as cache misses so the cache is best-effort and
self-healing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "surface-priors" / "stac"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StacDiskCache:
    """Disk-backed cache for ``list_scenes`` and ``tile_partition`` outputs.

    ``root`` is the directory used; it is created lazily on first write.
    All filenames are stable hex digests, so two processes can share the
    same cache directory safely.
    """

    root: Path

    @classmethod
    def from_arg(
        cls,
        arg: Union[bool, str, Path, "StacDiskCache", None],
    ) -> Optional["StacDiskCache"]:
        """Resolve constructor-friendly inputs to an instance or ``None``.

        - ``None``/``False``: disabled.
        - ``True``: default location (``~/.cache/surface-priors/stac``).
        - ``str``/``Path``: explicit directory.
        - ``StacDiskCache``: returned as-is.
        """

        if arg is None or arg is False:
            return None
        if isinstance(arg, cls):
            return arg
        if arg is True:
            return cls(root=DEFAULT_CACHE_DIR)
        return cls(root=Path(arg))

    def search_key(
        self,
        *,
        stac_url: str,
        collection: str,
        wgs84_bounds: Sequence[float],
        datetime_range: Sequence[str],
        max_cloud_cover: Optional[float],
    ) -> str:
        payload = {
            "schema": SCHEMA_VERSION,
            "stac_url": str(stac_url),
            "collection": str(collection),
            "bbox": [float(value) for value in wgs84_bounds],
            "start": str(datetime_range[0]),
            "end": str(datetime_range[1]),
            "max_cloud_cover": (
                None if max_cloud_cover is None else float(max_cloud_cover)
            ),
        }
        return _hash_payload(payload)

    def partition_key(
        self,
        *,
        scenes_signature: str,
        grid_signature: Sequence[Any],
        layout_signature: Sequence[Any],
    ) -> str:
        payload = {
            "schema": SCHEMA_VERSION,
            "scenes": scenes_signature,
            "grid": list(grid_signature),
            "layout": list(layout_signature),
        }
        return _hash_payload(payload)

    def load_search(self, key: str) -> Optional[List[Mapping[str, Any]]]:
        path = self._search_path(key)
        return _safe_load(path)

    def store_search(self, key: str, raw_items: Sequence[Mapping[str, Any]]) -> None:
        path = self._search_path(key)
        _atomic_write(path, list(raw_items))

    def load_partition(self, key: str) -> Optional[Mapping[str, Any]]:
        path = self._partition_path(key)
        return _safe_load(path)

    def store_partition(self, key: str, payload: Mapping[str, Any]) -> None:
        path = self._partition_path(key)
        _atomic_write(path, dict(payload))

    def scout_key(
        self,
        *,
        stac_url: str,
        collection: str,
        item_id: str,
        grid_signature: Sequence[Any],
        layout_signature: Sequence[Any],
        scout_factor: int,
    ) -> str:
        payload = {
            "schema": SCHEMA_VERSION,
            "kind": "scout",
            "stac_url": str(stac_url),
            "collection": str(collection),
            "item_id": str(item_id),
            "grid": list(grid_signature),
            "layout": list(layout_signature),
            "scout_factor": int(scout_factor),
        }
        return _hash_payload(payload)

    def load_scout(self, key: str) -> Optional[List[Mapping[str, Any]]]:
        path = self._scout_path(key)
        return _safe_load(path)

    def store_scout(self, key: str, stats: Sequence[Mapping[str, Any]]) -> None:
        path = self._scout_path(key)
        _atomic_write(path, list(stats))

    def _search_path(self, key: str) -> Path:
        return self.root / "search" / f"{key}.json"

    def _partition_path(self, key: str) -> Path:
        return self.root / "partition" / f"{key}.json"

    def _scout_path(self, key: str) -> Path:
        return self.root / "scout" / f"{key}.json"


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _safe_load(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, Mapping):
        if payload.get("schema") != SCHEMA_VERSION:
            return None
        return payload.get("data")
    return payload  # legacy / bare-array (treat as data)


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA_VERSION, "data": data}
    fd, tmp_name = tempfile.mkstemp(prefix=".cache-", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_path, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp_path.unlink()


def scenes_signature(items: Sequence[Mapping[str, Any]]) -> str:
    """Stable hash over an unsigned item list.

    Includes only the fields the partition classifier actually consumes:
    item id, MGRS code (parsed via :func:`surface_priors.sources.stac_api._mgrs_tile_from_item`
    by the caller), and geometry. Other item metadata is irrelevant.
    """

    digest = hashlib.sha1()
    for item in items:
        sig = json.dumps(
            {
                "id": str(item.get("id", "")),
                "geometry": item.get("geometry"),
                "props": {
                    key: item.get("properties", {}).get(key)
                    for key in ("s2:mgrs_tile", "mgrs:tile", "grid:code")
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(sig.encode("utf-8"))
    return digest.hexdigest()
