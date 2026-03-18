"""AnnData serialization helpers for Cell Annotation checkpoints."""

from __future__ import annotations

import copy
import hashlib
import importlib.machinery
import json
import os
import re
import sys
import tempfile
import time
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_dask_stub = sys.modules.get("dask")
if _dask_stub is not None and getattr(_dask_stub, "__spec__", None) is None:  # pragma: no cover - test bootstrap quirk
    _dask_stub.__spec__ = importlib.machinery.ModuleSpec("dask", loader=None)
if _dask_stub is not None and not hasattr(_dask_stub, "__path__"):  # pragma: no cover - test bootstrap quirk
    _dask_stub.__path__ = []
_dask_array_stub = sys.modules.get("dask.array")
if _dask_array_stub is None and _dask_stub is not None:  # pragma: no cover - test bootstrap quirk
    dask_array_stub = types.ModuleType("dask.array")
    dask_array_stub.__spec__ = importlib.machinery.ModuleSpec("dask.array", loader=None)
    sys.modules["dask.array"] = dask_array_stub
    _dask_stub.array = dask_array_stub
    _dask_array_stub = dask_array_stub
if _dask_array_stub is not None and not hasattr(_dask_array_stub, "Array"):  # pragma: no cover - test bootstrap quirk
    _dask_array_stub.Array = type("Array", (), {})

import anndata as ad
import numpy as np

if hasattr(ad, "settings"):  # pragma: no branch - depends on anndata version
    try:
        ad.settings.allow_write_nullable_strings = True
    except Exception:  # pragma: no cover - defensive for older/newer anndata variants
        pass

from .store import atomic_replace

SCHEMA_VERSION = "1.0.0"
ZERO_SHA256 = "0" * 64
_MISSING = object()
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?$")
_SCHEMA_SPEC = {
    "version": SCHEMA_VERSION,
    "required_uns": [
        "artifact",
        "ui",
        "palette",
        "zscore_params",
        "filters",
        "row_linkage",
        "row_linkage_basis",
        "marker_sets",
        "flowsom",
        "checkpoint",
    ],
    "required_ui": [
        "orientation",
        "row_sort",
        "col_sort",
        "selected_channels_ordered",
        "row_order",
        "col_order",
    ],
    "required_marker_sets": [
        "training",
        "display_extra",
        "available",
        "linkage",
        "expanded_training",
        "panel",
    ],
}
SCHEMA_HASH = hashlib.sha256(
    json.dumps(_SCHEMA_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def serialize_heatmap_state(display: dict, *, flowsom: dict, meta: dict) -> ad.AnnData:
    """Serialize a Cell Annotation heatmap checkpoint into an :class:`AnnData` artifact."""

    obs_names = _normalize_name_list(
        _coalesce(
            (display, meta),
            ("obs_names", "row_names", "cluster_ids", "rows"),
        ),
        "obs_names",
    )
    var_names = _normalize_name_list(
        _coalesce(
            (display, meta),
            ("var_names", "marker_ids", "markers", "cols", "columns"),
        ),
        "var_names",
    )

    X = _to_matrix(
        _coalesce(
            (display, meta),
            ("X", "x", "zscore_matrix", "z_scored_medians", "zscored_medians"),
        ),
        shape=(len(obs_names), len(var_names)),
        label="X",
    )

    adata = ad.AnnData(X=X)
    adata.obs_names = obs_names
    adata.var_names = var_names

    median = _coalesce(
        (display, meta),
        ("layers.median", "median_matrix", "median", "raw_medians"),
        default=None,
    )
    if median is not None:
        adata.layers["median"] = _to_matrix(
            median,
            shape=(len(obs_names), len(var_names)),
            label="layers['median']",
        )

    obsm = _coalesce((display, meta), ("obsm", "embeddings"), default={})
    if obsm is None:
        obsm = {}
    if not isinstance(obsm, Mapping):
        raise ValueError("obsm must be a mapping of embedding names to 2D arrays")
    for name, value in obsm.items():
        adata.obsm[str(name)] = _to_embedding(value, n_obs=len(obs_names), label=f"obsm[{name!r}]")

    marker_sets = _normalize_marker_sets(display, flowsom, meta, var_names)
    row_linkage_basis = _normalize_row_linkage_basis(display, meta, marker_sets)
    row_linkage = _normalize_row_linkage(_coalesce((display, meta), ("row_linkage",), default=[]))

    adata.uns = {
        "artifact": {
            "version": str(_coalesce((meta,), ("artifact.version", "artifact_version"), default=SCHEMA_VERSION)),
            "schema_hash": SCHEMA_HASH,
            "checksums": {},
        },
        "ui": _normalize_ui(display, meta, obs_names, var_names),
        "palette": _normalize_palette(display, meta),
        "zscore_params": _normalize_zscore_params(display, meta, var_names),
        "filters": _normalize_filters(display, meta),
        "row_linkage": row_linkage,
        "row_linkage_basis": row_linkage_basis,
        "marker_sets": marker_sets,
        "flowsom": _normalize_flowsom(flowsom, marker_sets),
        "checkpoint": _normalize_checkpoint(meta),
    }

    _refresh_checksums(adata)
    return validate_artifact(adata)


def validate_artifact(path_or_adata: str | Path | ad.AnnData) -> ad.AnnData:
    """Validate a serialized Cell Annotation artifact and return the loaded object."""

    source_path: Path | None = None
    if isinstance(path_or_adata, (str, Path)):
        source_path = Path(path_or_adata)
        adata = ad.read_h5ad(source_path)
    elif isinstance(path_or_adata, ad.AnnData):
        adata = path_or_adata
        filename = getattr(adata, "filename", None)
        if filename:
            source_path = Path(filename)
    else:
        raise TypeError("validate_artifact expects a filesystem path or an AnnData object")

    if adata.X is None:
        raise ValueError("artifact is missing X")
    X = np.asarray(adata.X)
    if X.ndim != 2:
        raise ValueError("artifact X must be a 2D matrix")
    if tuple(X.shape) != (adata.n_obs, adata.n_vars):
        raise ValueError("artifact X shape does not match obs/var dimensions")
    if X.dtype != np.float32:
        raise ValueError("artifact X must use float32")

    if len(set(map(str, adata.obs_names))) != adata.n_obs:
        raise ValueError("obs_names must be unique")
    if len(set(map(str, adata.var_names))) != adata.n_vars:
        raise ValueError("var_names must be unique")

    uns = adata.uns
    for key in _SCHEMA_SPEC["required_uns"]:
        if key not in uns:
            raise ValueError(f"artifact.uns is missing required block {key!r}")

    artifact = _ensure_mapping(uns["artifact"], "artifact")
    if artifact.get("schema_hash") != SCHEMA_HASH:
        raise ValueError("artifact schema_hash does not match the expected Cell Annotation schema")

    checksums = _ensure_mapping(artifact.get("checksums"), "artifact.checksums")
    expected_x_sha = checksums.get("x_sha256")
    if expected_x_sha != _hash_array(X):
        raise ValueError("artifact checksum mismatch for X")

    if "median_sha256" in checksums:
        if "median" not in adata.layers:
            raise ValueError("artifact checksum advertises median layer, but layers['median'] is missing")
        median = np.asarray(adata.layers["median"])
        if median.dtype != np.float32:
            raise ValueError("layers['median'] must use float32")
        if checksums["median_sha256"] != _hash_array(median):
            raise ValueError("artifact checksum mismatch for layers['median']")

    ui = _ensure_mapping(uns["ui"], "ui")
    for key in _SCHEMA_SPEC["required_ui"]:
        if key not in ui:
            raise ValueError(f"ui block is missing required key {key!r}")
    _validate_order(ui["row_order"], list(map(str, adata.obs_names)), "row_order")
    _validate_order(ui["col_order"], list(map(str, adata.var_names)), "col_order")
    if checksums.get("row_order_sha256") != _hash_json(ui["row_order"]):
        raise ValueError("artifact checksum mismatch for row_order")
    if checksums.get("col_order_sha256") != _hash_json(ui["col_order"]):
        raise ValueError("artifact checksum mismatch for col_order")

    selected_channels = _normalize_name_list(ui["selected_channels_ordered"], "selected_channels_ordered")
    if len(set(selected_channels)) != len(selected_channels):
        raise ValueError("selected_channels_ordered must not contain duplicates")
    unknown_channels = [name for name in selected_channels if name not in set(map(str, adata.var_names))]
    if unknown_channels:
        raise ValueError(f"selected_channels_ordered contains unknown markers: {unknown_channels!r}")
    adata.uns["ui"]["row_order"] = [str(item) for item in ui["row_order"]]
    adata.uns["ui"]["col_order"] = [str(item) for item in ui["col_order"]]
    adata.uns["ui"]["selected_channels_ordered"] = selected_channels

    palette = _ensure_mapping(uns["palette"], "palette")
    _validate_palette_block(palette.get("meta_cluster_colors_present"), "palette.meta_cluster_colors_present")
    _validate_palette_block(palette.get("meta_cluster_colors_all"), "palette.meta_cluster_colors_all")

    zscore_params = _ensure_mapping(uns["zscore_params"], "zscore_params")
    per_marker = _ensure_mapping(zscore_params.get("per_marker"), "zscore_params.per_marker")
    if list(per_marker.keys()) != list(map(str, adata.var_names)):
        raise ValueError("zscore_params.per_marker must align exactly with var_names")
    for marker, stats in per_marker.items():
        stats_mapping = _ensure_mapping(stats, f"zscore_params.per_marker[{marker!r}]")
        if "mean" not in stats_mapping or "std" not in stats_mapping:
            raise ValueError(f"zscore_params.per_marker[{marker!r}] must contain mean and std")

    marker_sets = _ensure_mapping(uns["marker_sets"], "marker_sets")
    flowsom = _ensure_mapping(uns["flowsom"], "flowsom")
    row_linkage_basis = _ensure_mapping(uns["row_linkage_basis"], "row_linkage_basis")

    training = _normalize_name_list(marker_sets.get("training"), "marker_sets.training")
    linkage_markers = _normalize_name_list(marker_sets.get("linkage"), "marker_sets.linkage")
    flowsom_training = _normalize_name_list(flowsom.get("training_markers"), "flowsom.training_markers")
    basis_markers = _normalize_name_list(row_linkage_basis.get("marker_ids"), "row_linkage_basis.marker_ids")
    if flowsom_training != training:
        raise ValueError("flowsom.training_markers must match marker_sets.training")
    if basis_markers != linkage_markers:
        raise ValueError("row_linkage_basis.marker_ids must match marker_sets.linkage")
    adata.uns["marker_sets"]["training"] = training
    adata.uns["marker_sets"]["linkage"] = linkage_markers
    adata.uns["flowsom"]["training_markers"] = flowsom_training
    adata.uns["row_linkage_basis"]["marker_ids"] = basis_markers

    if source_path is not None:
        expected_file_sha = checksums.get("h5ad_sha256")
        if expected_file_sha and expected_file_sha != ZERO_SHA256:
            actual_file_sha = _normalized_h5ad_sha256(source_path, str(expected_file_sha))
            if actual_file_sha != expected_file_sha:
                raise ValueError("artifact checksum mismatch for on-disk h5ad bytes")

    return adata


def write_h5ad_atomic(adata: ad.AnnData, dst_path: str | Path) -> None:
    """Write a validated artifact atomically and persist its on-disk checksum."""

    validated = validate_artifact(adata)
    target = Path(dst_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.tmp",
        suffix=".h5ad",
    )
    os.close(fd)
    tmp = Path(tmp_path)
    working = validated.copy()
    try:
        checksums = _ensure_mapping(working.uns["artifact"]["checksums"], "artifact.checksums")
        checksums["h5ad_sha256"] = ZERO_SHA256
        working.write_h5ad(tmp)

        file_sha = _normalized_h5ad_sha256(tmp, ZERO_SHA256)
        checksums["h5ad_sha256"] = file_sha
        working.write_h5ad(tmp)

        if _normalized_h5ad_sha256(tmp, file_sha) != file_sha:
            raise ValueError("failed to persist a stable h5ad checksum")

        atomic_replace(tmp, target)
        _ensure_mapping(adata.uns["artifact"]["checksums"], "artifact.checksums")["h5ad_sha256"] = file_sha
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _normalize_ui(display: Mapping[str, Any], meta: Mapping[str, Any], obs_names: list[str], var_names: list[str]) -> dict[str, Any]:
    row_order = _normalize_order(
        _coalesce((display, meta), ("ui.row_order", "row_order"), default=obs_names),
        universe=obs_names,
        label="row_order",
    )
    col_order = _normalize_order(
        _coalesce((display, meta), ("ui.col_order", "col_order"), default=var_names),
        universe=var_names,
        label="col_order",
    )
    selected = _coalesce(
        (display, meta),
        ("ui.selected_channels_ordered", "selected_channels_ordered", "selected_channels"),
        default=var_names,
    )
    return {
        "orientation": copy.deepcopy(_coalesce((display, meta), ("ui.orientation", "orientation"), default={})),
        "row_sort": copy.deepcopy(_coalesce((display, meta), ("ui.row_sort", "row_sort"), default={})),
        "col_sort": copy.deepcopy(_coalesce((display, meta), ("ui.col_sort", "col_sort"), default={})),
        "selected_channels_ordered": _normalize_name_list(selected, "selected_channels_ordered"),
        "row_order": row_order,
        "col_order": col_order,
    }


def _normalize_palette(display: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    palette = _ensure_mapping(_coalesce((display, meta), ("palette",), default={}), "palette")
    normalized = {
        "meta_cluster_colors_present": copy.deepcopy(palette.get("meta_cluster_colors_present", [])),
        "meta_cluster_colors_all": copy.deepcopy(palette.get("meta_cluster_colors_all", [])),
    }
    _validate_palette_block(normalized["meta_cluster_colors_present"], "palette.meta_cluster_colors_present")
    _validate_palette_block(normalized["meta_cluster_colors_all"], "palette.meta_cluster_colors_all")
    return normalized


def _normalize_zscore_params(display: Mapping[str, Any], meta: Mapping[str, Any], var_names: list[str]) -> dict[str, Any]:
    payload = _ensure_mapping(_coalesce((display, meta), ("zscore_params",), default=_MISSING), "zscore_params")
    stats_source = payload.get("per_marker", payload.get("stats"))
    if stats_source is _MISSING or stats_source is None:
        raise ValueError("zscore_params.per_marker is required for z-scored X")

    per_marker: dict[str, dict[str, float]] = {}
    if isinstance(stats_source, Mapping):
        missing = [name for name in var_names if name not in stats_source]
        if missing:
            raise ValueError(f"zscore_params.per_marker is missing marker stats for {missing!r}")
        for name in var_names:
            stats = _ensure_mapping(stats_source[name], f"zscore_params.per_marker[{name!r}]")
            if "mean" not in stats or "std" not in stats:
                raise ValueError(f"zscore_params.per_marker[{name!r}] must contain mean and std")
            per_marker[name] = {
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
            }
    else:
        raise ValueError("zscore_params.per_marker must be a mapping keyed by marker id")

    return {
        "method": str(payload.get("method", "unknown")),
        "per_marker": per_marker,
        "clipped": bool(payload.get("clipped", False)),
    }


def _normalize_filters(display: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    filters = _ensure_mapping(_coalesce((display, meta), ("filters",), default={}), "filters")
    return {
        "expr": copy.deepcopy(filters.get("expr")),
        "structured": copy.deepcopy(filters.get("structured")),
        "source": copy.deepcopy(filters.get("source")),
    }


def _normalize_marker_sets(
    display: Mapping[str, Any],
    flowsom: Mapping[str, Any],
    meta: Mapping[str, Any],
    var_names: list[str],
) -> dict[str, list[str]]:
    raw = _ensure_mapping(_coalesce((display, meta), ("marker_sets",), default={}), "marker_sets")
    training = _normalize_name_list(
        raw.get("training", flowsom.get("training_markers", flowsom.get("channels", var_names))),
        "marker_sets.training",
    )
    display_extra = _normalize_name_list(
        raw.get("display_extra", []),
        "marker_sets.display_extra",
        allow_empty=True,
    )
    available = _normalize_name_list(raw.get("available", var_names), "marker_sets.available")
    linkage = _normalize_name_list(raw.get("linkage", training), "marker_sets.linkage")
    expanded_training = _normalize_name_list(
        raw.get("expanded_training", _unique(training + display_extra)),
        "marker_sets.expanded_training",
    )
    panel = _normalize_name_list(raw.get("panel", var_names), "marker_sets.panel")
    return {
        "training": training,
        "display_extra": display_extra,
        "available": available,
        "linkage": linkage,
        "expanded_training": expanded_training,
        "panel": panel,
    }


def _normalize_flowsom(flowsom: Mapping[str, Any], marker_sets: Mapping[str, list[str]]) -> dict[str, Any]:
    params = _ensure_mapping(flowsom.get("params", {}), "flowsom.params")
    grid = _ensure_mapping(flowsom.get("grid", {}), "flowsom.grid")
    return {
        "training_markers": _normalize_name_list(
            flowsom.get("training_markers", marker_sets["training"]),
            "flowsom.training_markers",
        ),
        "imputation": copy.deepcopy(_ensure_mapping(flowsom.get("imputation", {}), "flowsom.imputation")),
        "projection": copy.deepcopy(_ensure_mapping(flowsom.get("projection", {}), "flowsom.projection")),
        "availability": copy.deepcopy(_ensure_mapping(flowsom.get("availability", {}), "flowsom.availability")),
        "seed": int(flowsom.get("seed", params.get("seed", 0))),
        "grid": {
            "xdim": int(grid.get("xdim", params.get("xdim", 0))),
            "ydim": int(grid.get("ydim", params.get("ydim", 0))),
            "rlen": int(grid.get("rlen", params.get("rlen", 0))),
        },
        "params": copy.deepcopy(dict(params)),
        "deps": list(flowsom.get("deps", [])),
        "hashes": copy.deepcopy(_ensure_mapping(flowsom.get("hashes", {}), "flowsom.hashes")),
    }


def _normalize_row_linkage_basis(
    display: Mapping[str, Any],
    meta: Mapping[str, Any],
    marker_sets: Mapping[str, list[str]],
) -> dict[str, Any]:
    basis = _ensure_mapping(_coalesce((display, meta), ("row_linkage_basis",), default={}), "row_linkage_basis")
    return {
        "marker_ids": _normalize_name_list(
            basis.get("marker_ids", marker_sets["linkage"]),
            "row_linkage_basis.marker_ids",
        ),
        "distance": copy.deepcopy(basis.get("distance")),
    }


def _normalize_row_linkage(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            raise ValueError("row_linkage must be a 2D linkage matrix")
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return copy.deepcopy(list(value))
    raise ValueError("row_linkage must be a list-like value")


def _normalize_checkpoint(meta: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _ensure_mapping(_coalesce((meta,), ("checkpoint",), default={}), "checkpoint")
    created_at = checkpoint.get("created_at")
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    normalized = {
        "id": str(checkpoint.get("id", _uuid7_like())),
        "parents": [str(value) for value in checkpoint.get("parents", [])],
        "op": str(checkpoint.get("op", meta.get("op", "serialize_heatmap_state"))),
        "description": str(checkpoint.get("description", meta.get("description", ""))),
        "created_at": str(created_at),
        "producer": copy.deepcopy(checkpoint.get("producer", meta.get("producer", {"name": "ueler"}))),
        "id_namespace": str(checkpoint.get("id_namespace", meta.get("id_namespace", "ueler.cell_annotation"))),
    }
    if "step_id" in checkpoint or "step_id" in meta:
        normalized["step_id"] = str(checkpoint.get("step_id", meta.get("step_id")))
    return normalized


def _refresh_checksums(adata: ad.AnnData) -> None:
    checksums = _ensure_mapping(adata.uns["artifact"]["checksums"], "artifact.checksums")
    checksums["x_sha256"] = _hash_array(np.asarray(adata.X, dtype=np.float32))
    checksums["row_order_sha256"] = _hash_json(adata.uns["ui"]["row_order"])
    checksums["col_order_sha256"] = _hash_json(adata.uns["ui"]["col_order"])
    if "median" in adata.layers:
        checksums["median_sha256"] = _hash_array(np.asarray(adata.layers["median"], dtype=np.float32))
    checksums.setdefault("h5ad_sha256", ZERO_SHA256)


def _to_matrix(value: Any, *, shape: tuple[int, int], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D matrix")
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"{label} has shape {tuple(array.shape)!r}, expected {shape!r}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _to_embedding(value: Any, *, n_obs: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D matrix")
    if array.shape[0] != n_obs:
        raise ValueError(f"{label} must have {n_obs} rows")
    return np.ascontiguousarray(array, dtype=np.float32)


def _normalize_name_list(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = list(value)
    else:
        raise ValueError(f"{label} must be a sequence of strings")
    normalized = [str(item) for item in values]
    if not normalized and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _normalize_order(value: Any, *, universe: list[str], label: str) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        values = list(value)
    else:
        raise ValueError(f"{label} must be a sequence")
    if values and all(isinstance(item, (int, np.integer)) for item in values):
        try:
            ordered = [universe[int(index)] for index in values]
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{label} contains invalid positional indices") from exc
        _validate_order(ordered, universe, label)
        return ordered
    ordered = [str(item) for item in values]
    _validate_order(ordered, universe, label)
    return ordered


def _validate_order(value: Any, universe: list[str], label: str) -> None:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{label} must be a sequence")
    normalized = [str(item) for item in value]
    if sorted(normalized) != sorted(universe):
        raise ValueError(f"{label} must be a permutation of the corresponding axis names")


def _validate_palette_block(value: Any, label: str) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        entries = value.values()
    elif isinstance(value, (list, tuple)):
        entries = value
    else:
        raise ValueError(f"{label} must be a mapping or sequence of colors")
    for color in entries:
        if color is None:
            continue
        if not isinstance(color, str) or not _HEX_COLOR_RE.fullmatch(color):
            raise ValueError(f"{label} contains invalid hex color {color!r}")


def _coalesce(mappings: tuple[Mapping[str, Any], ...], keys: tuple[str, ...], default: Any = _MISSING) -> Any:
    for mapping in mappings:
        for key in keys:
            value = _lookup(mapping, key)
            if value is not _MISSING:
                return value
    if default is not _MISSING:
        return default
    raise ValueError(f"missing required value; looked for {keys!r}")


def _lookup(mapping: Mapping[str, Any], key: str) -> Any:
    current: Any = mapping
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _ensure_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if isinstance(value, dict):
        return value
    return dict(value)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hash_array(array: np.ndarray) -> str:
    payload = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(payload.dtype).encode("utf-8"))
    digest.update(json.dumps(payload.shape).encode("utf-8"))
    digest.update(payload.tobytes(order="C"))
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalized_h5ad_sha256(path: str | Path, embedded_value: str) -> str:
    raw = Path(path).read_bytes()
    token = embedded_value.encode("ascii")
    if token not in raw:
        raise ValueError("unable to locate embedded h5ad checksum marker in artifact bytes")
    normalized = raw.replace(token, ZERO_SHA256.encode("ascii"), 1)
    return hashlib.sha256(normalized).hexdigest()


def _uuid7_like() -> str:
    timestamp_ms = int(time.time() * 1000)
    time_high = timestamp_ms & ((1 << 48) - 1)
    random_bits = uuid.uuid4().int & ((1 << 74) - 1)
    value = (time_high << 80) | (0x7 << 76) | (((random_bits >> 62) & 0x0FFF) << 64) | (0x2 << 62) | (random_bits & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))
