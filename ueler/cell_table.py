"""AnnData support for the viewer's cell table (issue #123).

The viewer exposes its cell table as ``ImageMaskViewer.cell_table``, and roughly a
hundred call sites across the plugins treat that attribute as a pandas
``DataFrame`` (``.loc``/``.iloc``/``.index``/``select_dtypes``/``groupby``/
``pd.merge``/``.query()``/column assignment).  Rather than emulate that surface,
an ``AnnData`` input is *kept* on the viewer and a DataFrame **view** of it is
derived for the plugins:

* ``obs`` columns become metadata columns,
* ``X`` (or ``layers[layer]``) becomes one column per ``var_names`` entry,
* narrow ``obsm`` arrays become numbered columns (``X_umap1``, ``X_umap2``, …),
* ``obs_names`` is kept as a plain column and the frame gets a ``RangeIndex``.

The last point matters: the plugins mix label-based access (``cell_table.index``)
with positional access (``cell_table.iloc[i]``), which only agree on a
``RangeIndex`` — which is what the CSV path has always produced.

Columns that plugins add or edit afterwards (FlowSOM clusters, heatmap
meta-clusters, the cell-table editor) are pushed back into ``adata.obs`` by
:func:`sync_cell_table_to_obs`, so the user's own object stays authoritative and
can be written out with ``write_h5ad``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CATEGORICAL_DTYPES",
    "DEFAULT_MAX_OBSM_WIDTH",
    "categorical_columns",
    "dataframe_to_anndata",
    "flatten_anndata",
    "is_anndata",
    "new_provenance",
    "sync_cell_table_to_obs",
]

#: ``obsm`` entries wider than this are skipped unless requested explicitly, so
#: a 50-component ``X_pca`` does not flood the marker pickers.
DEFAULT_MAX_OBSM_WIDTH = 3

#: dtypes that count as "class / cluster / grouping" columns.  ``category`` and
#: ``string`` are what an ``.h5ad`` round-trip produces for string columns; a CSV
#: never yields them, so adding them here is behaviour-preserving for the
#: DataFrame path while fixing the AnnData one.
CATEGORICAL_DTYPES: Tuple[str, ...] = ("int", "int64", "object", "category", "string")

_LEGACY_CATEGORICAL_DTYPES: Tuple[str, ...] = ("int", "int64", "object")

_INDEX_COLUMN = "obs_names"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def is_anndata(obj: Any) -> bool:
    """Return ``True`` when *obj* looks like an :class:`anndata.AnnData`.

    Duck-typed on purpose: this keeps ``anndata`` out of the import path of the
    viewer module and lets AnnData-like objects (and lightweight test doubles)
    take the same route.
    """

    if obj is None or isinstance(obj, (pd.DataFrame, pd.Series)):
        return False
    return all(hasattr(obj, attr) for attr in ("obs", "var", "X", "n_obs"))


def new_provenance() -> Dict[str, Any]:
    """Return an empty provenance record (the shape stored on the viewer)."""

    return {
        "obs": [],
        "var": [],
        "obsm": [],
        "obsm_map": {},
        "categorical_obs": [],
        "renamed": {},
        "synced": [],
        "index": None,
        "layer": None,
    }


# ---------------------------------------------------------------------------
# AnnData -> DataFrame
# ---------------------------------------------------------------------------
def _unique_name(name: str, taken: set, suffix: str) -> str:
    """Return *name*, or ``name + suffix`` (numbered) when it is already taken."""

    if name not in taken:
        return name
    candidate = f"{name}{suffix}"
    counter = 2
    while candidate in taken:
        candidate = f"{name}{suffix}{counter}"
        counter += 1
    return candidate


def _decategorise(obs: pd.DataFrame) -> List[str]:
    """Cast ``category`` columns to a plain dtype in place; return their names.

    An ``.h5ad`` round-trip turns every string ``obs`` column into a
    ``Categorical``, and a ``Categorical`` rejects assignment of a value outside
    its categories:

        TypeError: Cannot setitem on a Categorical with a new category

    That is exactly what the Cell Table Editor does
    (``ct.loc[mask, column] = value``), so keeping the dtype would break manual
    labelling on any AnnData table.  The derived frame therefore behaves like a
    CSV-loaded one; ``write_h5ad`` re-categorises strings on the way out, so the
    round-trip is not lossy.  Numeric categories become a real numeric column,
    which is more useful than ``object`` anyway.
    """

    converted: List[str] = []
    for column in list(obs.columns):
        dtype = obs[column].dtype
        if not isinstance(dtype, pd.CategoricalDtype):
            continue
        categories = dtype.categories
        target = (
            categories.dtype
            if pd.api.types.is_numeric_dtype(categories) or pd.api.types.is_bool_dtype(categories)
            else object
        )
        try:
            obs[column] = obs[column].astype(target)
        except Exception:  # pragma: no cover - defensive guard
            obs[column] = obs[column].astype(object)
        converted.append(str(column))
    return converted


def _marker_frame(adata: Any, layer: Optional[str]) -> pd.DataFrame:
    """Return the expression matrix as a DataFrame with ``var_names`` columns."""

    matrix = adata.layers[layer] if layer is not None else getattr(adata, "X", None)
    if matrix is None:
        return pd.DataFrame(index=pd.RangeIndex(int(adata.n_obs)))

    if hasattr(matrix, "toarray"):
        # ``to_df`` densifies; say so up front because the cost is the user's.
        n_obs = int(adata.n_obs)
        n_var = int(getattr(adata, "n_vars", 0))
        logger.warning(
            "Cell table: densifying a sparse expression matrix (%d x %d) to build "
            "the cell-table view.",
            n_obs,
            n_var,
        )

    frame = adata.to_df(layer=layer)
    return frame


def _obsm_frame(
    adata: Any,
    obsm_keys: Optional[Sequence[str]],
    max_obsm_width: int,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Return the (narrow) ``obsm`` entries as numbered columns."""

    obsm = getattr(adata, "obsm", None)
    if not obsm:
        return pd.DataFrame(index=pd.RangeIndex(int(adata.n_obs))), {}

    explicit = obsm_keys is not None
    keys = list(obsm_keys) if explicit else list(obsm.keys())

    data: Dict[str, np.ndarray] = {}
    mapping: Dict[str, List[str]] = {}
    for key in keys:
        if key not in obsm:
            logger.warning("Cell table: obsm key %r not found; skipping.", key)
            continue
        values = np.asarray(obsm[key])
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            logger.warning(
                "Cell table: obsm[%r] has %d dimensions; only 1D/2D entries are "
                "exposed as columns.",
                key,
                values.ndim,
            )
            continue
        width = values.shape[1]
        if not explicit and width > max_obsm_width:
            logger.info(
                "Cell table: skipping obsm[%r] (%d columns > %d); pass "
                "obsm_keys=['%s'] to include it.",
                key,
                width,
                max_obsm_width,
                key,
            )
            continue
        names = [f"{key}{i + 1}" for i in range(width)]
        for i, name in enumerate(names):
            data[name] = values[:, i]
        mapping[key] = names

    frame = pd.DataFrame(data, index=pd.RangeIndex(int(adata.n_obs)))
    return frame, mapping


def flatten_anndata(
    adata: Any,
    *,
    layer: Optional[str] = None,
    obsm_keys: Optional[Sequence[str]] = None,
    max_obsm_width: int = DEFAULT_MAX_OBSM_WIDTH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the DataFrame view of *adata* plus its column provenance.

    Column order is ``obs`` → markers → ``obsm`` → ``obs_names``.  ``obs`` dtypes
    are preserved except for ``category``, which is widened by
    :func:`_decategorise` so the write paths keep working.  Unlike the CSV path
    there is no float→``Int64`` downcast: AnnData already carries correct dtypes
    and coercing marker columns would be wrong.

    On a name clash ``obs`` wins; the marker/``obsm`` column is suffixed and the
    rename is recorded in ``provenance["renamed"]``.
    """

    n_obs = int(adata.n_obs)
    provenance = new_provenance()
    provenance["layer"] = layer

    obs = adata.obs.copy()
    # Re-index positionally rather than aligning on ``obs_names``: AnnData
    # tolerates duplicate obs_names, which would scramble a concat-by-index.
    obs.index = pd.RangeIndex(n_obs)
    decategorised = _decategorise(obs)
    if decategorised:
        logger.debug(
            "Cell table: widened %d categorical obs column(s): %s",
            len(decategorised),
            ", ".join(decategorised),
        )
    provenance["categorical_obs"] = decategorised
    provenance["obs"] = list(obs.columns)
    taken = set(obs.columns)

    markers = _marker_frame(adata, layer)
    markers.index = pd.RangeIndex(n_obs)
    marker_renames: Dict[str, str] = {}
    for name in list(markers.columns):
        target = _unique_name(str(name), taken, "_var")
        if target != name:
            marker_renames[name] = target
        taken.add(target)
    if marker_renames:
        markers = markers.rename(columns=marker_renames)
        provenance["renamed"].update({str(k): v for k, v in marker_renames.items()})
        logger.info(
            "Cell table: renamed %d marker column(s) that clashed with obs: %s",
            len(marker_renames),
            ", ".join(f"{k} -> {v}" for k, v in marker_renames.items()),
        )
    provenance["var"] = [str(col) for col in markers.columns]

    obsm_df, obsm_map = _obsm_frame(adata, obsm_keys, max_obsm_width)
    obsm_renames: Dict[str, str] = {}
    for name in list(obsm_df.columns):
        target = _unique_name(str(name), taken, "_obsm")
        if target != name:
            obsm_renames[name] = target
        taken.add(target)
    if obsm_renames:
        obsm_df = obsm_df.rename(columns=obsm_renames)
        provenance["renamed"].update({str(k): v for k, v in obsm_renames.items()})
        obsm_map = {
            key: [obsm_renames.get(col, col) for col in cols]
            for key, cols in obsm_map.items()
        }
    provenance["obsm"] = [str(col) for col in obsm_df.columns]
    provenance["obsm_map"] = obsm_map

    frames = [frame for frame in (obs, markers, obsm_df) if len(frame.columns)]
    if frames:
        cell_table = pd.concat(frames, axis=1)
    else:
        cell_table = pd.DataFrame(index=pd.RangeIndex(n_obs))

    index_column = _unique_name(_INDEX_COLUMN, taken, "_index")
    cell_table[index_column] = pd.Index(adata.obs_names).astype(str).to_numpy()
    provenance["index"] = index_column

    cell_table.index = pd.RangeIndex(n_obs)
    return cell_table, provenance


# ---------------------------------------------------------------------------
# DataFrame -> AnnData
# ---------------------------------------------------------------------------
def _values_equal(left: pd.Series, right: pd.Series) -> bool:
    """Compare two same-length columns ignoring index *and* dtype.

    Ignoring dtype matters: :func:`_decategorise` widens ``category`` obs columns
    in the derived frame, and an untouched column must still compare equal so the
    sync leaves the user's AnnData dtypes alone.
    """

    try:
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        if left.dtype != right.dtype:
            left = left.astype(object)
            right = right.astype(object)
        return left.equals(right)
    except Exception:  # pragma: no cover - defensive guard
        return False


def sync_cell_table_to_obs(
    adata: Any,
    cell_table: Any,
    provenance: Optional[Dict[str, Any]],
) -> List[str]:
    """Push metadata columns of *cell_table* back into ``adata.obs``.

    Returns the list of columns that were written.  Marker/``obsm``/index columns
    are skipped — they are views of ``X``/``obsm`` and are never edited.  The sync
    is positional, so it no-ops (with a warning) when the frame's row count no
    longer matches ``adata.n_obs``; that happens if a plugin replaces the table
    with a ``pd.merge`` result of different cardinality.
    """

    if adata is None or cell_table is None or not isinstance(provenance, dict):
        return []

    columns = getattr(cell_table, "columns", None)
    if columns is None:
        return []

    n_obs = int(getattr(adata, "n_obs", 0))
    if len(cell_table) != n_obs:
        logger.warning(
            "Cell table: not syncing back to AnnData — the table now has %d rows "
            "but the AnnData has %d.",
            len(cell_table),
            n_obs,
        )
        return []

    derived = set(provenance.get("var", ())) | set(provenance.get("obsm", ()))
    index_column = provenance.get("index")
    original_obs = set(provenance.get("obs", ()))
    previously_synced = set(provenance.get("synced", ()))

    current = [
        str(col) for col in columns if str(col) not in derived and str(col) != index_column
    ]

    written: List[str] = []
    for column in current:
        series = cell_table[column]
        if column in adata.obs.columns and _values_equal(series, adata.obs[column]):
            continue
        try:
            # ``.values`` keeps extension dtypes (Int64, Categorical) intact and
            # assigns positionally, so duplicate obs_names cannot misalign it.
            adata.obs[column] = series.values
        except Exception:
            logger.warning(
                "Cell table: could not sync column %r into adata.obs.", column, exc_info=True
            )
            continue
        written.append(column)

    # Columns we added earlier and that have since been dropped from the table
    # should not linger in obs.
    for column in previously_synced - set(current):
        if column not in original_obs and column in adata.obs.columns:
            try:
                adata.obs.drop(columns=[column], inplace=True)
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Cell table: could not drop stale obs column %r.", column)

    provenance["synced"] = [col for col in current if col not in original_obs]
    if written:
        logger.debug("Cell table: synced %d column(s) into adata.obs.", len(written))
    return written


def dataframe_to_anndata(
    cell_table: pd.DataFrame,
    provenance: Optional[Dict[str, Any]] = None,
    *,
    system_keys: Sequence[str] = (),
    source: Any = None,
):
    """Build an :class:`anndata.AnnData` from the cell-table DataFrame.

    With *provenance* (an AnnData-sourced table) the original split is restored:
    ``X`` holds the recorded marker columns under their original ``var_names``,
    ``obs_names`` is restored from the index column, and ``var``/``obsm``/``uns``
    are carried over from *source*.  Without provenance (a CSV-sourced table)
    ``X`` is the numeric columns minus *system_keys* and ``obs`` is the rest.
    """

    import anndata

    provenance = provenance or {}
    renamed = provenance.get("renamed") or {}
    # provenance["renamed"] maps original name -> column name; invert it to put
    # the original var_names back on the way out.
    reverse_renames = {column: original for original, column in renamed.items()}

    frame_columns = [str(col) for col in cell_table.columns]
    if provenance.get("var"):
        marker_columns = [col for col in provenance["var"] if col in frame_columns]
    else:
        system = set(system_keys)
        marker_columns = [
            col
            for col in frame_columns
            if col not in system and pd.api.types.is_numeric_dtype(cell_table[col])
        ]

    obsm_columns = set(provenance.get("obsm", ()))
    index_column = provenance.get("index")
    obs_columns = [
        col
        for col in frame_columns
        if col not in marker_columns and col not in obsm_columns and col != index_column
    ]

    n_obs = len(cell_table)
    if marker_columns:
        X = cell_table[marker_columns].to_numpy(dtype="float32")
    else:
        X = np.zeros((n_obs, 0), dtype="float32")

    obs = cell_table[obs_columns].copy()
    if index_column and index_column in frame_columns:
        obs.index = pd.Index(cell_table[index_column].astype(str))
    else:
        obs.index = pd.Index([str(i) for i in range(n_obs)])

    var_names = [reverse_renames.get(col, col) for col in marker_columns]
    var = pd.DataFrame(index=pd.Index(var_names, dtype=object))
    source_var = getattr(source, "var", None)
    if source_var is not None and len(var_names):
        try:
            var = source_var.reindex(var_names).copy()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Cell table: could not carry over adata.var; using bare var.")

    adata = anndata.AnnData(X=X, obs=obs, var=var)

    for attribute in ("obsm", "uns"):
        payload = getattr(source, attribute, None)
        if not payload:
            continue
        try:
            getattr(adata, attribute).update(dict(payload))
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Cell table: could not carry over adata.%s.", attribute)

    return adata


# ---------------------------------------------------------------------------
# Column-role helpers
# ---------------------------------------------------------------------------
def categorical_columns(cell_table: Any, *, include_bool: bool = False) -> List[str]:
    """Return the class / cluster / grouping columns of *cell_table*.

    This is the single rule behind the cluster, subset-on, identifier and
    tooltip-label dropdowns.  It extends the historical
    ``select_dtypes(['int', 'int64', 'object'])`` with ``category``/``string`` so
    that string columns coming back from an ``.h5ad`` round-trip are offered too;
    ``bool`` stays opt-in because only the mask painter used to include it.
    """

    if cell_table is None:
        return []
    include = list(CATEGORICAL_DTYPES)
    if include_bool:
        include.append("bool")
    try:
        return cell_table.select_dtypes(include=include).columns.tolist()
    except TypeError:
        # Older pandas (or a minimal stand-in) may not know every dtype alias.
        include = list(_LEGACY_CATEGORICAL_DTYPES)
        if include_bool:
            include.append("bool")
        return cell_table.select_dtypes(include=include).columns.tolist()
