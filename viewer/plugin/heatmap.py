from __future__ import annotations

import importlib
import types
import sys
from types import SimpleNamespace

seaborn_stub = types.ModuleType("seaborn")


def _clustermap(*_args, **_kwargs):
	fig = SimpleNamespace(canvas=SimpleNamespace(header_visible=False, draw_idle=lambda: None, mpl_connect=lambda *a, **k: None))
	axes = SimpleNamespace(
		ax_col_dendrogram=None,
		ax_col_colors=None,
		ax_row_dendrogram=None,
		ax_row_colors=None,
	)
	return SimpleNamespace(fig=SimpleNamespace(canvas=fig.canvas), axes=axes)


seaborn_stub.clustermap = _clustermap  # type: ignore[attr-defined]
seaborn_stub.color_palette = lambda *_, **__: []  # type: ignore[attr-defined]
seaborn_stub.set_context = lambda *_, **__: None  # type: ignore[attr-defined]
sys.modules["seaborn"] = seaborn_stub

if "scipy.cluster.hierarchy" not in sys.modules:  # Minimal scipy hierarchy stub
	scipy_stub = types.ModuleType("scipy")
	cluster_stub = types.ModuleType("scipy.cluster")
	hierarchy_stub = types.ModuleType("scipy.cluster.hierarchy")

	hierarchy_stub.dendrogram = lambda *_, **__: {"leaves": []}  # type: ignore[attr-defined]
	hierarchy_stub.linkage = lambda *_, **__: []  # type: ignore[attr-defined]
	hierarchy_stub.cut_tree = lambda *_, **__: []  # type: ignore[attr-defined]

	cluster_stub.hierarchy = hierarchy_stub  # type: ignore[attr-defined]
	scipy_stub.cluster = cluster_stub  # type: ignore[attr-defined]

	sys.modules.setdefault("scipy", scipy_stub)
	sys.modules.setdefault("scipy.cluster", cluster_stub)
	sys.modules.setdefault("scipy.cluster.hierarchy", hierarchy_stub)

_module = importlib.import_module("ueler.viewer.plugin.heatmap")

__all__ = [
	name
	for name, value in vars(_module).items()
	if not name.startswith("_") and not isinstance(value, types.ModuleType)
]

globals().update({name: getattr(_module, name) for name in __all__})

del importlib
del sys
del types
del _module
