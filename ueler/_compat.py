"""Compatibility helpers for importing legacy modules through the ``ueler`` namespace."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Dict, Iterable, Mapping

# Alias tables derived from the Task 4 mapping proposal. The dictionaries are
# intentionally flat (module path -> legacy target) so the registration helper
# can iterate without additional structure.
UTILITY_ALIASES: Dict[str, str] = {
	"ueler.constants": "constants",
	"ueler.data_loader": "data_loader",
	"ueler.image_utils": "image_utils",
}

VIEWER_CORE_ALIASES: Dict[str, str] = {
	"ueler.viewer.main_viewer": "viewer.main_viewer",
	"ueler.viewer.ui_components": "viewer.ui_components",
	"ueler.viewer.annotation_display": "viewer.annotation_display",
	"ueler.viewer.image_display": "viewer.image_display",
	"ueler.viewer.roi_manager": "viewer.roi_manager",
	"ueler.viewer.color_palettes": "viewer.color_palettes",
}

VIEWER_PLUGIN_ALIASES: Dict[str, str] = {
	"ueler.viewer.plugin": "viewer.plugin",
	"ueler.viewer.plugin.plugin_base": "viewer.plugin.plugin_base",
	"ueler.viewer.plugin.chart": "viewer.plugin.chart",
	"ueler.viewer.plugin.chart_heatmap": "viewer.plugin.chart_heatmap",
	"ueler.viewer.plugin.heatmap": "viewer.plugin.heatmap",
	"ueler.viewer.plugin.heatmap_layers": "viewer.plugin.heatmap_layers",
	"ueler.viewer.plugin.heatmap_adapter": "viewer.plugin.heatmap_adapter",
	"ueler.viewer.plugin.scatter_widget": "viewer.plugin.scatter_widget",
	"ueler.viewer.plugin.cell_gallery": "viewer.plugin.cell_gallery",
	"ueler.viewer.plugin.export_fovs": "viewer.plugin.export_fovs",
	"ueler.viewer.plugin.mask_painter": "viewer.plugin.mask_painter",
	"ueler.viewer.plugin.region_annotation": "viewer.plugin.region_annotation",
	"ueler.viewer.plugin.roi_manager_plugin": "viewer.plugin.roi_manager_plugin",
	"ueler.viewer.plugin.run_flowsom": "viewer.plugin.run_flowsom",
	"ueler.viewer.plugin.go_to": "viewer.plugin.go_to",
}

LEGACY_VIEWER_ALIASES: Dict[str, str] = {
	"viewer.ui_components": "ueler.viewer.ui_components",
	"viewer.color_palettes": "ueler.viewer.color_palettes",
	"viewer.decorators": "ueler.viewer.decorators",
	"viewer.observable": "ueler.viewer.observable",
	"viewer.annotation_palette_editor": "ueler.viewer.annotation_palette_editor",
	"viewer.annotation_display": "ueler.viewer.annotation_display",
}

COMPAT_ALIAS_GROUPS: Iterable[Mapping[str, str]] = (
	UTILITY_ALIASES,
	VIEWER_CORE_ALIASES,
	VIEWER_PLUGIN_ALIASES,
	LEGACY_VIEWER_ALIASES,
)

# Flattened view that tests can import for validation.
SHIM_ALIAS_MAP: Dict[str, str] = {}
for _group in COMPAT_ALIAS_GROUPS:
	SHIM_ALIAS_MAP.update(_group)


class _AliasModuleFinder(MetaPathFinder, Loader):
	"""Meta path finder / loader that resolves alias modules lazily."""

	def __init__(self) -> None:
		self._aliases: Dict[str, str] = {}

	def add_aliases(self, aliases: Mapping[str, str]) -> None:
		self._aliases.update(aliases)

	def has_alias(self, name: str) -> bool:
		return name in self._aliases

	@property
	def aliases(self) -> Mapping[str, str]:
		return dict(self._aliases)

	# MetaPathFinder API -------------------------------------------------
	def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
		target_name = self._aliases.get(fullname)
		if target_name is None:
			return None

		try:
			target_spec = importlib.util.find_spec(target_name)
		except (ModuleNotFoundError, ValueError):
			target_spec = None

		module_stub = sys.modules.get(target_name)
		if target_spec is None and module_stub is None:
			raise ModuleNotFoundError(
				f"Cannot locate target module '{target_name}' for alias '{fullname}'"
			)

		if target_spec is not None:
			is_package = target_spec.submodule_search_locations is not None
			spec = ModuleSpec(fullname, self, is_package=is_package)
			spec.origin = target_spec.origin
			spec.has_location = target_spec.has_location
			if is_package and target_spec.submodule_search_locations is not None:
				spec.submodule_search_locations = list(target_spec.submodule_search_locations)
			return spec

		# Fallback for stub modules injected directly into sys.modules without a
		# populated ModuleSpec (common in the fast-stub test environment).
		is_package = hasattr(module_stub, "__path__")
		spec = ModuleSpec(fullname, self, is_package=is_package)
		spec.origin = getattr(module_stub, "__file__", None)
		spec.has_location = spec.origin is not None
		if is_package:
			submodule_locations = list(getattr(module_stub, "__path__", []))
			spec.submodule_search_locations = submodule_locations
		return spec

	# Loader API ---------------------------------------------------------
	def create_module(self, spec):  # type: ignore[override]
		return None  # default module creation

	def exec_module(self, module):  # type: ignore[override]
		alias = module.__spec__.name  # type: ignore[union-attr]
		target_name = self._aliases[alias]
		target_module = importlib.import_module(target_name)
		sys.modules[alias] = target_module


_ALIAS_FINDER: _AliasModuleFinder | None = None


def register_module_aliases(aliases: Mapping[str, str]) -> Mapping[str, str]:
	"""Register alias modules if they are not already defined.

	Parameters
	----------
	aliases:
		Mapping of `alias -> target` module paths.

	Returns
	-------
	Mapping[str, str]
		Subset of aliases that were newly registered during this call.
	"""

	if not aliases:
		return {}

	global _ALIAS_FINDER
	if _ALIAS_FINDER is None:
		_ALIAS_FINDER = _AliasModuleFinder()
		sys.meta_path.insert(0, _ALIAS_FINDER)

	new_aliases: Dict[str, str] = {}
	for alias, target in aliases.items():
		if alias in sys.modules:
			continue
		if _ALIAS_FINDER.has_alias(alias):
			continue
		# If the alias already has a concrete implementation (e.g., once modules
		# migrate into the package) we skip registering the shim to avoid hiding it.
		try:
			existing_spec = importlib.util.find_spec(alias)
		except ModuleNotFoundError:
			existing_spec = None
		if existing_spec is not None:
			continue
		new_aliases[alias] = target

	if new_aliases:
		_ALIAS_FINDER.add_aliases(new_aliases)

	return new_aliases


def ensure_aliases_loaded() -> None:
	"""Ensure all compatibility aliases are registered."""

	for group in COMPAT_ALIAS_GROUPS:
		register_module_aliases(group)