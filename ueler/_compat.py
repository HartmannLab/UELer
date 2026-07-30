"""Compatibility helpers for importing legacy modules through the ``ueler`` namespace."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Dict, Iterable, Mapping

# Top-level utility modules that moved into the package. The dictionary is
# intentionally flat (alias path -> target module) so the registration helper
# can iterate without additional structure.
UTILITY_ALIASES: Dict[str, str] = {
	"constants": "ueler.constants",
	"data_loader": "ueler.data_loader",
	"image_utils": "ueler.image_utils",
}

# The package migration is complete: every module that used to live in the
# top-level ``viewer`` package now lives under ``ueler.viewer``. Rather than
# enumerating each module, the whole legacy namespace is rewritten onto the new
# one so that ``viewer.<anything>`` keeps resolving for existing notebooks. The
# prefix form also guarantees that a legacy import yields the *same* module
# object as its canonical counterpart, which a per-module table cannot do for
# submodules it forgets to list (they would be imported a second time under a
# different name, duplicating classes).
LEGACY_PACKAGE_PREFIXES: Dict[str, str] = {
	"viewer": "ueler.viewer",
}

# Legacy module paths that are explicitly supported (and validated by the shim
# tests). Resolution happens through LEGACY_PACKAGE_PREFIXES, so this table is a
# documented subset of the covered surface rather than the routing itself.
LEGACY_VIEWER_ALIASES: Dict[str, str] = {
	"viewer.ui_components": "ueler.viewer.ui_components",
	"viewer.color_palettes": "ueler.viewer.color_palettes",
	"viewer.decorators": "ueler.viewer.decorators",
	"viewer.observable": "ueler.viewer.observable",
	"viewer.annotation_palette_editor": "ueler.viewer.annotation_palette_editor",
	"viewer.annotation_display": "ueler.viewer.annotation_display",
	"viewer.roi_manager": "ueler.viewer.roi_manager",
	"viewer.main_viewer": "ueler.viewer.main_viewer",
}

LEGACY_PLUGIN_ALIASES: Dict[str, str] = {
	"viewer.plugin.plugin_base": "ueler.viewer.plugin.plugin_base",
	"viewer.plugin.export_fovs": "ueler.viewer.plugin.export_fovs",
	"viewer.plugin.go_to": "ueler.viewer.plugin.go_to",
	"viewer.plugin.cell_gallery": "ueler.viewer.plugin.cell_gallery",
	"viewer.plugin.run_flowsom": "ueler.viewer.plugin.run_flowsom",
}

# Groups routed through the explicit alias table (``_AliasModuleFinder``).
COMPAT_ALIAS_GROUPS: Iterable[Mapping[str, str]] = (
	UTILITY_ALIASES,
)

# Flattened view that tests can import for validation.
SHIM_ALIAS_MAP: Dict[str, str] = {}
for _group in (*COMPAT_ALIAS_GROUPS, LEGACY_VIEWER_ALIASES, LEGACY_PLUGIN_ALIASES):
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


class _PrefixAliasFinder(MetaPathFinder, Loader):
	"""Meta path finder that rewrites a whole legacy namespace onto a new one.

	``viewer.plugin.chart`` is served by importing ``ueler.viewer.plugin.chart``
	and binding the *same* module object under the legacy name, so legacy and
	canonical imports always share state and class identity.
	"""

	def __init__(self) -> None:
		self._prefixes: Dict[str, str] = {}

	def add_prefixes(self, prefixes: Mapping[str, str]) -> None:
		self._prefixes.update(prefixes)

	@property
	def prefixes(self) -> Mapping[str, str]:
		return dict(self._prefixes)

	def resolve(self, fullname: str) -> str | None:
		"""Return the canonical module name for ``fullname``, if it is aliased."""

		for prefix, target_prefix in self._prefixes.items():
			if fullname == prefix:
				return target_prefix
			if fullname.startswith(f"{prefix}."):
				return target_prefix + fullname[len(prefix):]
		return None

	# MetaPathFinder API -------------------------------------------------
	def find_spec(self, fullname: str, path=None, target=None):  # type: ignore[override]
		target_name = self.resolve(fullname)
		if target_name is None:
			return None

		try:
			target_spec = importlib.util.find_spec(target_name)
		except (ImportError, ValueError):
			target_spec = None

		if target_spec is None:
			# Fall back to stub modules injected straight into sys.modules (the
			# fast-stub test environment does this) before giving up.
			module_stub = sys.modules.get(target_name)
			if module_stub is None:
				return None
			is_package = hasattr(module_stub, "__path__")
			spec = ModuleSpec(fullname, self, is_package=is_package)
			spec.origin = getattr(module_stub, "__file__", None)
			spec.has_location = spec.origin is not None
			if is_package:
				spec.submodule_search_locations = list(getattr(module_stub, "__path__", []))
			return spec

		is_package = target_spec.submodule_search_locations is not None
		spec = ModuleSpec(fullname, self, is_package=is_package)
		spec.origin = target_spec.origin
		spec.has_location = target_spec.has_location
		if is_package and target_spec.submodule_search_locations is not None:
			spec.submodule_search_locations = list(target_spec.submodule_search_locations)
		return spec

	# Loader API ---------------------------------------------------------
	def create_module(self, spec):  # type: ignore[override]
		target_name = self.resolve(spec.name)
		if target_name is None:  # pragma: no cover - defensive guard
			return None
		return importlib.import_module(target_name)

	def exec_module(self, module):  # type: ignore[override]
		# ``create_module`` already returned the fully initialised target module;
		# just make the legacy name point at it.
		alias = module.__spec__.name  # type: ignore[union-attr]
		sys.modules[alias] = module


_ALIAS_FINDER: _AliasModuleFinder | None = None
_PREFIX_FINDER: _PrefixAliasFinder | None = None


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


def register_package_prefixes(prefixes: Mapping[str, str]) -> Mapping[str, str]:
	"""Register legacy namespace prefixes (``legacy -> canonical``).

	Prefixes whose canonical namespace cannot be located are skipped, as are
	prefixes shadowed by a real package still present on ``sys.path``.
	"""

	if not prefixes:
		return {}

	global _PREFIX_FINDER
	if _PREFIX_FINDER is None:
		_PREFIX_FINDER = _PrefixAliasFinder()
		# Must precede the standard PathFinder: the aliased parent package shares
		# its ``__path__`` with the canonical one, so PathFinder would happily load
		# ``viewer.main_viewer`` a second time from disk instead of reusing
		# ``ueler.viewer.main_viewer``.
		sys.meta_path.insert(0, _PREFIX_FINDER)

	new_prefixes: Dict[str, str] = {}
	for prefix, target_prefix in prefixes.items():
		if prefix in _PREFIX_FINDER.prefixes:
			continue
		# A concrete implementation of the legacy name wins over the shim.
		try:
			existing_spec = importlib.util.find_spec(prefix)
		except (ImportError, ValueError):
			existing_spec = None
		if existing_spec is not None:
			continue
		new_prefixes[prefix] = target_prefix

	if new_prefixes:
		_PREFIX_FINDER.add_prefixes(new_prefixes)

	return new_prefixes


def ensure_aliases_loaded() -> None:
	"""Ensure all compatibility aliases are registered."""

	for group in COMPAT_ALIAS_GROUPS:
		register_module_aliases(group)
	register_package_prefixes(LEGACY_PACKAGE_PREFIXES)