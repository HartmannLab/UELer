# viewer/__init__.py

__all__ = ["ImageMaskViewer", "create_widgets", "display_ui"]


def __getattr__(name):
	if name == "ImageMaskViewer":
		from .main_viewer import ImageMaskViewer

		return ImageMaskViewer
	if name in {"create_widgets", "display_ui"}:
		from .ui_components import create_widgets, display_ui

		mapping = {
			"create_widgets": create_widgets,
			"display_ui": display_ui,
		}
		return mapping[name]
	raise AttributeError(f"module 'viewer' has no attribute '{name}'")


def __dir__():
	return sorted(set(globals().keys()) | set(__all__))

