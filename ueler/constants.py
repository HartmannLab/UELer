"""Core constants for the UELer viewer package."""

PREDEFINED_COLORS = {
    "Red": "#ff0000",
    "Green": "#00ff00",
    "Blue": "#0000ff",
    "Yellow": "#ffff00",
    "Cyan": "#00ffff",
    "Magenta": "#ff00ff",
    "Orange": "#ffa500",
    "Purple": "#800080",
    "Brown": "#a52a2a",
    "Pink": "#ffc0cb",
    "Gray": "#808080",
    "Black": "#000000",
    "White": "#ffffff",
}

# How a cell selection is drawn (ueler/viewer/selection_style.py). The amber is
# deliberately not one of PREDEFINED_COLORS, so it differs from the default
# white masks and from the pure RGB/CMY channel colours. The width is in pixels
# at full resolution and scaled with the downsample factor; a fill opacity of 0
# turns the tint off. All four are editable under Advanced Settings.
SELECTION_COLOR = "#ffb000"
SELECTION_OUTLINE_WIDTH = 2
SELECTION_FILL_OPACITY = 0.25
# Unselected points and bars in the scatter and histogram.
PLOT_BASE_COLOR = "#1f77b4"
# Colour of a mask with no saved colour of its own.
DEFAULT_MASK_COLOR = "White"

DOWNSAMPLE_FACTORS =[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Longest-edge pixel budget: frames whose largest dimension is <= this value
# render at full resolution; larger frames are downsampled by the smallest
# power-of-two factor that fits within it. (Issue #116)
DOWNSAMPLE_MAX_DIMENSION = 2048

UICOMPNENTS_SKIP = ["marker_set_dropdown", "status_bar"]
