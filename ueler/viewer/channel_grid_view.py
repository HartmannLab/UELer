# viewer/channel_grid_view.py
"""GridChannelDisplay — multi-pane channel grid for ImageMaskViewer.

Each selected-and-visible channel is rendered as a separate matplotlib subplot.
All sub-axes share x/y limits (``sharex=True, sharey=True``), so pan/zoom in
any pane automatically propagates to all others.  The primary axes is kept in
sync with ``viewer.image_display.ax`` so that the existing viewport helpers
(``get_axis_limits_with_padding``) continue to work without modification.

Mouse interactivity (cell selection and hover tooltip) mirrors
``ImageDisplay``: clicks toggle mask-selection highlights across all panes,
and hovering shows a debounced tooltip in the active pane.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

from ueler.image_utils import calculate_downsample_factor, get_axis_limits_with_padding
from ueler.viewer.tooltip_utils import format_tooltip_value


class GridChannelDisplay:
    """Multi-pane channel grid backed by a single matplotlib figure.

    Parameters
    ----------
    main_viewer:
        The owning :class:`~ueler.viewer.main_viewer.ImageMaskViewer` instance.
    channels:
        Ordered list of channel names to display.
    xlim:
        Initial x-axis limits ``(xmin, xmax)`` taken from the main viewer.
    ylim:
        Initial y-axis limits ``(ymin, ymax)`` taken from the main viewer.
    """

    def __init__(self, main_viewer, channels, xlim, ylim):
        self.main_viewer = main_viewer
        self.channels = list(channels)

        n = len(self.channels)
        ncols = max(1, math.ceil(math.sqrt(n)))
        nrows = max(1, math.ceil(n / ncols))

        self.fig, axes_raw = plt.subplots(
            nrows,
            ncols,
            figsize=(3.5 * ncols, 3.5 * nrows),
            sharex=True,
            sharey=True,
        )
        self.fig.canvas.header_visible = False
        self.fig.subplots_adjust(wspace=0.04, hspace=0.04)

        # Normalise to a flat list regardless of shape
        if n == 1:
            self.axes = [axes_raw]
        else:
            self.axes = list(np.array(axes_raw).flatten())

        # Create one imshow artist and one tooltip annotation per channel pane
        self.img_artists = []
        self.pane_annotations = []
        for i, ch in enumerate(self.channels):
            ax = self.axes[i]
            img = ax.imshow(
                np.zeros((1, 1, 3), dtype=np.float32),
                extent=(0, main_viewer.width, 0, main_viewer.height),
                origin="upper",
            )
            ax.axis("off")
            ax.text(
                0.02,
                0.02,
                ch,
                transform=ax.transAxes,
                fontsize=9,
                color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"),
            )
            self.img_artists.append(img)
            # Tooltip annotation (hidden by default)
            ann = ax.annotate(
                "",
                xy=(0, 0),
                xycoords="data",
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=10,
                color="yellow",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="yellow", lw=1),
                arrowprops=dict(arrowstyle="->"),
                visible=False,
            )
            self.pane_annotations.append(ann)

        # Hide empty axes (when n doesn't fill the grid perfectly)
        for j in range(n, len(self.axes)):
            self.axes[j].set_visible(False)

        # Apply initial viewport
        if self.axes:
            self.axes[0].set_xlim(xlim)
            self.axes[0].set_ylim(ylim)

        # Stored (clean) arrays — used to restore panes after removing a selection highlight
        self._stored_arrays: dict = {}

        # Set of axis object-ids for fast pane membership test
        self._axes_set: set = {id(ax) for ax in self.axes[:n]}

        # Hover debounce state
        self._hover_timer = None
        self._last_hover_event = None
        self._hover_pane_idx: int = 0

        # Tooltip cache (avoids redundant cell-table lookups)
        self._cached_tooltip_key = None
        self._cached_tooltip_row = None

        # Internal state for draw-change detection
        self._prev_cx: float | None = None
        self._prev_cy: float | None = None

        # Connect events
        self.fig.canvas.callbacks.connect("draw_event", self._on_draw)
        self._setup_grid_events()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_viewport(self):
        """Return ``(xlim, ylim)`` of the primary axes."""
        ax = self.axes[0]
        return ax.get_xlim(), ax.get_ylim()

    def update_panes(self, arrays_by_channel: dict, region_xy) -> None:
        """Push freshly rendered arrays into each pane's imshow artist.

        Also stores a copy of each array so that selection highlights can be
        removed cleanly without triggering a full re-render.

        Parameters
        ----------
        arrays_by_channel:
            Mapping ``{channel_name: np.ndarray}`` of shape ``(H, W, 3)``.
        region_xy:
            ``(xmin, xmax, ymin, ymax)`` in full-resolution pixel coordinates,
            used to update the imshow extent so the image is positioned
            correctly after a pan/zoom.
        """
        xmin, xmax, ymin, ymax = region_xy
        extent = (xmin, xmax, ymax, ymin)  # origin='upper' convention

        new_stored: dict = {}
        for i, ch in enumerate(self.channels):
            arr = arrays_by_channel.get(ch)
            if arr is not None:
                self.img_artists[i].set_data(arr)
                self.img_artists[i].set_extent(extent)
                new_stored[ch] = arr.copy()

        self._stored_arrays = new_stored
        self.fig.canvas.draw_idle()

    def clear_patches(self) -> None:
        """Clear all mask-selection highlights (delegates to shared selection state)."""
        self.main_viewer.image_display.selected_masks_label.clear()
        self._update_grid_patches()

    # ------------------------------------------------------------------
    # Mouse event setup
    # ------------------------------------------------------------------

    def _setup_grid_events(self) -> None:
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_click)

    # ------------------------------------------------------------------
    # Hover tooltip
    # ------------------------------------------------------------------

    def _on_mouse_move(self, event) -> None:
        if id(event.inaxes) not in self._axes_set:
            # Cursor left all channel panes — hide all tooltips
            for ann in self.pane_annotations:
                ann.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            for ann in self.pane_annotations:
                ann.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        # Cancel existing debounce timer
        if self._hover_timer is not None:
            try:
                self._hover_timer.stop()
            except Exception:
                pass

        self._last_hover_event = event
        try:
            self._hover_pane_idx = self.axes.index(event.inaxes)
        except ValueError:
            self._hover_pane_idx = 0

        self._hover_timer = self.fig.canvas.new_timer(interval=300)
        self._hover_timer.single_shot = True
        self._hover_timer.add_callback(self._process_hover)
        self._hover_timer.start()

    def _process_hover(self) -> None:
        event = self._last_hover_event
        if event is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            for ann in self.pane_annotations:
                ann.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        hit = self.main_viewer.resolve_mask_hit_at_viewport(x, y)

        # Always hide all tooltips first (only one pane shows tooltip at a time)
        for ann in self.pane_annotations:
            ann.set_visible(False)

        if hit is None:
            self.fig.canvas.draw_idle()
            return

        # Cache cell-table lookup (avoid redundant lookups on repeated hover)
        lookup_key = (hit.fov_name, hit.mask_name, hit.mask_id)
        if lookup_key != self._cached_tooltip_key:
            cell_row = hit.cell_record
            self._cached_tooltip_row = cell_row
            self._cached_tooltip_key = lookup_key
        else:
            cell_row = self._cached_tooltip_row

        mask_label = hit.mask_name or "Mask"
        tooltip_lines = [f"{mask_label} ID: {hit.mask_id}"]

        if cell_row is not None:
            channel_selector = getattr(self.main_viewer.ui_component, "channel_selector", None)
            selected_channels = getattr(channel_selector, "value", ()) if channel_selector else ()
            for ch in selected_channels or ():
                if ch in cell_row.index:
                    tooltip_lines.append(f"{ch}: {format_tooltip_value(cell_row[ch])}")
            for label in getattr(self.main_viewer, "selected_tooltip_labels", ()):
                if label in cell_row.index:
                    tooltip_lines.append(f"{label}: {format_tooltip_value(cell_row[label])}")

        tooltip_text = "\n".join(tooltip_lines)

        pane_idx = min(self._hover_pane_idx, len(self.pane_annotations) - 1)
        ann = self.pane_annotations[pane_idx]
        ann.xy = (x, y)
        ann.set_text(tooltip_text)
        ann.set_visible(True)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Cell selection click
    # ------------------------------------------------------------------

    def _on_mouse_click(self, event) -> None:
        if id(event.inaxes) not in self._axes_set:
            return
        if self.fig.canvas.toolbar is not None and self.fig.canvas.toolbar.mode != "":
            return  # navigation tool active — ignore

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        hit = self.main_viewer.resolve_mask_hit_at_viewport(x, y)

        # Import here to avoid circular issues at module load time
        from ueler.viewer.image_display import MaskSelection

        # Use the shared selection set on image_display so selections survive mode-switches
        selected = self.main_viewer.image_display.selected_masks_label

        if event.button == MouseButton.LEFT:
            multi_select = getattr(event, "key", None) == "control"
            if not multi_select:
                selected.clear()
            if hit is None:
                self._update_grid_patches()
                return
            sel = MaskSelection(
                fov=str(hit.fov_name),
                mask=str(hit.mask_name),
                mask_id=int(hit.mask_id),
            )
            if sel in selected:
                selected.discard(sel)
            else:
                selected.add(sel)
            self._update_grid_patches(do_not_reset=multi_select)
        elif event.button in {MouseButton.RIGHT, 3}:
            self.clear_patches()

    # ------------------------------------------------------------------
    # Edge-highlight overlay (mirrors ImageDisplay.update_patches)
    # ------------------------------------------------------------------

    def _update_grid_patches(self, do_not_reset: bool = False) -> None:
        """Apply (or remove) the white-edge selection highlight on all panes."""
        from skimage.segmentation import find_boundaries
        from ueler.rendering.engine import scale_outline_thickness, thicken_outline

        viewer = self.main_viewer
        downsample_factor = viewer.current_downsample_factor

        xmin, xmax, ymin, ymax, xmin_ds, xmax_ds, ymin_ds, ymax_ds = get_axis_limits_with_padding(
            viewer, downsample_factor
        )

        selector = getattr(viewer.ui_component, "image_selector", None)
        current_fov = selector.value if selector is not None else None
        selected = viewer.image_display.selected_masks_label
        selections = [sel for sel in selected if sel.fov == current_fov]

        if not selections:
            # Restore clean stored arrays to every pane
            xmin_r, xmax_r, ymin_r, ymax_r = xmin, xmax, ymin, ymax
            extent = (xmin_r, xmax_r, ymax_r, ymin_r)
            for i, ch in enumerate(self.channels):
                arr = self._stored_arrays.get(ch)
                if arr is not None:
                    self.img_artists[i].set_data(arr)
                    self.img_artists[i].set_extent(extent)
            self.fig.canvas.draw_idle()
            return

        # Build the edge mask from full-resolution label masks
        selected_mask_visible_ds = None
        for mask_name, label_mask_full in viewer.full_resolution_label_masks.items():
            matching_ids = {sel.mask_id for sel in selections if sel.mask == mask_name}
            if not matching_ids:
                continue
            try:
                mask_slice = label_mask_full[ymin:ymax:downsample_factor, xmin:xmax:downsample_factor].compute()
            except AttributeError:
                mask_slice = np.asarray(
                    label_mask_full[ymin:ymax:downsample_factor, xmin:xmax:downsample_factor]
                )
            if selected_mask_visible_ds is None:
                selected_mask_visible_ds = np.zeros_like(mask_slice)
            for mask_id in matching_ids:
                selected_mask_visible_ds[mask_slice == mask_id] = mask_id

        if selected_mask_visible_ds is None:
            self.fig.canvas.draw_idle()
            return

        outline_thickness = scale_outline_thickness(
            getattr(viewer, "mask_outline_thickness", 1),
            downsample_factor,
        )
        edge_mask = find_boundaries(selected_mask_visible_ds.astype(np.uint8), mode="inner")
        if outline_thickness > 1:
            edge_mask = thicken_outline(edge_mask, outline_thickness - 1)

        rows, cols = np.nonzero(edge_mask)

        extent = (xmin, xmax, ymax, ymin)
        for i, ch in enumerate(self.channels):
            base = self._stored_arrays.get(ch)
            if base is None:
                continue
            combined = np.array(base, copy=True)
            h, w = combined.shape[:2]

            region_h = max(0, int(ymax_ds - ymin_ds))
            region_w = max(0, int(xmax_ds - xmin_ds))

            if h == region_h and w == region_w:
                mapped_rows, mapped_cols = rows, cols
            else:
                mapped_rows = rows + int(ymin_ds)
                mapped_cols = cols + int(xmin_ds)

            valid = (
                (mapped_rows >= 0)
                & (mapped_rows < h)
                & (mapped_cols >= 0)
                & (mapped_cols < w)
            )
            mr = mapped_rows[valid]
            mc = mapped_cols[valid]
            if mr.size > 0:
                combined[mr, mc] = [1.0, 1.0, 1.0]

            self.img_artists[i].set_data(combined)
            self.img_artists[i].set_extent(extent)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal: draw-event handler
    # ------------------------------------------------------------------

    def _on_draw(self, event):
        """Triggered after each matplotlib draw (pan/zoom/resize).

        Syncs ``image_display.ax`` limits so existing padding helpers work,
        then asks the viewer to re-render all channel panes.
        """
        ax = self.axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2

        # Skip if the viewport has not actually changed
        if self._prev_cx is not None and self._prev_cy is not None:
            if math.isclose(cx, self._prev_cx) and math.isclose(cy, self._prev_cy):
                return

        self._prev_cx = cx
        self._prev_cy = cy

        if not getattr(self.main_viewer, "initialized", False):
            return

        # Keep image_display.ax in sync so get_axis_limits_with_padding works
        self.main_viewer.image_display.ax.set_xlim(xlim)
        self.main_viewer.image_display.ax.set_ylim(ylim)

        # Compute new downsample factor from the visible range
        range_x = abs(xlim[1] - xlim[0])
        range_y = abs(ylim[1] - ylim[0])
        disable_ds = not getattr(
            self.main_viewer.ui_component, "enable_downsample_checkbox", None
        ) or not self.main_viewer.ui_component.enable_downsample_checkbox.value
        new_factor = calculate_downsample_factor(range_x, range_y, disable_ds)

        if new_factor != self.main_viewer.current_downsample_factor:
            self.main_viewer.on_downsample_factor_changed(new_factor)

        self.main_viewer._update_grid_display(self.main_viewer.current_downsample_factor)

