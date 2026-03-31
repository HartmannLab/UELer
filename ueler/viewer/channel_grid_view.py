# viewer/channel_grid_view.py
"""GridChannelDisplay — multi-pane channel grid for ImageMaskViewer.

Each selected-and-visible channel is rendered as a separate matplotlib subplot.
All sub-axes share x/y limits (``sharex=True, sharey=True``), so pan/zoom in
any pane automatically propagates to all others.  The primary axes is kept in
sync with ``viewer.image_display.ax`` so that the existing viewport helpers
(``get_axis_limits_with_padding``) continue to work without modification.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from ueler.image_utils import calculate_downsample_factor


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

        # Create one imshow artist per channel pane
        self.img_artists = []
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

        # Hide empty axes (when n doesn't fill the grid perfectly)
        for j in range(n, len(self.axes)):
            self.axes[j].set_visible(False)

        # Apply initial viewport
        if self.axes:
            self.axes[0].set_xlim(xlim)
            self.axes[0].set_ylim(ylim)

        # Internal state for change detection
        self._prev_cx: float | None = None
        self._prev_cy: float | None = None

        # Connect draw event — fires after every pan/zoom
        self.fig.canvas.callbacks.connect("draw_event", self._on_draw)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_viewport(self):
        """Return ``(xlim, ylim)`` of the primary axes."""
        ax = self.axes[0]
        return ax.get_xlim(), ax.get_ylim()

    def update_panes(self, arrays_by_channel: dict, region_xy) -> None:
        """Push freshly rendered arrays into each pane's imshow artist.

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

        for i, ch in enumerate(self.channels):
            arr = arrays_by_channel.get(ch)
            if arr is not None:
                self.img_artists[i].set_data(arr)
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
