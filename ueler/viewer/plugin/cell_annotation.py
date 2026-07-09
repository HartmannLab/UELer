"""Cell Annotation plugin — checkpoint save/load browser for heatmap workflows.

Provides a side-panel plugin that lets users:

1. **Save** the current heatmap state (z-scored medians, meta-cluster palette,
   UI settings, optional FlowSOM params) as an ``.h5ad`` checkpoint under
   ``<dataset_root>/.UELer/dataset_<hash>/checkpoints/``.
2. **Browse** saved checkpoints in a parent-child tree view.
3. **Load** a selected checkpoint back into the heatmap plugin to resume a
   previous annotation session.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)
from ipywidgets import (
    Button,
    Dropdown,
    HTML,
    HBox,
    Layout,
    Output,
    Text,
    VBox,
)

from ueler.viewer.plugin.plugin_base import PluginBase

# ------------------------------------------------------------------
# CheckpointTreeWidget — anywidget-based tree renderer
# ------------------------------------------------------------------

try:
    import anywidget
    import traitlets

    class CheckpointTreeWidget(anywidget.AnyWidget):  # type: ignore[misc]
        """Renders a parent-child tree of checkpoints and emits selection/action events."""

        nodes: list = traitlets.List(traitlets.Dict()).tag(sync=True)
        selected_id: str = traitlets.Unicode("").tag(sync=True)
        action_requested: str = traitlets.Unicode("").tag(sync=True)

        _css = """
.ca-tree-root {
    font-family: var(--jp-ui-font-family, sans-serif);
    font-size: var(--jp-ui-font-size1, 12px);
    overflow-y: auto;
    max-height: 320px;
    border: 1px solid var(--jp-border-color2, #ccc);
    border-radius: 3px;
    padding: 4px 0;
    background: var(--jp-layout-color1, #fff);
}
.ca-tree-item {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 4px;
    cursor: pointer;
    user-select: none;
    border-bottom: 1px solid var(--jp-border-color3, #eee);
    transition: background 0.1s;
}
.ca-tree-item:hover {
    background: var(--jp-layout-color2, #f5f5f5);
}
.ca-tree-item.ca-selected {
    background: var(--jp-brand-color3, #bbdefb);
    font-weight: bold;
}
.ca-tree-op {
    display: inline-block;
    padding: 1px 5px;
    border-radius: 8px;
    font-size: 10px;
    color: #fff;
    background: var(--jp-warn-color1, #888);
    flex: 0 0 auto;
    text-transform: uppercase;
}
.ca-tree-op[data-op="initial"]   { background: #4caf50; }
.ca-tree-op[data-op="recluster"] { background: #2196f3; }
.ca-tree-op[data-op="subset"]    { background: #ff9800; }
.ca-tree-op[data-op="finalize"]  { background: #9c27b0; }
.ca-tree-label {
    flex: 1 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ca-tree-meta {
    font-size: 10px;
    color: var(--jp-ui-font-color2, #888);
    flex: 0 0 auto;
    white-space: nowrap;
}
.ca-tree-indent {
    flex: 0 0 auto;
}
"""

        _esm = r"""
function render({ model, el }) {
  var root = document.createElement('div');
  root.className = 'ca-tree-root';

  function buildChildren(nodes) {
    // nodes: list of {id, parent_id, step_id, description, op, n_clusters, created_at}
    var childMap = {};
    var roots = [];
    nodes.forEach(function(n) {
      var pid = n.parent_id || null;
      if (!pid) {
        roots.push(n.id);
      } else {
        if (!childMap[pid]) { childMap[pid] = []; }
        childMap[pid].push(n.id);
      }
    });
    return { childMap: childMap, roots: roots };
  }

  function formatDate(iso) {
    if (!iso) { return ''; }
    try {
      var d = new Date(iso);
      return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
    } catch(e) { return iso.slice(0, 10); }
  }

  function render() {
    root.innerHTML = '';
    var nodes = model.get('nodes') || [];
    var selectedId = model.get('selected_id') || '';
    var byId = {};
    nodes.forEach(function(n) { byId[n.id] = n; });
    var tree = buildChildren(nodes);

    function renderNode(id, depth) {
      var n = byId[id];
      if (!n) { return; }
      var row = document.createElement('div');
      row.className = 'ca-tree-item' + (id === selectedId ? ' ca-selected' : '');
      row.dataset.id = id;

      // Indentation
      var indent = document.createElement('span');
      indent.className = 'ca-tree-indent';
      indent.style.paddingLeft = (depth * 16) + 'px';
      if (depth > 0) {
        indent.textContent = '└ ';
      }
      row.appendChild(indent);

      // Op badge
      var badge = document.createElement('span');
      badge.className = 'ca-tree-op';
      badge.dataset.op = n.op || 'initial';
      badge.textContent = (n.op || 'init').slice(0, 6);
      row.appendChild(badge);

      // Label
      var label = document.createElement('span');
      label.className = 'ca-tree-label';
      var stepText = n.step_id ? '[' + n.step_id + '] ' : '';
      var desc = n.description || '(no description)';
      label.textContent = stepText + desc;
      label.title = stepText + desc + ' — ' + (n.n_clusters || '?') + ' clusters, ' + (n.n_markers || '?') + ' markers';
      row.appendChild(label);

      // Date
      var meta = document.createElement('span');
      meta.className = 'ca-tree-meta';
      meta.textContent = formatDate(n.created_at);
      row.appendChild(meta);

      row.addEventListener('click', function() {
        model.set('selected_id', id);
        model.save_changes();
      });

      root.appendChild(row);

      // Recurse into children
      var children = tree.childMap[id] || [];
      children.forEach(function(childId) {
        renderNode(childId, depth + 1);
      });
    }

    if (nodes.length === 0) {
      var empty = document.createElement('div');
      empty.style.cssText = 'padding:8px;color:var(--jp-ui-font-color2,#888);font-style:italic;';
      empty.textContent = 'No checkpoints saved yet.';
      root.appendChild(empty);
    } else {
      tree.roots.forEach(function(id) { renderNode(id, 0); });
    }
  }

  model.on('change:nodes',       render);
  model.on('change:selected_id', render);
  render();
  el.appendChild(root);
}

export default { render };
"""

except (ImportError, AttributeError):
    # Headless fallback for test environments without anywidget.
    import traitlets  # type: ignore[import]

    class CheckpointTreeWidget(traitlets.HasTraits):  # type: ignore[no-redef]
        """Headless fallback — same traitlet interface, no rendering."""

        nodes: list = traitlets.List(traitlets.Dict())
        selected_id: str = traitlets.Unicode("")
        action_requested: str = traitlets.Unicode("")


# ------------------------------------------------------------------
# CellAnnotationPlugin
# ------------------------------------------------------------------


class CellAnnotationPlugin(PluginBase):
    """Side-panel plugin for saving and loading heatmap annotation checkpoints."""

    def __init__(self, main_viewer, width, height):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "cell_annotation_output"
        self.displayed_name = "Cell Annotation"
        self.main_viewer = main_viewer

        self._store = None          # CheckpointStore — created in after_all_plugins_loaded
        self._heatmap_plugin = None  # wired in after_all_plugins_loaded
        self._flowsom_plugin = None  # wired in after_all_plugins_loaded

        # Save-form widgets
        self.step_id_input = Text(
            value="",
            placeholder="e.g. 1, 2a, 3-CD4",
            description="Step:",
            style={"description_width": "50px"},
            layout=Layout(width="99%"),
        )
        self.description_input = Text(
            value="",
            placeholder="Short description of this checkpoint",
            description="Desc:",
            style={"description_width": "50px"},
            layout=Layout(width="99%"),
        )
        self.parent_dropdown = Dropdown(
            description="Parent:",
            options=[("(none)", None)],
            value=None,
            style={"description_width": "50px"},
            layout=Layout(width="99%"),
        )
        self.op_dropdown = Dropdown(
            description="Op:",
            options=["initial", "subset", "recluster", "finalize"],
            value="initial",
            style={"description_width": "50px"},
            layout=Layout(width="99%"),
        )
        self.save_button = Button(
            description="Save checkpoint",
            button_style="primary",
            icon="save",
            layout=Layout(width="99%"),
        )
        self.save_button.on_click(self._on_save_button)

        # Checkpoint browser
        self.tree_widget = CheckpointTreeWidget()
        self.load_button = Button(
            description="Load selected",
            button_style="",
            icon="download",
            layout=Layout(width="49%"),
        )
        self.delete_button = Button(
            description="Delete selected",
            button_style="danger",
            icon="trash",
            layout=Layout(width="49%"),
        )
        self.load_button.on_click(self._on_load_button)
        self.delete_button.on_click(self._on_delete_button)

        self.status_label = HTML(value="")

        self.plot_output = Output()  # for error / progress messages

        self.initiate_ui()
        self.initialized = True

    def initiate_ui(self):
        save_section = VBox(
            [
                HTML(value="<b>Save checkpoint</b>"),
                self.step_id_input,
                self.description_input,
                HBox([self.parent_dropdown], layout=Layout(width="99%")),
                HBox([self.op_dropdown], layout=Layout(width="99%")),
                self.save_button,
            ],
            layout=Layout(margin="0 0 8px 0"),
        )
        browser_section = VBox(
            [
                HTML(value="<b>Checkpoint browser</b>"),
                self.tree_widget if hasattr(self.tree_widget, "_esm") else HTML(
                    value="<i>(tree not available in this environment)</i>"
                ),
                HBox([self.load_button, self.delete_button]),
            ]
        )
        self.ui = VBox(
            [
                save_section,
                HTML(value="<hr style='margin:4px 0;border-color:var(--jp-border-color2,#ccc)'>"),
                browser_section,
                self.status_label,
                self.plot_output,
            ],
            layout=Layout(overflow="hidden"),
        )

    def after_all_plugins_loaded(self):
        """Wire heatmap/FlowSOM refs and initialise the checkpoint store.

        Does NOT call super() — CellAnnotationPlugin stores widgets directly on
        self (no ui_component), so PluginBase.load_widget_states() would crash
        with AttributeError if a state file ever exists on disk.
        """
        from ueler.viewer.checkpoint_store import CheckpointStore

        base_folder = getattr(self.main_viewer, "base_folder", None)
        if base_folder:
            self._store = CheckpointStore(base_folder)
            self._refresh_tree()

        side = getattr(self.main_viewer, "SidePlots", None)
        self._heatmap_plugin = getattr(side, "heatmap_output", None)
        self._flowsom_plugin = getattr(side, "run_flowsom_output", None)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        *,
        step_id: str = "",
        description: str = "",
        parent_id: "str | None" = None,
        op: str = "initial",
    ) -> "str | None":
        """Export the current heatmap state and persist it as a checkpoint.

        Returns the new checkpoint ID, or ``None`` on failure.
        """
        _logger.debug("[cell_annotation] save_checkpoint: entry (step_id=%r)", step_id)
        if self._store is None:
            self._set_status("Cannot save: no dataset folder configured.", error=True)
            return None
        if self._heatmap_plugin is None:
            self._set_status("Cannot save: Heatmap plugin not loaded.", error=True)
            return None

        _logger.debug("[cell_annotation] save_checkpoint: exporting heatmap state")
        try:
            adata = self._heatmap_plugin.export_heatmap_state()
        except Exception as exc:
            self._set_status(f"Export failed: {exc}", error=True)
            return None
        _logger.debug("[cell_annotation] save_checkpoint: export ok (shape=%s)", adata.shape)

        # Attach FlowSOM params if available
        if self._flowsom_plugin is not None:
            try:
                adata.uns["flowsom"] = self._flowsom_plugin.export_flowsom_params()
                _logger.debug("[cell_annotation] save_checkpoint: FlowSOM params attached")
            except Exception:
                pass  # FlowSOM params are optional

        try:
            ckpt_id = self._store.write_checkpoint(
                adata,
                parent_id=parent_id,
                op=op,
                step_id=step_id,
                description=description,
            )
        except Exception as exc:
            self._set_status(f"Write failed: {exc}", error=True)
            return None

        _logger.debug("[cell_annotation] save_checkpoint: written as %s", ckpt_id)
        self._set_status(
            f'<span style="color:green">Saved checkpoint [{step_id}] {description}</span>'
        )
        self._refresh_tree()
        self._update_parent_dropdown()
        return ckpt_id

    def _on_save_button(self, _btn=None):
        parent_val = self.parent_dropdown.value
        self.save_checkpoint(
            step_id=self.step_id_input.value.strip(),
            description=self.description_input.value.strip(),
            parent_id=parent_val,
            op=self.op_dropdown.value,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Load a saved checkpoint and restore the heatmap state from it."""
        _logger.debug("[cell_annotation] load_checkpoint: entry (id=%r)", checkpoint_id)
        if self._store is None:
            self._set_status("Cannot load: no dataset folder configured.", error=True)
            return False
        if self._heatmap_plugin is None:
            self._set_status("Cannot load: Heatmap plugin not loaded.", error=True)
            return False

        try:
            adata = self._store.read_checkpoint(checkpoint_id)
        except FileNotFoundError as exc:
            self._set_status(f"Load failed: {exc}", error=True)
            return False
        _logger.debug("[cell_annotation] load_checkpoint: read ok (shape=%s)", adata.shape)

        _logger.debug("[cell_annotation] load_checkpoint: calling import_heatmap_state")
        try:
            self._heatmap_plugin.import_heatmap_state(adata)
        except Exception as exc:
            self._set_status(f"Import failed: {exc}", error=True)
            return False
        _logger.debug("[cell_annotation] load_checkpoint: import_heatmap_state done")

        # Optionally restore FlowSOM params
        if self._flowsom_plugin is not None and "flowsom" in adata.uns:
            _logger.debug("[cell_annotation] load_checkpoint: restoring FlowSOM params")
            try:
                self._flowsom_plugin.import_flowsom_params(adata.uns["flowsom"])
            except Exception:
                pass

        ckpt_meta = adata.uns.get("checkpoint", {})
        step = ckpt_meta.get("step_id", "")
        desc = ckpt_meta.get("description", "")
        _logger.debug("[cell_annotation] load_checkpoint: complete (step=%r)", step)
        self._set_status(
            f'<span style="color:green">Loaded checkpoint [{step}] {desc}</span>'
        )
        return True

    def _on_load_button(self, _btn=None):
        selected = getattr(self.tree_widget, "selected_id", "")
        if not selected:
            self._set_status("Select a checkpoint first.", error=True)
            return
        self.load_checkpoint(selected)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _on_delete_button(self, _btn=None):
        selected = getattr(self.tree_widget, "selected_id", "")
        if not selected:
            self._set_status("Select a checkpoint first.", error=True)
            return
        if self._store is None:
            return
        try:
            self._store.delete_checkpoint(selected)
        except Exception as exc:
            self._set_status(f"Delete failed: {exc}", error=True)
            return
        self.tree_widget.selected_id = ""
        self._set_status('<span style="color:green">Checkpoint deleted.</span>')
        self._refresh_tree()
        self._update_parent_dropdown()

    # ------------------------------------------------------------------
    # Tree / dropdown refresh helpers
    # ------------------------------------------------------------------

    def _refresh_tree(self):
        """Reload manifest and push updated nodes to the tree widget."""
        if self._store is None:
            return
        entries = self._store.list_checkpoints()
        tree = getattr(self, "tree_widget", None)
        if tree is not None:
            tree.nodes = list(entries)

    def _update_parent_dropdown(self):
        """Repopulate the parent dropdown from the current manifest."""
        if self._store is None:
            return
        dropdown = getattr(self, "parent_dropdown", None)
        if dropdown is None:
            return
        entries = self._store.list_checkpoints()
        options = [("(none)", None)] + [
            (
                f"[{e.get('step_id', '')}] {e.get('description', '')} ({e['id'][:8]}…)",
                e["id"],
            )
            for e in entries
        ]
        current = dropdown.value
        dropdown.options = options
        # Preserve selection if still valid
        valid_ids = {e["id"] for e in entries}
        if current not in valid_ids:
            dropdown.value = None

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------

    def _set_status(self, message: str, *, error: bool = False) -> None:
        _logger.warning(message) if error else _logger.info(message)
        label = getattr(self, "status_label", None)
        if label is None:
            return
        if error:
            label.value = f'<span style="color:orange">{message}</span>'
        else:
            label.value = message
