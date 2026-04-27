"""anywidget-based drag-sortable class list for the Mask Painter.

Each row renders as:

    [≡ drag]  [□ vis]  [■ <color input>  ClassName]  [□ fill]  [× remove]

A footer row below the list shows a ``<select>`` of available (inactive) classes
and an "Add" button so users can expand the active selection on demand.

Drag handles allow the user to reorder rows. Visibility and fill checkboxes,
and the native ``<input type="color">`` picker, sync to Python via traitlets.

Traitlets (all ``sync=True`` when anywidget is available):
- ``class_order``       — list of active class name strings; defines the row display order
- ``class_colors``      — dict mapping class name → hex color string
- ``class_visible``     — dict mapping class name → bool (True = visible/checked)
- ``class_fill``        — dict mapping class name → bool (True = fill, False = outline)
- ``default_color``     — the global default color string
- ``available_classes`` — list of dataset classes not currently in ``class_order``
- ``add_requested``     — JS→Python signal; set to a class name when "Add" is clicked
- ``remove_requested``  — JS→Python signal; set to a class name when "×" is clicked
"""

try:
    import anywidget
    import traitlets

    class MaskClassListWidget(anywidget.AnyWidget):  # type: ignore[misc]
        """anywidget that renders the full per-class color/mode list with drag reorder."""

        class_order: list = traitlets.List(traitlets.Unicode()).tag(sync=True)
        class_colors: dict = traitlets.Dict({}).tag(sync=True)
        class_visible: dict = traitlets.Dict({}).tag(sync=True)
        class_fill: dict = traitlets.Dict({}).tag(sync=True)
        default_color: str = traitlets.Unicode("#ffffff").tag(sync=True)
        available_classes: list = traitlets.List(traitlets.Unicode()).tag(sync=True)
        add_requested: str = traitlets.Unicode("").tag(sync=True)
        remove_requested: str = traitlets.Unicode("").tag(sync=True)

        _css = """
.mask-cl-root {
    display: flex;
    flex-direction: column;
    font-family: var(--jp-ui-font-family, sans-serif);
    font-size: var(--jp-ui-font-size1, 13px);
}
.mask-cl-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 4px;
    padding: 2px 0;
    border-bottom: 1px solid var(--jp-border-color2, #e0e0e0);
    cursor: default;
    user-select: none;
}
.mask-cl-row.drag-over {
    border-top: 2px solid var(--jp-brand-color1, #2196f3);
}
.mask-cl-row.dragging {
    opacity: 0.4;
}
.mask-cl-drag {
    cursor: grab;
    color: var(--jp-ui-font-color2, #888);
    font-size: 16px;
    flex: 0 0 auto;
    padding: 0 2px;
    line-height: 1;
}
.mask-cl-drag:active {
    cursor: grabbing;
}
.mask-cl-vis {
    flex: 0 0 auto;
    cursor: pointer;
    margin: 0;
}
.mask-cl-color {
    flex: 0 0 auto;
    width: 28px;
    height: 20px;
    padding: 1px;
    border: 1px solid var(--jp-border-color1, #ccc);
    border-radius: 3px;
    cursor: pointer;
    background: none;
}
.mask-cl-name {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 13px;
    color: var(--jp-ui-font-color1, #000);
}
.mask-cl-fill {
    flex: 0 0 auto;
    cursor: pointer;
    margin: 0;
}
.mask-cl-fill-label {
    flex: 0 0 auto;
    font-size: 11px;
    color: var(--jp-ui-font-color2, #888);
    white-space: nowrap;
    user-select: none;
}
.mask-cl-remove {
    flex: 0 0 auto;
    background: none;
    border: none;
    color: var(--jp-error-color1, #c00);
    font-size: 14px;
    line-height: 1;
    cursor: pointer;
    padding: 0 2px;
    opacity: 0.6;
}
.mask-cl-remove:hover {
    opacity: 1;
}
.mask-cl-scroll {
    max-height: 300px;
    overflow-y: auto;
}
.mask-cl-add-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 4px;
    padding: 4px 0 2px 0;
    border-top: 1px solid var(--jp-border-color2, #e0e0e0);
    margin-top: 2px;
}
.mask-cl-add-select {
    flex: 1 1 auto;
    min-width: 0;
    padding: 2px 4px;
    border: 1px solid var(--jp-border-color1, #ccc);
    border-radius: 3px;
    background: var(--jp-layout-color1, #fff);
    color: var(--jp-ui-font-color1, #000);
    font-size: inherit;
    font-family: inherit;
}
.mask-cl-add-btn {
    flex: 0 0 auto;
    padding: 2px 10px;
    background: var(--jp-brand-color1, #2196f3);
    color: #fff;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-size: inherit;
    font-family: inherit;
    white-space: nowrap;
}
.mask-cl-add-btn:hover {
    background: var(--jp-brand-color0, #1976d2);
}
.mask-cl-add-btn:disabled {
    opacity: 0.4;
    cursor: default;
}
"""

        _esm = r"""
export function render({ model, el }) {
  var _dragSrcCls = null;
  var _rebuilding = false;

  // ---------- helpers ----------
  function getOrder()     { return (model.get('class_order')       || []).slice(); }
  function getColors()    { return model.get('class_colors')        || {}; }
  function getVisible()   { return model.get('class_visible')       || {}; }
  function getFill()      { return model.get('class_fill')          || {}; }
  function getAvailable() { return (model.get('available_classes')  || []).slice(); }

  // ---------- root ----------
  var root = document.createElement('div');
  root.className = 'mask-cl-root';

  var scroll = document.createElement('div');
  scroll.className = 'mask-cl-scroll';
  root.appendChild(scroll);

  // ---------- add-row (rendered once, options updated in place) ----------
  var addRow = document.createElement('div');
  addRow.className = 'mask-cl-add-row';

  var addSelect = document.createElement('select');
  addSelect.className = 'mask-cl-add-select';

  var addBtn = document.createElement('button');
  addBtn.className = 'mask-cl-add-btn';
  addBtn.textContent = '+ Add';
  addBtn.addEventListener('mousedown', function(e) { e.preventDefault(); });
  addBtn.addEventListener('click', function() {
    var val = addSelect.value;
    if (!val) return;
    model.set('add_requested', val);
    model.save_changes();
  });

  addRow.appendChild(addSelect);
  addRow.appendChild(addBtn);
  root.appendChild(addRow);

  function rebuildAddOptions() {
    var avail = getAvailable();
    while (addSelect.firstChild) { addSelect.removeChild(addSelect.firstChild); }
    if (avail.length === 0) {
      addBtn.disabled = true;
      var placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = '(all classes active)';
      addSelect.appendChild(placeholder);
    } else {
      addBtn.disabled = false;
      var placeholder2 = document.createElement('option');
      placeholder2.value = '';
      placeholder2.textContent = 'Select class to add…';
      addSelect.appendChild(placeholder2);
      avail.forEach(function(cls) {
        var opt = document.createElement('option');
        opt.value = cls;
        opt.textContent = cls;
        addSelect.appendChild(opt);
      });
    }
  }

  // ---------- row factory ----------
  function makeRow(cls) {
    var colors  = getColors();
    var visible = getVisible();
    var fill    = getFill();
    var color   = colors[cls]  || model.get('default_color') || '#ffffff';
    var vis     = visible[cls] !== false;
    var fillOn  = !!fill[cls];

    var row = document.createElement('div');
    row.className = 'mask-cl-row';
    row.dataset.cls = cls;

    // --- drag handle ---
    var drag = document.createElement('span');
    drag.className = 'mask-cl-drag';
    drag.innerHTML = '&#9776;';  // ☰
    drag.title = 'Drag to reorder';
    drag.draggable = true;

    drag.addEventListener('dragstart', function(e) {
      _dragSrcCls = cls;
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', cls);
      setTimeout(function() { row.classList.add('dragging'); }, 0);
    });
    drag.addEventListener('dragend', function() {
      _dragSrcCls = null;
      scroll.querySelectorAll('.mask-cl-row').forEach(function(r) {
        r.classList.remove('dragging', 'drag-over');
      });
    });

    row.addEventListener('dragover', function(e) {
      if (!_dragSrcCls || _dragSrcCls === cls) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      scroll.querySelectorAll('.mask-cl-row').forEach(function(r) {
        r.classList.remove('drag-over');
      });
      row.classList.add('drag-over');
    });
    row.addEventListener('dragleave', function(e) {
      if (!row.contains(e.relatedTarget)) {
        row.classList.remove('drag-over');
      }
    });
    row.addEventListener('drop', function(e) {
      e.preventDefault();
      row.classList.remove('drag-over');
      if (!_dragSrcCls || _dragSrcCls === cls) return;
      var order = getOrder();
      var fromIdx = order.indexOf(_dragSrcCls);
      var toIdx   = order.indexOf(cls);
      if (fromIdx === -1 || toIdx === -1) return;
      order.splice(fromIdx, 1);
      order.splice(toIdx, 0, _dragSrcCls);
      model.set('class_order', order);
      model.save_changes();
      _dragSrcCls = null;
    });

    // --- vis checkbox ---
    var visCb = document.createElement('input');
    visCb.type = 'checkbox';
    visCb.className = 'mask-cl-vis';
    visCb.checked = vis;
    visCb.title = 'Show / hide ' + cls;
    visCb.addEventListener('change', function() {
      if (_rebuilding) return;
      var d = Object.assign({}, getVisible());
      d[cls] = visCb.checked;
      model.set('class_visible', d);
      model.save_changes();
    });

    // --- color input ---
    var colorInp = document.createElement('input');
    colorInp.type = 'color';
    colorInp.className = 'mask-cl-color';
    colorInp.value = color;
    colorInp.title = cls;
    colorInp.addEventListener('input', function() {
      if (_rebuilding) return;
      var d = Object.assign({}, getColors());
      d[cls] = colorInp.value;
      model.set('class_colors', d);
      model.save_changes();
    });

    // --- class name label ---
    var name = document.createElement('span');
    name.className = 'mask-cl-name';
    name.textContent = cls;
    name.title = cls;

    // --- fill checkbox ---
    var fillCb = document.createElement('input');
    fillCb.type = 'checkbox';
    fillCb.className = 'mask-cl-fill';
    fillCb.checked = fillOn;
    fillCb.title = 'Render ' + cls + ' as filled (unchecked = outline)';
    fillCb.addEventListener('change', function() {
      if (_rebuilding) return;
      var d = Object.assign({}, getFill());
      d[cls] = fillCb.checked;
      model.set('class_fill', d);
      model.save_changes();
    });

    var fillLabel = document.createElement('span');
    fillLabel.className = 'mask-cl-fill-label';
    fillLabel.textContent = 'fill';

    // --- remove button ---
    var removeBtn = document.createElement('button');
    removeBtn.className = 'mask-cl-remove';
    removeBtn.textContent = '×';
    removeBtn.title = 'Remove ' + cls + ' from active list';
    removeBtn.addEventListener('mousedown', function(e) { e.preventDefault(); });
    removeBtn.addEventListener('click', function() {
      model.set('remove_requested', cls);
      model.save_changes();
    });

    row.appendChild(drag);
    row.appendChild(visCb);
    row.appendChild(colorInp);
    row.appendChild(name);
    row.appendChild(fillCb);
    row.appendChild(fillLabel);
    row.appendChild(removeBtn);
    return row;
  }

  // ---------- full rebuild (Python → DOM) ----------
  function rebuildRows() {
    _rebuilding = true;
    while (scroll.firstChild) { scroll.removeChild(scroll.firstChild); }
    var order = getOrder();
    order.forEach(function(cls) {
      scroll.appendChild(makeRow(cls));
    });
    _rebuilding = false;
  }

  // ---------- targeted updates (Python → DOM, avoid full rebuild) ----------
  function syncColors() {
    var colors = getColors();
    var def    = model.get('default_color') || '#ffffff';
    scroll.querySelectorAll('.mask-cl-row').forEach(function(row) {
      var cls = row.dataset.cls;
      var inp = row.querySelector('.mask-cl-color');
      if (inp && cls in colors) { inp.value = colors[cls] || def; }
    });
  }

  function syncVisible() {
    var visible = getVisible();
    scroll.querySelectorAll('.mask-cl-row').forEach(function(row) {
      var cls = row.dataset.cls;
      var cb  = row.querySelector('.mask-cl-vis');
      if (cb && cls in visible) { cb.checked = visible[cls] !== false; }
    });
  }

  function syncFill() {
    var fill = getFill();
    scroll.querySelectorAll('.mask-cl-row').forEach(function(row) {
      var cls = row.dataset.cls;
      var cb  = row.querySelector('.mask-cl-fill');
      if (cb && cls in fill) { cb.checked = !!fill[cls]; }
    });
  }

  // Re-build on order changes; only sync values on color/vis/fill changes.
  model.on('change:class_order',       rebuildRows);
  model.on('change:class_colors',      syncColors);
  model.on('change:class_visible',     syncVisible);
  model.on('change:class_fill',        syncFill);
  model.on('change:default_color',     syncColors);
  model.on('change:available_classes', rebuildAddOptions);

  rebuildRows();
  rebuildAddOptions();
  el.appendChild(root);
}
"""

except (ImportError, AttributeError):
    # Fallback for environments without anywidget (e.g. unit tests).
    # Provides the same traitlet interface so Python code works unchanged.
    import traitlets  # type: ignore[import]

    class MaskClassListWidget(traitlets.HasTraits):  # type: ignore[no-redef]
        """Headless fallback used when anywidget is not installed."""

        class_order: list = traitlets.List(traitlets.Unicode())
        class_colors: dict = traitlets.Dict({})
        class_visible: dict = traitlets.Dict({})
        class_fill: dict = traitlets.Dict({})
        default_color: str = traitlets.Unicode("#ffffff")
        available_classes: list = traitlets.List(traitlets.Unicode())
        add_requested: str = traitlets.Unicode("")
        remove_requested: str = traitlets.Unicode("")


