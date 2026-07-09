"""Self-contained anywidget-based expression editor for ROI manager advanced filtering.

The widget renders its own DOM: a text input, an Apply button, operator buttons
(``(`` ``)`` ``&`` ``|`` ``!``) and dynamically created tag buttons.

Because all buttons live inside the same widget ``el`` as the ``<input>``, a
``mousedown`` + ``preventDefault()`` listener keeps the text input focused
when a button is clicked.  ``selectionStart``/``End`` are therefore always
accurate when the ``click`` handler fires — no Python comm round-trip needed.

Traitlets (all ``sync=True`` when anywidget is available):
- ``expression``      — current text of the expression field
- ``tags``            — list of tag names shown as insert buttons
- ``apply_requested`` — integer counter bumped on every Apply click;
                        Python observes this to trigger gallery refresh
"""

try:
    import anywidget
    import traitlets

    class ROIExpressionEditorWidget(anywidget.AnyWidget):  # type: ignore[misc]
        """anywidget that hosts the entire advanced expression filter UI in its own DOM."""

        expression: str = traitlets.Unicode("").tag(sync=True)
        tags: list = traitlets.List(traitlets.Unicode()).tag(sync=True)
        apply_requested: int = traitlets.Int(0).tag(sync=True)

        _css = """
.roi-expr-editor {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-family: var(--jp-ui-font-family, sans-serif);
    font-size: var(--jp-ui-font-size1, 13px);
}
.roi-expr-editor-top {
    display: flex;
    flex-direction: row;
    gap: 6px;
    align-items: center;
}
.roi-expr-editor-input {
    flex: 1 1 auto;
    min-width: 0;
    padding: 3px 6px;
    border: 1px solid var(--jp-border-color1, #ccc);
    border-radius: 3px;
    background: var(--jp-layout-color1, #fff);
    color: var(--jp-ui-font-color1, #000);
    font-size: inherit;
    font-family: inherit;
}
.roi-expr-editor-apply {
    flex: 0 0 auto;
    padding: 3px 10px;
    background: var(--jp-brand-color1, #2196f3);
    color: #fff;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-size: inherit;
    font-family: inherit;
}
.roi-expr-editor-apply:hover {
    background: var(--jp-brand-color0, #1976d2);
}
.roi-expr-editor-ops,
.roi-expr-editor-tags {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 4px;
}
.roi-expr-btn {
    padding: 2px 8px;
    background: var(--jp-layout-color2, #f5f5f5);
    color: var(--jp-ui-font-color1, #000);
    border: 1px solid var(--jp-border-color1, #ccc);
    border-radius: 3px;
    cursor: pointer;
    font-size: inherit;
    font-family: monospace;
    user-select: none;
}
.roi-expr-btn:hover {
    background: var(--jp-layout-color3, #e0e0e0);
}
"""

        _esm = r"""
function formatInsertion(before, after, snippet) {
  // Replicates Python _format_expression_insertion spacing rules.
  var bL = {'': 1, ' ': 1, '\t': 1, '(': 1, '&': 1, '|': 1, '!': 1};
  var bR = {'': 1, ' ': 1, '\t': 1, ')': 1, '&': 1, '|': 1};
  var bChar = before.length ? before[before.length - 1] : '';
  var aChar = after.length ? after[0] : '';
  var needsL = before.length > 0 && !bL[bChar];

  if (snippet === '!') {
    var l = needsL ? ' ' : '';
    return { ins: l + '!', cursor: l.length + 1 };
  }

  if (snippet === ')') {
    var l2 = (needsL && bChar !== ' ' && bChar !== '(') ? ' ' : '';
    return { ins: l2 + ')', cursor: l2.length + 1 };
  }

  var l3 = needsL ? ' ' : '';
  var aB = !!bR[aChar];
  var needsT = (snippet === '(') ? (!aB && aChar !== ')') : !aB;
  var t = needsT ? ' ' : '';
  return { ins: l3 + snippet + t, cursor: l3.length + snippet.length };
}

function insertSnippet(input, snippet) {
  var current = input.value || '';
  var start = (input.selectionStart == null) ? current.length : input.selectionStart;
  var end   = (input.selectionEnd   == null) ? start          : input.selectionEnd;
  var before = current.slice(0, start);
  var after  = current.slice(end);
  var result = formatInsertion(before, after, snippet);
  var newVal = before + result.ins + after;
  var newCur = start + result.cursor;

  if (input.setRangeText) {
    input.setRangeText(result.ins, start, end, 'end');
    input.setSelectionRange(newCur, newCur);
  } else {
    input.value = newVal;
    if (input.setSelectionRange) {
      input.setSelectionRange(newCur, newCur);
    }
  }
  input.focus();
}

export function render({ model, el }) {
  // --- Root container ---
  var root = document.createElement('div');
  root.className = 'roi-expr-editor';

  // --- Top row: input + Apply ---
  var topRow = document.createElement('div');
  topRow.className = 'roi-expr-editor-top';

  var input = document.createElement('input');
  input.type = 'text';
  input.className = 'roi-expr-editor-input';
  input.placeholder = '(good & figure1) & !excluded';
  input.value = model.get('expression') || '';

  var applyBtn = document.createElement('button');
  applyBtn.className = 'roi-expr-editor-apply';
  applyBtn.textContent = 'Apply';

  topRow.appendChild(input);
  topRow.appendChild(applyBtn);

  // --- Operator buttons ---
  var opsRow = document.createElement('div');
  opsRow.className = 'roi-expr-editor-ops';

  ['(', ')', '&', '|', '!'].forEach(function(op) {
    var btn = document.createElement('button');
    btn.className = 'roi-expr-btn';
    btn.textContent = op;
    btn.title = "Insert '" + op + "'";
    // Prevent focus leaving the text input
    btn.addEventListener('mousedown', function(e) { e.preventDefault(); });
    btn.addEventListener('click', function() {
      insertSnippet(input, op);
      model.set('expression', input.value);
      model.save_changes();
    });
    opsRow.appendChild(btn);
  });

  // --- Tag buttons ---
  var tagsRow = document.createElement('div');
  tagsRow.className = 'roi-expr-editor-tags';

  function rebuildTagButtons(tags) {
    while (tagsRow.firstChild) { tagsRow.removeChild(tagsRow.firstChild); }
    (tags || []).forEach(function(tag) {
      var btn = document.createElement('button');
      btn.className = 'roi-expr-btn';
      btn.textContent = tag;
      btn.title = "Insert '" + tag + "'";
      btn.addEventListener('mousedown', function(e) { e.preventDefault(); });
      btn.addEventListener('click', function() {
        insertSnippet(input, tag);
        model.set('expression', input.value);
        model.save_changes();
      });
      tagsRow.appendChild(btn);
    });
  }

  rebuildTagButtons(model.get('tags') || []);

  // --- Wire up Apply ---
  applyBtn.addEventListener('mousedown', function(e) { e.preventDefault(); });
  applyBtn.addEventListener('click', function() {
    model.set('apply_requested', model.get('apply_requested') + 1);
    model.save_changes();
  });

  // --- Sync expression: DOM → model ---
  input.addEventListener('input', function() {
    model.set('expression', input.value);
    model.save_changes();
  });

  // --- Sync expression: model → DOM (Python-initiated changes) ---
  model.on('change:expression', function() {
    var newVal = model.get('expression') || '';
    if (input.value !== newVal) {
      input.value = newVal;
    }
  });

  // --- Rebuild tag buttons when model.tags changes ---
  model.on('change:tags', function() {
    rebuildTagButtons(model.get('tags') || []);
  });

  // --- Assemble ---
  root.appendChild(topRow);
  root.appendChild(opsRow);
  root.appendChild(tagsRow);
  el.appendChild(root);
}
"""

except (ImportError, AttributeError):
    # Fallback for test environments without anywidget installed.
    # Provides the same traitlet attributes so Python code can read/write them.
    import traitlets  # type: ignore[import]

    class ROIExpressionEditorWidget(traitlets.HasTraits):  # type: ignore[no-redef]
        """Headless fallback used when anywidget is not installed (e.g. in tests)."""

        expression = traitlets.Unicode("")
        tags = traitlets.List(traitlets.Unicode())
        apply_requested = traitlets.Int(0)
