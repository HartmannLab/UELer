"""Searchable, always-scrollable marker/feature picker (issue #125).

ipywidgets' ``TagsInput`` — the widget every marker/feature picker in UELer used to
be — does not render its own option list.  Its view creates a native ``<datalist>``
and points the text input at it::

    this.taginput.setAttribute("list", this.datalistID)
    this.autocompleteList = document.createElement("datalist")

The suggestion popup is therefore drawn by the *browser*, outside our control: its
height and item count are host-defined and cannot be styled or scrolled, it only
appears while typing, and in embedded notebook hosts (VS Code notebooks, webviews,
iframes) it is clipped at the container edge.  With a long marker list most options
are unreachable — the bug reported in #125.

``ChannelPickerWidget`` replaces it with an in-DOM picker we own::

    Channels:  [ CD45 x] [ CD3 x]              <- chips: current selection, ordered
    [ filter markers...                 ] [ v ]
    +--------------------------------------+
    | + CD45                               |   scrollable list (max-height + overflow-y)
    |   CD4                                |   every option reachable, keyboard navigable
    |   CD8                                |
    +--------------------------------------+
      12 of 148 shown - 2 selected   Select all shown - Clear

The option list is a plain scrollable ``div`` rendered **in the widget's own layout
flow** — not a native popup and not a floating overlay — so no ancestor ``overflow``,
stacking context, iframe boundary, or host viewport can clip it.

The public API is deliberately a drop-in for ``TagsInput``: the observable traits
``value`` (ordered list of selected names) and ``allowed_tags`` (available options)
behave the same, so existing viewer/plugin code keeps working unchanged.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional, Sequence

_logger = logging.getLogger(__name__)

# Default height of the option list before it starts scrolling (~10 rows).
DEFAULT_LIST_MAX_HEIGHT = 220


def _normalise_options(raw: Any) -> List[str]:
    """Coerce an ``allowed_tags``-like value into a list of unique strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    try:
        items: Iterable[Any] = list(raw)
    except TypeError:  # pragma: no cover - defensive
        return []
    seen: set = set()
    options: List[str] = []
    for item in items:
        name = item if isinstance(item, str) else str(item)
        if name not in seen:
            seen.add(name)
            options.append(name)
    return options


def _normalise_selection(raw: Any, allowed: Sequence[str]) -> List[str]:
    """Coerce a ``value``-like assignment into a clean, ordered selection.

    * a bare string becomes a single-item selection (``run_flowsom`` passes one),
    * tuples/other iterables are accepted (the viewer assigns tuples),
    * duplicates are dropped while preserving the first occurrence's order,
    * names absent from ``allowed`` are dropped — ``TagsInput`` raised ``TraitError``
      here; dropping keeps stale checkpoint/marker-set restores from exploding.
    """
    selection = _normalise_options(raw)
    if not allowed:
        return selection
    allowed_set = set(allowed)
    return [name for name in selection if name in allowed_set]


_CSS = """
.ucp-root {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
    box-sizing: border-box;
    font-family: var(--jp-ui-font-family, sans-serif);
    font-size: var(--jp-ui-font-size1, 13px);
    color: var(--jp-ui-font-color1, #000);
}
.ucp-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    max-height: 96px;
    overflow-y: auto;
}
.ucp-chip {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 1px 4px 1px 6px;
    border-radius: 9px;
    background: var(--jp-brand-color3, #bbdefb);
    color: var(--jp-ui-font-color1, #000);
    font-size: 11px;
    max-width: 100%;
}
.ucp-chip-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ucp-chip-x {
    border: none;
    background: none;
    cursor: pointer;
    padding: 0 1px;
    font-size: 12px;
    line-height: 1;
    color: inherit;
    opacity: 0.65;
}
.ucp-chip-x:hover { opacity: 1; }
.ucp-empty {
    font-size: 11px;
    font-style: italic;
    color: var(--jp-ui-font-color2, #888);
}
.ucp-search {
    display: flex;
    align-items: center;
    gap: 4px;
    width: 100%;
    box-sizing: border-box;
}
.ucp-input {
    flex: 1 1 auto;
    min-width: 0;
    padding: 2px 5px;
    border: 1px solid var(--jp-border-color1, #ccc);
    border-radius: 3px;
    background: var(--jp-layout-color1, #fff);
    color: var(--jp-ui-font-color1, #000);
    font-family: inherit;
    font-size: inherit;
    box-sizing: border-box;
}
.ucp-toggle {
    flex: 0 0 auto;
    width: 24px;
    padding: 2px 0;
    border: 1px solid var(--jp-border-color1, #ccc);
    border-radius: 3px;
    background: var(--jp-layout-color2, #f5f5f5);
    color: var(--jp-ui-font-color1, #000);
    cursor: pointer;
    font-size: 10px;
    line-height: 1.2;
}
.ucp-toggle:hover { background: var(--jp-layout-color3, #eee); }
.ucp-panel {
    display: flex;
    flex-direction: column;
    border: 1px solid var(--jp-border-color1, #ccc);
    border-radius: 3px;
    background: var(--jp-layout-color1, #fff);
    box-sizing: border-box;
    width: 100%;
}
.ucp-list {
    overflow-y: auto;
    overflow-x: hidden;
}
.ucp-opt {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 5px;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
}
.ucp-opt:hover { background: var(--jp-layout-color2, #f5f5f5); }
.ucp-opt.is-active { background: var(--jp-layout-color2, #f0f0f0); }
.ucp-opt.is-selected { font-weight: 600; }
.ucp-mark {
    flex: 0 0 12px;
    width: 12px;
    text-align: center;
    color: var(--jp-brand-color1, #2196f3);
    font-size: 11px;
}
.ucp-opt-label {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
}
.ucp-none {
    padding: 4px 6px;
    font-size: 11px;
    font-style: italic;
    color: var(--jp-ui-font-color2, #888);
}
.ucp-foot {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 2px 5px;
    border-top: 1px solid var(--jp-border-color2, #e0e0e0);
    font-size: 10px;
    color: var(--jp-ui-font-color2, #888);
}
.ucp-count { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.ucp-act {
    flex: 0 0 auto;
    border: none;
    background: none;
    padding: 0 2px;
    cursor: pointer;
    font-size: 10px;
    font-family: inherit;
    color: var(--jp-brand-color1, #2196f3);
}
.ucp-act:hover { text-decoration: underline; }
.ucp-root.is-disabled { opacity: 0.5; pointer-events: none; }
"""


_ESM = r"""
export function render({ model, el }) {
  var open = false;
  var query = '';
  var activeIdx = -1;
  var filtered = [];

  function options() { return (model.get('allowed_tags') || []).map(String); }
  function selection() { return (model.get('value') || []).map(String); }

  function commit(next) {
    var seen = {}, clean = [];
    next.forEach(function (name) {
      if (!seen[name]) { seen[name] = true; clean.push(name); }
    });
    model.set('value', clean);
    model.save_changes();
    renderChips();
    renderList();
  }

  // ---------- DOM skeleton ----------
  var root = document.createElement('div');
  root.className = 'ucp-root';

  var chips = document.createElement('div');
  chips.className = 'ucp-chips';

  var searchRow = document.createElement('div');
  searchRow.className = 'ucp-search';

  var input = document.createElement('input');
  input.type = 'text';
  input.className = 'ucp-input';

  var toggle = document.createElement('button');
  toggle.className = 'ucp-toggle';
  toggle.type = 'button';
  toggle.title = 'Show / hide the option list';

  searchRow.appendChild(input);
  searchRow.appendChild(toggle);

  // The option list is an in-flow panel (never a native popup, never an
  // overlay): nothing above it in the DOM can clip it, and it always scrolls.
  var panel = document.createElement('div');
  panel.className = 'ucp-panel';

  var list = document.createElement('div');
  list.className = 'ucp-list';

  var foot = document.createElement('div');
  foot.className = 'ucp-foot';

  var count = document.createElement('span');
  count.className = 'ucp-count';

  var selectAllBtn = document.createElement('button');
  selectAllBtn.className = 'ucp-act';
  selectAllBtn.type = 'button';
  selectAllBtn.textContent = 'Select all shown';

  var clearBtn = document.createElement('button');
  clearBtn.className = 'ucp-act';
  clearBtn.type = 'button';
  clearBtn.textContent = 'Clear';

  foot.appendChild(count);
  foot.appendChild(selectAllBtn);
  foot.appendChild(clearBtn);
  panel.appendChild(list);
  panel.appendChild(foot);

  root.appendChild(chips);
  root.appendChild(searchRow);
  root.appendChild(panel);

  // ---------- chips (current selection) ----------
  function renderChips() {
    while (chips.firstChild) { chips.removeChild(chips.firstChild); }
    var sel = selection();
    if (!sel.length) {
      var hint = document.createElement('span');
      hint.className = 'ucp-empty';
      hint.textContent = 'No selection yet - pick from the list below.';
      chips.appendChild(hint);
      return;
    }
    sel.forEach(function (name) {
      var chip = document.createElement('span');
      chip.className = 'ucp-chip';
      var label = document.createElement('span');
      label.className = 'ucp-chip-label';
      label.textContent = name;
      label.title = name;
      var x = document.createElement('button');
      x.className = 'ucp-chip-x';
      x.type = 'button';
      x.textContent = '×';
      x.title = 'Remove ' + name;
      x.addEventListener('mousedown', function (e) { e.preventDefault(); });
      x.addEventListener('click', function () {
        commit(selection().filter(function (n) { return n !== name; }));
      });
      chip.appendChild(label);
      chip.appendChild(x);
      chips.appendChild(chip);
    });
  }

  // ---------- option list ----------
  function renderList() {
    panel.style.display = open ? 'flex' : 'none';
    toggle.textContent = open ? '▴' : '▾';
    var all = options();
    var q = query.trim().toLowerCase();
    filtered = q ? all.filter(function (o) { return o.toLowerCase().indexOf(q) !== -1; }) : all;
    if (activeIdx >= filtered.length) { activeIdx = filtered.length - 1; }
    if (!open) { return; }

    var selected = {};
    selection().forEach(function (n) { selected[n] = true; });

    list.style.maxHeight = String(model.get('list_max_height') || 220) + 'px';
    while (list.firstChild) { list.removeChild(list.firstChild); }

    if (!filtered.length) {
      var none = document.createElement('div');
      none.className = 'ucp-none';
      none.textContent = all.length ? 'No option matches "' + query + '".' : 'No options available.';
      list.appendChild(none);
    } else {
      filtered.forEach(function (name, idx) {
        var row = document.createElement('div');
        row.className = 'ucp-opt';
        if (selected[name]) { row.classList.add('is-selected'); }
        if (idx === activeIdx) { row.classList.add('is-active'); }
        row.dataset.name = name;

        var mark = document.createElement('span');
        mark.className = 'ucp-mark';
        mark.textContent = selected[name] ? '✓' : '';

        var label = document.createElement('span');
        label.className = 'ucp-opt-label';
        label.textContent = name;
        label.title = name;

        row.appendChild(mark);
        row.appendChild(label);
        row.addEventListener('mousedown', function (e) { e.preventDefault(); });
        row.addEventListener('click', function () {
          activeIdx = idx;
          toggleName(name);
        });
        list.appendChild(row);
      });
    }

    var parts = [filtered.length + ' of ' + all.length + ' shown'];
    parts.push(selection().length + ' selected');
    count.textContent = parts.join(' · ');
    count.title = count.textContent;
  }

  function toggleName(name) {
    var sel = selection();
    if (sel.indexOf(name) === -1) {
      commit(sel.concat([name]));
    } else {
      commit(sel.filter(function (n) { return n !== name; }));
    }
  }

  function setOpen(next) {
    if (open === next) { return; }
    open = next;
    if (!open) { activeIdx = -1; }
    renderList();
  }

  function scrollActiveIntoView() {
    var rows = list.querySelectorAll('.ucp-opt');
    if (activeIdx >= 0 && activeIdx < rows.length && rows[activeIdx].scrollIntoView) {
      rows[activeIdx].scrollIntoView({ block: 'nearest' });
    }
  }

  // ---------- interaction ----------
  toggle.addEventListener('mousedown', function (e) { e.preventDefault(); });
  toggle.addEventListener('click', function () {
    setOpen(!open);
    if (open) { input.focus(); }
  });

  input.addEventListener('focus', function () { setOpen(true); });
  input.addEventListener('input', function () {
    query = input.value;
    activeIdx = query ? 0 : -1;
    if (!open) { open = true; }
    renderList();
  });
  input.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!open) { setOpen(true); }
      if (!filtered.length) { return; }
      activeIdx = e.key === 'ArrowDown'
        ? (activeIdx + 1) % filtered.length
        : (activeIdx <= 0 ? filtered.length - 1 : activeIdx - 1);
      renderList();
      scrollActiveIntoView();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIdx >= 0 && activeIdx < filtered.length) {
        toggleName(filtered[activeIdx]);
      } else if (filtered.length === 1) {
        toggleName(filtered[0]);
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setOpen(false);
    } else if (e.key === 'Backspace' && !input.value) {
      var sel = selection();
      if (sel.length) { commit(sel.slice(0, -1)); }
    }
  });

  selectAllBtn.addEventListener('mousedown', function (e) { e.preventDefault(); });
  selectAllBtn.addEventListener('click', function () {
    commit(selection().concat(filtered));
  });
  clearBtn.addEventListener('mousedown', function (e) { e.preventDefault(); });
  clearBtn.addEventListener('click', function () { commit([]); });

  function onDocMouseDown(e) {
    if (open && !root.contains(e.target)) { setOpen(false); }
  }
  document.addEventListener('mousedown', onDocMouseDown, true);

  // ---------- Python -> DOM ----------
  function applyStatics() {
    input.placeholder = model.get('placeholder') || 'Type to filter...';
    input.setAttribute('aria-label', model.get('description') || 'Filter options');
    input.title = model.get('description') || '';
    var disabled = !!model.get('disabled');
    input.disabled = disabled;
    root.classList.toggle('is-disabled', disabled);
  }

  model.on('change:value', function () { renderChips(); renderList(); });
  model.on('change:allowed_tags', function () { activeIdx = -1; renderList(); });
  model.on('change:list_max_height', renderList);
  model.on('change:placeholder', applyStatics);
  model.on('change:description', applyStatics);
  model.on('change:disabled', applyStatics);

  applyStatics();
  renderChips();
  renderList();
  el.appendChild(root);

  return function cleanup() {
    document.removeEventListener('mousedown', onDocMouseDown, true);
  };
}
"""


def _anywidget_module():
    """Return the imported ``anywidget`` module, or ``None`` if unavailable.

    The unit-test bootstrap installs a stub module tagged ``__bootstrap_stub__``;
    that stub has no ``AnyWidget``, which we treat like a missing dependency.
    """
    try:
        import anywidget  # type: ignore[import]
    except ImportError:
        return None
    return anywidget


_ANYWIDGET = _anywidget_module()
#: ``True`` when a *real* anywidget is installed, i.e. the browser-side picker works.
ANYWIDGET_AVAILABLE = bool(_ANYWIDGET is not None and hasattr(_ANYWIDGET, "AnyWidget"))
#: ``True`` when the test bootstrap's placeholder module is in play.
ANYWIDGET_STUBBED = bool(
    _ANYWIDGET is not None and getattr(_ANYWIDGET, "__bootstrap_stub__", False)
)


if ANYWIDGET_AVAILABLE:
    import traitlets  # type: ignore[import]

    class ChannelPickerWidget(_ANYWIDGET.AnyWidget):  # type: ignore[misc,name-defined]
        """Drop-in ``TagsInput`` replacement with a scrollable, searchable option list.

        ``value``/``allowed_tags`` are untyped lists on purpose: the validators below
        coerce whatever the caller passes (a bare string, a tuple, non-string column
        labels) into a clean list of strings instead of raising ``TraitError``.
        """

        value: list = traitlets.List().tag(sync=True)
        allowed_tags: list = traitlets.List().tag(sync=True)
        placeholder: str = traitlets.Unicode("Type to filter...").tag(sync=True)
        description: str = traitlets.Unicode("").tag(sync=True)
        disabled: bool = traitlets.Bool(False).tag(sync=True)
        list_max_height: int = traitlets.Int(DEFAULT_LIST_MAX_HEIGHT).tag(sync=True)

        _css = _CSS
        _esm = _ESM

        @traitlets.validate("allowed_tags")
        def _validate_allowed_tags(self, proposal):
            return _normalise_options(proposal["value"])

        @traitlets.validate("value")
        def _validate_value(self, proposal):
            return _normalise_selection(proposal["value"], self.allowed_tags)

else:  # pragma: no cover - exercised by the headless unit-test bootstrap
    import traitlets  # type: ignore[import]

    class ChannelPickerWidget(traitlets.HasTraits):  # type: ignore[no-redef]
        """Headless fallback with the same traits (no anywidget available)."""

        value: list = traitlets.List()
        allowed_tags: list = traitlets.List()
        placeholder: str = traitlets.Unicode("Type to filter...")
        description: str = traitlets.Unicode("")
        disabled: bool = traitlets.Bool(False)
        list_max_height: int = traitlets.Int(DEFAULT_LIST_MAX_HEIGHT)

        _css = _CSS
        _esm = _ESM

        def __init__(self, **kwargs):
            layout = kwargs.pop("layout", None)
            # ``allowed_tags`` must land before ``value`` so the selection can be
            # validated against it (traitlets applies kwargs in dict order).
            ordered = {}
            if "allowed_tags" in kwargs:
                ordered["allowed_tags"] = kwargs.pop("allowed_tags")
            ordered.update(kwargs)
            super().__init__(**ordered)
            # Keep a ``layout`` attribute so code that reads/tweaks it (as it would
            # on a real DOMWidget) keeps working under the headless fallback.
            self.layout = layout if layout is not None else SimpleNamespace()

        @traitlets.validate("allowed_tags")
        def _validate_allowed_tags(self, proposal):
            return _normalise_options(proposal["value"])

        @traitlets.validate("value")
        def _validate_value(self, proposal):
            return _normalise_selection(proposal["value"], self.allowed_tags)


def _tags_input_class():
    """Return ``ipywidgets.TagsInput`` if importable, else ``None``."""
    try:
        import ipywidgets  # type: ignore[import]
    except ImportError:  # pragma: no cover - ipywidgets is a hard dependency
        return None
    return getattr(ipywidgets, "TagsInput", None)


def build_channel_picker(
    *,
    allowed_tags: Optional[Sequence[Any]] = None,
    value: Any = None,
    description: str = "",
    placeholder: str = "Type to filter markers...",
    list_max_height: int = DEFAULT_LIST_MAX_HEIGHT,
    disabled: bool = False,
    layout: Any = None,
    **ignored: Any,
):
    """Return a marker/feature picker widget.

    Prefers :class:`ChannelPickerWidget` (the #125 fix).  When anywidget is missing
    from a real runtime — as opposed to stubbed by the test bootstrap — falls back to
    ipywidgets' ``TagsInput`` so the viewer still has a working picker.

    ``TagsInput``-only keyword arguments (``allow_duplicates``, ``style``, ...) are
    accepted and ignored so existing call sites need no edits.
    """
    options = _normalise_options(allowed_tags)
    selection = _normalise_selection(value, options)

    if not ANYWIDGET_AVAILABLE and not ANYWIDGET_STUBBED:
        tags_input = _tags_input_class()
        if tags_input is not None:
            _logger.debug(
                "anywidget unavailable; falling back to TagsInput for the channel picker"
            )
            kwargs = {
                "allowed_tags": options,
                "value": selection,
                "description": description,
                "allow_duplicates": False,
            }
            if layout is not None:
                kwargs["layout"] = layout
            try:
                return tags_input(**kwargs)
            except Exception:  # pragma: no cover - very old ipywidgets
                _logger.debug("TagsInput fallback failed; using headless picker")

    kwargs = {
        "allowed_tags": options,
        "value": selection,
        "description": description,
        "placeholder": placeholder,
        "list_max_height": int(list_max_height),
        "disabled": bool(disabled),
    }
    if layout is not None:
        kwargs["layout"] = layout
    return ChannelPickerWidget(**kwargs)


__all__ = [
    "ANYWIDGET_AVAILABLE",
    "ANYWIDGET_STUBBED",
    "DEFAULT_LIST_MAX_HEIGHT",
    "ChannelPickerWidget",
    "build_channel_picker",
]
