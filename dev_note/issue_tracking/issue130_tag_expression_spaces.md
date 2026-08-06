# Issue #130: ROI manager advanced filtering does not support tags containing spaces

## Problem
The ROI manager accepts tags with spaces (`ROIManager._normalise_tags` only strips the
ends of each comma-separated tag), but the advanced tag filter cannot match them.
Typing `my tag` into the expression field fails with the opaque message
`Expression did not reduce to a single value.`, and clicking the helper button for a
spaced tag inserts text that cannot be parsed.

## Root cause
`ueler/viewer/tag_expression.py::_tokenize` treats every whitespace character as a
token separator:

```python
if char.isspace():
    flush_buffer()
    index += 1
    continue
```

So `my tag` lexes to two `name` tokens. `_to_postfix` passes both through, and
`_evaluate_postfix` ends with two values on the stack — hence the "did not reduce"
error. Quoting (`'my tag'`) already worked, but users have no reason to expect it and
the helper tag buttons do not quote what they insert.

## Chosen approach
Follow the suggestion in the issue: **spaces inside a name are part of the name;
spaces at the ends are stripped.** Names can never be adjacent without an operator
between them, so no ambiguity is introduced.

1. `_tokenize` no longer flushes on whitespace. Whitespace is appended to the current
   name buffer (skipped while the buffer is empty, so leading spaces are dropped) and
   the buffer is flushed only at an operator, a parenthesis, a quote, or end of input.
2. Flushed names are normalised with `" ".join(text.split())` — ends stripped, internal
   whitespace runs collapsed to a single space. The tag set the predicate matches
   against is normalised the same way, so `my   tag` still matches the tag `my tag` and
   no previously matching tag stops matching.
3. Quoted names are flushed-before and normalised the same way, so `'my tag'` and
   `my tag` are equivalent. Quoting stays useful for names containing operator
   characters (`a&b`). An empty quoted name now raises instead of silently producing a
   name that can never match.
4. `_to_postfix` reports two adjacent names as `Missing operator between 'a' and 'b'.`
   instead of the old opaque stack error — the only way to still produce adjacent names
   is quoting, and the message should say so.

No change is needed in `roi_manager_plugin.py` or the anywidget editor: the tag helper
buttons insert the raw tag text, which is now parseable as-is.

## Implementation steps
1. Add `_normalise_name()` to `tag_expression.py` and use it in `flush_buffer`, the
   quoted-string branch, and the predicate's tag set.
2. Rewrite the whitespace branch of `_tokenize`; add the missing `flush_buffer()` before
   the quoted-string branch (without it, `foo "bar"` emitted the tokens out of order).
3. Add the adjacent-name check in `_to_postfix`.
4. Extend `tests/test_tag_expression.py` with coverage for spaced tags, spaced tags
   combined with operators/parentheses/negation, whitespace normalisation, quoted
   equivalence, and the new error messages.
5. Run `tests.test_tag_expression`, `tests.test_roi_manager_tags`, and the full suite.

## Risks
- An expression that previously *accidentally* parsed as two names now parses as one
  name. Those expressions always raised an error before, so nothing that worked before
  changes meaning.
- Tags differing only by internal whitespace runs (`a b` vs `a  b`) become
  indistinguishable in the filter. Accepted: it is a pathological case, and the trade is
  forgiving matching for user-typed expressions.

## Validation
- `python -m unittest tests.test_tag_expression tests.test_roi_manager_tags`
- `python -m unittest discover -s tests -t .`
