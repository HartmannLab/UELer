"""Utilities for parsing boolean tag expressions.

The ROI browser allows users to compose tag filters using familiar
boolean operators. This module provides a small parser that converts the
user input into an evaluable predicate so the gallery can quickly decide
whether an ROI matches the active expression.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Set

__all__ = [
    "TagExpressionError",
    "Token",
    "compile_tag_expression",
    "evaluate_tag_expression",
]


class TagExpressionError(ValueError):
    """Raised when a tag expression cannot be parsed."""


@dataclass(frozen=True)
class Token:
    """Simple representation of a lexed token."""

    kind: str
    value: str


_OPERATOR_PRECEDENCE = {"!": 3, "&": 2, "|": 1}
_RIGHT_ASSOCIATIVE = {"!"}
_VALID_NAME_CHARS = set("+-._:/")


def _tokenize(expression: str) -> List[Token]:
    tokens: List[Token] = []
    buffer: List[str] = []
    length = len(expression)
    index = 0

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            token_text = "".join(buffer).strip()
            if token_text:
                tokens.append(Token("name", token_text))
            buffer = []

    while index < length:
        char = expression[index]
        if char.isspace():
            flush_buffer()
            index += 1
            continue
        if char in "()&|!":
            flush_buffer()
            tokens.append(Token("operator" if char in "&|!" else "paren", char))
            index += 1
            continue
        if char in ('"', "'"):
            quote = char
            index += 1
            start = index
            escaped: List[str] = []
            while index < length:
                current = expression[index]
                if current == "\\" and index + 1 < length:
                    escaped.append(expression[index + 1])
                    index += 2
                    continue
                if current == quote:
                    break
                escaped.append(current)
                index += 1
            else:
                raise TagExpressionError("Unclosed quoted string in expression.")
            tokens.append(Token("name", "".join(escaped)))
            index += 1  # skip closing quote
            continue
        if char.isalnum() or char in _VALID_NAME_CHARS:
            buffer.append(char)
            index += 1
            continue
        raise TagExpressionError(f"Unsupported character '{char}' in expression.")

    flush_buffer()
    return tokens


def _to_postfix(tokens: Sequence[Token]) -> List[Token]:
    output: List[Token] = []
    stack: List[Token] = []

    for token in tokens:
        if token.kind == "name":
            output.append(token)
            continue
        if token.kind == "operator":
            precedence = _OPERATOR_PRECEDENCE[token.value]
            while stack:
                top = stack[-1]
                if top.kind != "operator":
                    break
                top_precedence = _OPERATOR_PRECEDENCE[top.value]
                if (
                    top_precedence > precedence
                    or (
                        top_precedence == precedence
                        and token.value not in _RIGHT_ASSOCIATIVE
                    )
                ):
                    output.append(stack.pop())
                    continue
                break
            stack.append(token)
            continue
        if token.kind == "paren":
            if token.value == "(":
                stack.append(token)
            else:
                # Pop until matching opening paren encountered
                while stack and stack[-1].value != "(":
                    output.append(stack.pop())
                if not stack:
                    raise TagExpressionError("Mismatched parentheses in expression.")
                stack.pop()  # discard "("
            continue
        raise TagExpressionError(f"Unsupported token kind: {token.kind}")

    while stack:
        token = stack.pop()
        if token.kind == "paren":
            raise TagExpressionError("Mismatched parentheses in expression.")
        output.append(token)

    return output


def _evaluate_postfix(postfix: Sequence[Token], tag_set: Set[str]) -> bool:
    stack: List[bool] = []
    for token in postfix:
        if token.kind == "name":
            stack.append(token.value in tag_set)
            continue
        if token.kind == "operator":
            if token.value == "!":
                if not stack:
                    raise TagExpressionError("Missing operand for '!'.")
                operand = stack.pop()
                stack.append(not operand)
                continue
            if len(stack) < 2:
                raise TagExpressionError("Missing operands for binary operator.")
            rhs = stack.pop()
            lhs = stack.pop()
            if token.value == "&":
                stack.append(lhs and rhs)
            elif token.value == "|":
                stack.append(lhs or rhs)
            else:
                raise TagExpressionError(f"Unsupported operator '{token.value}'.")
            continue
        raise TagExpressionError(f"Unexpected token '{token.value}'.")

    if len(stack) != 1:
        raise TagExpressionError("Expression did not reduce to a single value.")
    return stack[0]


def compile_tag_expression(expression: str) -> Callable[[Iterable[str]], bool]:
    """Compile *expression* into a predicate over tag collections.

    Parameters
    ----------
    expression:
        The textual representation provided by the user.

    Returns
    -------
    Callable[[Iterable[str]], bool]
        A predicate that returns ``True`` when the supplied tags satisfy the
        compiled expression.
    """

    stripped = expression.strip()
    if not stripped:
        raise TagExpressionError("Empty expression.")

    tokens = _tokenize(stripped)
    if not tokens:
        raise TagExpressionError("Empty expression.")
    postfix = _to_postfix(tokens)

    # Validate structure eagerly so obvious syntax issues surface during
    # compilation rather than at evaluation time.
    try:
        _evaluate_postfix(postfix, set())
    except TagExpressionError as exc:
        raise TagExpressionError(str(exc)) from exc

    def predicate(tags: Iterable[str]) -> bool:
        tag_set = {str(tag).strip() for tag in tags if str(tag).strip()}
        return _evaluate_postfix(postfix, tag_set)

    return predicate


def evaluate_tag_expression(expression: str, tags: Iterable[str]) -> bool:
    """Convenience wrapper around :func:`compile_tag_expression`."""

    return compile_tag_expression(expression)(tags)
