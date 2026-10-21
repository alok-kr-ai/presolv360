"""Calculator tool: deterministic, AST-based, never uses ``eval``."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from pydantic import Field

from tools.base import ErrorCode, StrictArgs, ToolContext, ToolFailure, make_tool

MAX_EXPR_LEN = 200
MAX_NODES = 120
MAX_ABS_EXPONENT = 1000
MAX_RESULT_BITS = 1024

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "min": min, "max": max, "abs": abs}

Number = int | float


def _unsupported(what: str) -> ToolFailure:
    return ToolFailure(
        ErrorCode.UNSUPPORTED,
        f"Unsupported expression element: {what}.",
        hint="Only numbers, + - * / // % **, parentheses and round/min/max/abs are allowed. "
        "Rewrite the question as a plain arithmetic expression.",
    )


def _check_pow(base: Number, exp: Number) -> None:
    if abs(exp) > MAX_ABS_EXPONENT:
        raise ToolFailure(
            ErrorCode.MATH_ERROR,
            f"Exponent {exp} is too large (limit {MAX_ABS_EXPONENT}).",
            hint="Use a smaller exponent.",
        )
    if isinstance(base, int) and isinstance(exp, int) and exp > 0:
        if abs(base).bit_length() * exp > MAX_RESULT_BITS:
            raise ToolFailure(ErrorCode.MATH_ERROR, "The result would be too large to compute.", hint="Use smaller numbers.")


def _eval(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _unsupported(f"constant {value!r}")
        return value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise _unsupported(type(node.op).__name__)
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            _check_pow(left, right)
        try:
            result = op(left, right)
        except ZeroDivisionError:
            raise ToolFailure(ErrorCode.MATH_ERROR, "Division by zero.", hint="The divisor must not be zero.") from None
        except OverflowError:
            raise ToolFailure(ErrorCode.MATH_ERROR, "The result is too large.") from None
        if isinstance(result, complex):
            raise ToolFailure(ErrorCode.MATH_ERROR, "The result is not a real number.")
        return result
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise _unsupported(type(node.op).__name__)
        return op(_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise _unsupported("function call")
        args = [_eval(a) for a in node.args]
        if not args or len(args) > 10:
            raise _unsupported(f"{node.func.id}() with {len(args)} arguments")
        if node.func.id == "round" and len(args) > 2:
            raise _unsupported("round() with more than two arguments")
        if node.func.id == "abs" and len(args) != 1:
            raise _unsupported("abs() with more than one argument")
        try:
            return _FUNCS[node.func.id](*args)
        except (TypeError, ValueError) as exc:
            raise ToolFailure(ErrorCode.INVALID_ARGS, f"Bad arguments to {node.func.id}(): {exc}") from None
    kind = type(node).__name__
    raise _unsupported({"Name": "name", "Attribute": "attribute access", "Subscript": "indexing"}.get(kind, kind))


def _format(value: Number) -> Number:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ToolFailure(ErrorCode.MATH_ERROR, "The result is not a finite number.")
        value = float(f"{value:.12g}")
        if value.is_integer() and abs(value) < 1e15:
            return int(value)
    return value


def safe_calculate(expression: str) -> dict[str, Any]:
    expr = expression.strip()
    if not expr:
        raise ToolFailure(ErrorCode.INVALID_ARGS, "The expression is empty.")
    if len(expr) > MAX_EXPR_LEN:
        raise ToolFailure(ErrorCode.INVALID_ARGS, f"Expression too long (limit {MAX_EXPR_LEN} characters).")
    expr = expr.replace("^", "**") if "^" in expr and "**" not in expr else expr
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ToolFailure(
            ErrorCode.INVALID_ARGS,
            "That is not a valid arithmetic expression.",
            hint="Write it like (12.5 + 7) * 3 / 2.",
        ) from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ToolFailure(ErrorCode.INVALID_ARGS, "Expression is too complex.")
    try:
        value = _eval(tree)
    except RecursionError:
        raise ToolFailure(ErrorCode.INVALID_ARGS, "Expression is too deeply nested.") from None
    return {"expression": expression.strip(), "result": _format(value)}


class CalculateArgs(StrictArgs):
    expression: str = Field(
        min_length=1,
        max_length=MAX_EXPR_LEN,
        description="Arithmetic expression using numbers, + - * / // % **, parentheses and round/min/max/abs. "
        "Example: '(1250 * 0.18) + 1250'.",
    )


def _handler(args: CalculateArgs, ctx: ToolContext) -> dict[str, Any]:
    return safe_calculate(args.expression)


def build_calculator_tools() -> list:
    return [
        make_tool(
            "calculate",
            "Evaluate an arithmetic expression exactly (deterministic; no code execution). "
            "USE for any non-trivial arithmetic, percentages, totals, unit conversions with known factors, "
            "or when the user asks to multiply/add/divide numbers. "
            "DO NOT use for calendar dates/times, for looking up contacts, or for counting items in conversation.",
            CalculateArgs,
            _handler,
            group="calculator",
        )
    ]
