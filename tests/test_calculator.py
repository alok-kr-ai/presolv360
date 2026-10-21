import pytest

from tools.base import ErrorCode, ToolFailure
from tools.calculator import safe_calculate


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2 * 3", 6),
        ("(1250 * 0.18) + 1250", 1475),
        ("10 / 4", 2.5),
        ("7 // 2", 3),
        ("-5 + 2", -3),
        ("2 ** 10", 1024),
        ("round(3.14159, 2)", 3.14),
        ("max(1, 5, 3) + min(4, 2)", 7),
        ("abs(-9)", 9),
        ("0.1 + 0.2", 0.3),
        ("2^3", 8),
    ],
)
def test_evaluates_safely(expr, expected):
    assert safe_calculate(expr)["result"] == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd')",
        "().__class__",
        "x + 1",
        "'a' * 3",
        "[1, 2][0]",
        "lambda: 1",
        "(1).real",
        "True + 1",
        "1 if 2 else 3",
        "print(1)",
    ],
)
def test_rejects_unsafe_expressions(expr):
    with pytest.raises(ToolFailure) as exc:
        safe_calculate(expr)
    assert exc.value.code in (ErrorCode.UNSUPPORTED, ErrorCode.INVALID_ARGS)
    assert not exc.value.retryable


def test_division_by_zero_is_math_error():
    with pytest.raises(ToolFailure) as exc:
        safe_calculate("1 / 0")
    assert exc.value.code is ErrorCode.MATH_ERROR


@pytest.mark.parametrize("expr", ["9 ** 9 ** 9", "10 ** 5000", "2 ** 100000", "(-8) ** 0.5", "1e308 * 10"])
def test_huge_or_non_real_results_are_capped(expr):
    with pytest.raises(ToolFailure) as exc:
        safe_calculate(expr)
    assert exc.value.code is ErrorCode.MATH_ERROR


def test_syntax_errors_and_length_limits():
    with pytest.raises(ToolFailure) as exc:
        safe_calculate("2 +* 3")
    assert exc.value.code is ErrorCode.INVALID_ARGS
    with pytest.raises(ToolFailure) as exc:
        safe_calculate("1+" * 150 + "1")
    assert exc.value.code is ErrorCode.INVALID_ARGS
    with pytest.raises(ToolFailure):
        safe_calculate("   ")


def test_no_eval_in_source():
    import inspect

    import tools.calculator as calc

    src = inspect.getsource(calc)
    assert "eval(" not in src.replace("_eval(", "").replace("literal_eval", "")
    assert "exec(" not in src
