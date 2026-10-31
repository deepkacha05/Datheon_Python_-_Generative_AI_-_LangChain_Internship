"""Safe Python Calculator Tool using Abstract Syntax Tree (AST) parsing.

Security Highlights:
- Evaluates mathematical expressions WITHOUT using eval(), exec(), or arbitrary code execution.
- Strictly parses and traverses the AST using a restricted whitelist of mathematical nodes.
- Whitelists only safe binary operations (+, -, *, /, //, %, **), unary operators (+, -),
  safe math functions (sqrt, log, sin, cos, tan, abs, round, etc.), and math constants (pi, e).
- Immediately halts and rejects attribute lookups (__class__, __subclasses__), imports,
  variable assignments, subscripts, lambdas, loops, or malicious injection payloads.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, Set
from task3_ai_agent.tools.base import BaseAgentTool, CalculatorInput, ToolResult

SAFE_MATH_FUNCTIONS: Dict[str, Any] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
    "pow": pow,
}

SAFE_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

MAX_EXPONENT = 1000  # Prevent denial of service via astronomical exponents like 9**999999


def _safe_eval_ast_node(node: ast.AST) -> float:
    """Recursively evaluates an AST node with strict type and operator verification."""
    if isinstance(node, ast.Expression):
        return _safe_eval_ast_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type '{type(node.value).__name__}'. Only numbers are permitted.")

    if isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"Disallowed or undefined variable: '{node.id}'. Only constants ('pi', 'e') are allowed.")

    if isinstance(node, ast.BinOp):
        left_val = _safe_eval_ast_node(node.left)
        right_val = _safe_eval_ast_node(node.right)

        if isinstance(node.op, ast.Add):
            return left_val + right_val
        elif isinstance(node.op, ast.Sub):
            return left_val - right_val
        elif isinstance(node.op, ast.Mult):
            return left_val * right_val
        elif isinstance(node.op, ast.Div):
            if right_val == 0:
                raise ZeroDivisionError("Division by zero is undefined.")
            return left_val / right_val
        elif isinstance(node.op, ast.FloorDiv):
            if right_val == 0:
                raise ZeroDivisionError("Integer division by zero is undefined.")
            return left_val // right_val
        elif isinstance(node.op, ast.Mod):
            if right_val == 0:
                raise ZeroDivisionError("Modulo by zero is undefined.")
            return left_val % right_val
        elif isinstance(node.op, ast.Pow):
            if abs(right_val) > MAX_EXPONENT:
                raise ValueError(f"Exponent magnitude {right_val} exceeds safety limit of {MAX_EXPONENT}.")
            return left_val ** right_val
        else:
            raise ValueError(f"Disallowed binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand_val = _safe_eval_ast_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand_val
        elif isinstance(node.op, ast.USub):
            return -operand_val
        else:
            raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Disallowed function call structure. Attribute calls like obj.func() are forbidden.")
        fn_name = node.func.id
        if fn_name not in SAFE_MATH_FUNCTIONS:
            allowed = ", ".join(sorted(SAFE_MATH_FUNCTIONS.keys()))
            raise ValueError(f"Unauthorized function '{fn_name}'. Permitted math functions: {allowed}")

        evaluated_args = [_safe_eval_ast_node(arg) for arg in node.args]
        return float(SAFE_MATH_FUNCTIONS[fn_name](*evaluated_args))

    # Explicit rejection of all other AST nodes
    raise ValueError(f"Disallowed syntax or operation: '{type(node).__name__}'. Only pure arithmetic expressions are allowed.")


def safe_calculate(expression: str) -> float:
    """Parses and computes the result of an arithmetic string securely.

    Args:
        expression: Mathematical expression string.

    Returns:
        float result of the calculation.

    Raises:
        ValueError, SyntaxError, ZeroDivisionError on illegal syntax or operations.
    """
    clean_expr = str(expression).strip(" \t\n\r\"'`")
    if "expression" in clean_expr and ("{" in clean_expr or ":" in clean_expr):
        try:
            import json
            data = json.loads(clean_expr)
            if isinstance(data, dict) and "expression" in data:
                clean_expr = str(data["expression"]).strip(" \t\n\r\"'`")
        except Exception:
            import re
            match = re.search(r"['\"]expression['\"]\s*:\s*['\"]([^'\"\}\n\r`]+)['\"]", clean_expr)
            if match:
                clean_expr = match.group(1).strip(" \t\n\r\"'`")

    if not clean_expr:
        raise ValueError("Expression is empty.")

    # Parse into abstract syntax tree
    tree = ast.parse(clean_expr, mode="eval")
    result = _safe_eval_ast_node(tree)
    return result


class CalculatorTool(BaseAgentTool):
    """Secure calculator tool evaluated via AST for mathematical calculations."""

    name = "calculator"
    description = (
        "Useful for calculating arithmetic expressions, percentages, growth rates, "
        "and scientific formulas (e.g. '15000 * 1.15', 'sqrt(144) + 25', '100 / (5 + 5)'). "
        "Does NOT execute arbitrary Python code. Input must be a valid mathematical expression string."
    )
    args_schema = CalculatorInput

    def run(self, expression: str) -> ToolResult:
        try:
            val = safe_calculate(expression)
            # Format nicely: if integer float, display without decimals
            if val.is_integer():
                formatted = str(int(val))
            else:
                formatted = f"{val:.6g}"

            return ToolResult(
                success=True,
                output=formatted,
                metadata={"expression": expression, "raw_result": val},
            )
        except ZeroDivisionError as zde:
            return ToolResult(
                success=False,
                output="",
                error_message=f"Math error: {str(zde)}",
            )
        except (ValueError, SyntaxError, TypeError) as exc:
            return ToolResult(
                success=False,
                output="",
                error_message=f"Invalid arithmetic expression: {str(exc)}",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error_message=f"Unexpected calculation error: {str(exc)}",
            )
