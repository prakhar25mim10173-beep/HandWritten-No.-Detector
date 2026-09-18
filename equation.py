import re

def solve_equation(equation):
    """
    Safely evaluate a basic mathematical expression.

    Supported operators:
    + addition
    - subtraction
    * multiplication
    / division
    """

    equation = equation.replace("×", "*")
    equation = equation.replace("÷", "/")
    equation = equation.replace(" ", "")

    # Only allow numbers and basic mathematical operators
    if not re.fullmatch(r"[0-9+\-*/().]+", equation):
        raise ValueError(
            "Expression contains unsupported characters."
        )

    try:
        result = eval(equation, {"__builtins__": None}, {})
        return result

    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero.")

    except Exception:
        raise ValueError("Invalid mathematical expression.")


def format_equation(equation):
    """Make an equation easier to read."""

    equation = equation.replace("*", "×")
    equation = equation.replace("/", "÷")

    return equation
