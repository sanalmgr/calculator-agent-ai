# -*- coding: utf-8 -*-
"""
LangGraph calculator with separate input, internal,
and output schemas.

Studio input:
    user_request

Studio output:
    response
"""

import ast
import json
import re
from typing import Optional, TypedDict

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

from langgraph.graph import (
    END,
    START,
    StateGraph
)


# ---------------------------------------------------------
# Hugging Face model configuration
# ---------------------------------------------------------

MODEL_NAME = "google/flan-t5-small"


print("Loading Hugging Face tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print("Loading Hugging Face model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Hugging Face model loaded successfully.")


# ---------------------------------------------------------
# LangGraph schemas
# ---------------------------------------------------------

class CalculatorInput(TypedDict):
    """
    Fields supplied by the user.

    LangSmith Studio should display only this field
    in the input panel.
    """

    user_request: str


class CalculatorOutput(TypedDict):
    """
    Fields returned when graph execution finishes.
    """

    response: str


class CalculatorState(TypedDict, total=False):
    """
    Complete internal state used by graph nodes.
    """

    user_request: str

    operation: Optional[str]

    number_1: Optional[float]

    number_2: Optional[float]

    result: Optional[float]

    error: Optional[str]

    response: Optional[str]


# ---------------------------------------------------------
# Calculator tools
# ---------------------------------------------------------

def add_tool(
    number_1: float,
    number_2: float
) -> float:
    """Add two numbers."""

    return number_1 + number_2


def subtract_tool(
    number_1: float,
    number_2: float
) -> float:
    """Subtract the second number from the first."""

    return number_1 - number_2


def multiply_tool(
    number_1: float,
    number_2: float
) -> float:
    """Multiply two numbers."""

    return number_1 * number_2


def divide_tool(
    number_1: float,
    number_2: float
) -> float:
    """Divide the first number by the second."""

    if number_2 == 0:
        raise ZeroDivisionError(
            "Division by zero is not allowed."
        )

    return number_1 / number_2


# ---------------------------------------------------------
# Model helper
# ---------------------------------------------------------

def generate_model_response(
    prompt: str
) -> str:
    """
    Generate text using the local Hugging Face model.
    """

    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded_input,
            max_new_tokens=80,
            do_sample=False
        )

    generated_text = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True
    )

    return generated_text.strip()


# ---------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------

def extract_json_object(
    text: str
):
    """
    Extract a JSON object from model output.
    """

    if not text:
        return None

    match = re.search(
        r"\{.*?\}",
        text,
        flags=re.DOTALL
    )

    if match is None:
        return None

    json_text = match.group(0)

    try:
        return json.loads(json_text)

    except json.JSONDecodeError:
        try:
            return ast.literal_eval(
                json_text
            )

        except Exception:
            return None


def normalise_operation(
    operation
) -> Optional[str]:
    """
    Convert operation variants into one
    supported operation name.
    """

    if operation is None:
        return None

    operation = str(
        operation
    ).strip().lower()

    operation_map = {
        "add": "add",
        "addition": "add",
        "plus": "add",
        "sum": "add",
        "+": "add",

        "subtract": "subtract",
        "subtraction": "subtract",
        "minus": "subtract",
        "difference": "subtract",
        "-": "subtract",

        "multiply": "multiply",
        "multiplication": "multiply",
        "times": "multiply",
        "product": "multiply",
        "x": "multiply",
        "*": "multiply",

        "divide": "divide",
        "division": "divide",
        "divided": "divide",
        "quotient": "divide",
        "/": "divide"
    }

    return operation_map.get(
        operation
    )


def fallback_parse_request(
    user_request: str
):
    """
    Rule-based fallback parser.

    This runs when the model does not return
    usable structured output.
    """

    text = user_request.lower()

    number_matches = re.findall(
        r"-?\d+(?:\.\d+)?",
        text
    )

    if len(number_matches) < 2:
        return None, None, None

    numbers = [
        float(number)
        for number in number_matches
    ]

    number_1 = numbers[0]
    number_2 = numbers[1]

    if any(
        word in text
        for word in [
            "add",
            "plus",
            "sum"
        ]
    ):
        operation = "add"

    elif any(
        word in text
        for word in [
            "subtract",
            "minus",
            "difference"
        ]
    ):
        operation = "subtract"

        # Handle:
        # "Subtract 5 from 20"
        if " from " in text:
            number_1 = numbers[1]
            number_2 = numbers[0]

    elif any(
        word in text
        for word in [
            "multiply",
            "times",
            "product"
        ]
    ):
        operation = "multiply"

    elif any(
        word in text
        for word in [
            "divide",
            "divided",
            "quotient"
        ]
    ):
        operation = "divide"

    else:
        operation = None

    return (
        operation,
        number_1,
        number_2
    )


def format_number(
    value
):
    """
    Format whole-number floats without
    a decimal point.
    """

    if (
        isinstance(value, float)
        and value.is_integer()
    ):
        return int(value)

    return value


# ---------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------

def understand_request_node(
    state: CalculatorState
) -> CalculatorState:
    """
    Use the Hugging Face model to extract
    an operation and two numbers.
    """

    user_request = state.get(
        "user_request",
        ""
    ).strip()

    if not user_request:
        return {
            "error": (
                "Please enter a calculation request."
            )
        }

    prompt = f"""
You are a calculator routing agent.

Identify:
- the operation
- the first number
- the second number

The operation must be exactly one of:
add
subtract
multiply
divide

Return only valid JSON.

Required format:
{{"operation": "add", "number_1": 5, "number_2": 3}}

For subtraction, preserve the correct operand order.

Example:
User request: Subtract 5 from 20
Output:
{{"operation": "subtract", "number_1": 20, "number_2": 5}}

User request:
{user_request}
""".strip()

    try:
        generated_text = generate_model_response(
            prompt
        )

        print(
            "Model output:",
            generated_text
        )

        parsed_data = extract_json_object(
            generated_text
        )

        if parsed_data is not None:
            operation = normalise_operation(
                parsed_data.get(
                    "operation"
                )
            )

            number_1_value = parsed_data.get(
                "number_1"
            )

            number_2_value = parsed_data.get(
                "number_2"
            )

            if (
                operation is not None
                and number_1_value is not None
                and number_2_value is not None
            ):
                return {
                    "operation": operation,
                    "number_1": float(
                        number_1_value
                    ),
                    "number_2": float(
                        number_2_value
                    ),
                    "error": None
                }

    except Exception as error:
        print(
            "Model parsing error:",
            error
        )

    operation, number_1, number_2 = (
        fallback_parse_request(
            user_request
        )
    )

    if operation is None:
        return {
            "error": (
                "I could not identify the "
                "mathematical operation."
            )
        }

    if (
        number_1 is None
        or number_2 is None
    ):
        return {
            "error": (
                "I could not identify two numbers."
            )
        }

    return {
        "operation": operation,
        "number_1": number_1,
        "number_2": number_2,
        "error": None
    }


def add_node(
    state: CalculatorState
) -> CalculatorState:
    """Run the addition tool."""

    result = add_tool(
        state["number_1"],
        state["number_2"]
    )

    return {
        "result": result,
        "error": None
    }


def subtract_node(
    state: CalculatorState
) -> CalculatorState:
    """Run the subtraction tool."""

    result = subtract_tool(
        state["number_1"],
        state["number_2"]
    )

    return {
        "result": result,
        "error": None
    }


def multiply_node(
    state: CalculatorState
) -> CalculatorState:
    """Run the multiplication tool."""

    result = multiply_tool(
        state["number_1"],
        state["number_2"]
    )

    return {
        "result": result,
        "error": None
    }


def divide_node(
    state: CalculatorState
) -> CalculatorState:
    """Run the division tool."""

    try:
        result = divide_tool(
            state["number_1"],
            state["number_2"]
        )

        return {
            "result": result,
            "error": None
        }

    except ZeroDivisionError as error:
        return {
            "error": str(error)
        }


def error_node(
    state: CalculatorState
) -> CalculatorState:
    """Preserve or create an error message."""

    error_message = state.get(
        "error"
    )

    if not error_message:
        error_message = (
            "The requested operation "
            "is not supported."
        )

    return {
        "error": error_message
    }


def format_response_node(
    state: CalculatorState
) -> CalculatorState:
    """Create the final user-facing response."""

    if state.get("error"):
        return {
            "response": (
                f"Error: {state['error']}"
            )
        }

    operation = state["operation"]

    symbol_map = {
        "add": "+",
        "subtract": "-",
        "multiply": "×",
        "divide": "÷"
    }

    symbol = symbol_map.get(
        operation,
        operation
    )

    number_1 = format_number(
        state["number_1"]
    )

    number_2 = format_number(
        state["number_2"]
    )

    result = format_number(
        state["result"]
    )

    response = (
        f"Agent selected tool: {operation}\n\n"
        f"{number_1} {symbol} "
        f"{number_2} = {result}"
    )

    return {
        "response": response
    }


# ---------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------

def route_operation(
    state: CalculatorState
) -> str:
    """
    Return the name of the next node.
    """

    if state.get("error"):
        return "error"

    operation = state.get(
        "operation"
    )

    supported_operations = {
        "add",
        "subtract",
        "multiply",
        "divide"
    }

    if operation in supported_operations:
        return operation

    return "error"


# ---------------------------------------------------------
# Build the graph
# ---------------------------------------------------------

graph_builder = StateGraph(
    CalculatorState,
    input_schema=CalculatorInput,
    output_schema=CalculatorOutput
)


graph_builder.add_node(
    "understand_request",
    understand_request_node
)

graph_builder.add_node(
    "add",
    add_node
)

graph_builder.add_node(
    "subtract",
    subtract_node
)

graph_builder.add_node(
    "multiply",
    multiply_node
)

graph_builder.add_node(
    "divide",
    divide_node
)

graph_builder.add_node(
    "error",
    error_node
)

graph_builder.add_node(
    "format_response",
    format_response_node
)


graph_builder.add_edge(
    START,
    "understand_request"
)


graph_builder.add_conditional_edges(
    "understand_request",
    route_operation,
    {
        "add": "add",
        "subtract": "subtract",
        "multiply": "multiply",
        "divide": "divide",
        "error": "error"
    }
)


for node_name in [
    "add",
    "subtract",
    "multiply",
    "divide",
    "error"
]:
    graph_builder.add_edge(
        node_name,
        "format_response"
    )


graph_builder.add_edge(
    "format_response",
    END
)


# LangGraph Studio imports this variable.
calculator_graph = graph_builder.compile()