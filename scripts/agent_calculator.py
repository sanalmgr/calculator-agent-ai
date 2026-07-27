# -*- coding: utf-8 -*-
"""
Agent Calculator using:

- LangGraph
- Hugging Face Transformers
- Gradio
- Local open-source model

This application does not use OpenAI, Anthropic,
or another proprietary LLM API.
"""

import ast
import json
import re
from typing import Optional, TypedDict

import gradio as gr
from transformers import pipeline

from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

MODEL_NAME = "google/flan-t5-small"

print("Loading Hugging Face model...")

llm = pipeline(
    task="text2text-generation",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME
)

print("Model loaded successfully.")


# ---------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------

class CalculatorState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    user_request: str

    operation: Optional[str]

    number_1: Optional[float]

    number_2: Optional[float]

    result: Optional[float]

    error: Optional[str]

    response: Optional[str]

    # Outputs displayed in Gradio
    llm_output: Optional[str]

    interpretation_output: Optional[str]

    routing_output: Optional[str]

    tool_output: Optional[str]

    formatter_output: Optional[str]


# ---------------------------------------------------------
# Calculator tools
# ---------------------------------------------------------

def add_tool(number_1: float, number_2: float) -> float:
    """Add two numbers."""

    return number_1 + number_2


def subtract_tool(number_1: float, number_2: float) -> float:
    """Subtract the second number from the first."""

    return number_1 - number_2


def multiply_tool(number_1: float, number_2: float) -> float:
    """Multiply two numbers."""

    return number_1 * number_2


def divide_tool(number_1: float, number_2: float) -> float:
    """Divide the first number by the second."""

    if number_2 == 0:
        raise ZeroDivisionError(
            "Division by zero is not allowed."
        )

    return number_1 / number_2


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def format_number(value):
    """
    Display whole numbers without a decimal point.

    Example:
    5.0 becomes 5
    5.25 remains 5.25
    """

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def extract_json_object(text: str):
    """
    Extract a JSON object from model output.

    The model may occasionally include additional text.
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
            return ast.literal_eval(json_text)

        except Exception:
            return None


def normalise_operation(operation: str):
    """
    Convert model output into a supported operation.
    """

    if not operation:
        return None

    operation = str(operation).strip().lower()

    operation_map = {
        "add": "add",
        "addition": "add",
        "plus": "add",
        "+": "add",

        "subtract": "subtract",
        "subtraction": "subtract",
        "minus": "subtract",
        "-": "subtract",

        "multiply": "multiply",
        "multiplication": "multiply",
        "times": "multiply",
        "x": "multiply",
        "*": "multiply",

        "divide": "divide",
        "division": "divide",
        "divided": "divide",
        "/": "divide"
    }

    return operation_map.get(operation)


def fallback_parse_request(user_request: str):
    """
    Rule-based fallback parser.

    This is used when the language model does not return
    valid structured output.
    """

    text = user_request.lower()

    number_matches = re.findall(
        r"-?\d+(?:\.\d+)?",
        text
    )

    if len(number_matches) < 2:
        return None, None, None

    number_1 = float(number_matches[0])
    number_2 = float(number_matches[1])

    if any(
        word in text
        for word in ["add", "plus", "sum"]
    ):
        operation = "add"

    elif any(
        word in text
        for word in ["subtract", "minus", "difference"]
    ):
        operation = "subtract"

    elif any(
        word in text
        for word in ["multiply", "times", "product"]
    ):
        operation = "multiply"

    elif any(
        word in text
        for word in ["divide", "divided", "quotient"]
    ):
        operation = "divide"

    else:
        operation = None

    return operation, number_1, number_2


# ---------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------

def understand_request_node(
    state: CalculatorState
) -> CalculatorState:
    """
    Use the Hugging Face model to interpret the request.
    """

    user_request = state.get(
        "user_request",
        ""
    ).strip()

    if not user_request:
        return {
            "error": "Please enter a calculation request.",
            "llm_output": "No model request was made.",
            "interpretation_output": (
                "No user request was provided."
            )
        }

    prompt = f"""
You are a calculator routing agent.

Read the user request and identify:
1. the mathematical operation
2. the first number
3. the second number

The operation must be exactly one of:
add
subtract
multiply
divide

Return only valid JSON.

Use this exact format:
{{"operation": "add", "number_1": 5, "number_2": 3}}

User request:
{user_request}
""".strip()

    generated_text = ""

    try:
        model_output = llm(
            prompt,
            max_new_tokens=80,
            do_sample=False
        )

        generated_text = model_output[0][
            "generated_text"
        ]

        print("\n--- LLM OUTPUT ---")
        print(generated_text)

        parsed_data = extract_json_object(
            generated_text
        )

        if parsed_data is not None:
            operation = normalise_operation(
                parsed_data.get("operation")
            )

            number_1 = float(
                parsed_data.get("number_1")
            )

            number_2 = float(
                parsed_data.get("number_2")
            )

            if operation is not None:
                interpretation = (
                    f"Operation: {operation}\n"
                    f"Number 1: {format_number(number_1)}\n"
                    f"Number 2: {format_number(number_2)}\n"
                    f"Source: Hugging Face model"
                )

                print("\n--- INTERPRETATION OUTPUT ---")
                print(interpretation)

                return {
                    "operation": operation,
                    "number_1": number_1,
                    "number_2": number_2,
                    "error": None,
                    "llm_output": generated_text,
                    "interpretation_output": interpretation
                }

    except Exception as error:
        print(
            "LLM parsing error:",
            error
        )

    # Fallback if the LLM output cannot be parsed
    operation, number_1, number_2 = (
        fallback_parse_request(user_request)
    )

    if operation is None:
        interpretation = (
            "The operation could not be identified."
        )

        return {
            "error": (
                "I could not identify the operation. "
                "Try a request such as "
                "'Multiply 8 by 7'."
            ),
            "llm_output": (
                generated_text
                or "No model output was generated."
            ),
            "interpretation_output": interpretation
        }

    if number_1 is None or number_2 is None:
        interpretation = (
            "Two numbers could not be identified."
        )

        return {
            "error": (
                "I could not identify two numbers "
                "in the request."
            ),
            "llm_output": (
                generated_text
                or "No model output was generated."
            ),
            "interpretation_output": interpretation
        }

    interpretation = (
        f"Operation: {operation}\n"
        f"Number 1: {format_number(number_1)}\n"
        f"Number 2: {format_number(number_2)}\n"
        f"Source: fallback parser"
    )

    print("\n--- INTERPRETATION OUTPUT ---")
    print(interpretation)

    return {
        "operation": operation,
        "number_1": number_1,
        "number_2": number_2,
        "error": None,
        "llm_output": (
            generated_text
            or "The model did not return usable JSON."
        ),
        "interpretation_output": interpretation
    }


def add_node(
    state: CalculatorState
) -> CalculatorState:
    """Execute the addition tool."""

    number_1 = state["number_1"]
    number_2 = state["number_2"]

    routing_output = "LangGraph selected the add node."

    try:
        result = add_tool(
            number_1,
            number_2
        )

        tool_output = (
            f"Tool: add_tool\n"
            f"Input: {format_number(number_1)}, "
            f"{format_number(number_2)}\n"
            f"Result: {format_number(result)}"
        )

        print("\n--- ROUTING OUTPUT ---")
        print(routing_output)

        print("\n--- TOOL OUTPUT ---")
        print(tool_output)

        return {
            "result": result,
            "error": None,
            "routing_output": routing_output,
            "tool_output": tool_output
        }

    except Exception as error:
        return {
            "error": str(error),
            "routing_output": routing_output,
            "tool_output": f"Tool error: {error}"
        }


def subtract_node(
    state: CalculatorState
) -> CalculatorState:
    """Execute the subtraction tool."""

    number_1 = state["number_1"]
    number_2 = state["number_2"]

    routing_output = (
        "LangGraph selected the subtract node."
    )

    try:
        result = subtract_tool(
            number_1,
            number_2
        )

        tool_output = (
            f"Tool: subtract_tool\n"
            f"Input: {format_number(number_1)}, "
            f"{format_number(number_2)}\n"
            f"Result: {format_number(result)}"
        )

        print("\n--- ROUTING OUTPUT ---")
        print(routing_output)

        print("\n--- TOOL OUTPUT ---")
        print(tool_output)

        return {
            "result": result,
            "error": None,
            "routing_output": routing_output,
            "tool_output": tool_output
        }

    except Exception as error:
        return {
            "error": str(error),
            "routing_output": routing_output,
            "tool_output": f"Tool error: {error}"
        }


def multiply_node(
    state: CalculatorState
) -> CalculatorState:
    """Execute the multiplication tool."""

    number_1 = state["number_1"]
    number_2 = state["number_2"]

    routing_output = (
        "LangGraph selected the multiply node."
    )

    try:
        result = multiply_tool(
            number_1,
            number_2
        )

        tool_output = (
            f"Tool: multiply_tool\n"
            f"Input: {format_number(number_1)}, "
            f"{format_number(number_2)}\n"
            f"Result: {format_number(result)}"
        )

        print("\n--- ROUTING OUTPUT ---")
        print(routing_output)

        print("\n--- TOOL OUTPUT ---")
        print(tool_output)

        return {
            "result": result,
            "error": None,
            "routing_output": routing_output,
            "tool_output": tool_output
        }

    except Exception as error:
        return {
            "error": str(error),
            "routing_output": routing_output,
            "tool_output": f"Tool error: {error}"
        }


def divide_node(
    state: CalculatorState
) -> CalculatorState:
    """Execute the division tool."""

    number_1 = state["number_1"]
    number_2 = state["number_2"]

    routing_output = (
        "LangGraph selected the divide node."
    )

    try:
        result = divide_tool(
            number_1,
            number_2
        )

        tool_output = (
            f"Tool: divide_tool\n"
            f"Input: {format_number(number_1)}, "
            f"{format_number(number_2)}\n"
            f"Result: {format_number(result)}"
        )

        print("\n--- ROUTING OUTPUT ---")
        print(routing_output)

        print("\n--- TOOL OUTPUT ---")
        print(tool_output)

        return {
            "result": result,
            "error": None,
            "routing_output": routing_output,
            "tool_output": tool_output
        }

    except ZeroDivisionError as error:
        return {
            "error": str(error),
            "routing_output": routing_output,
            "tool_output": f"Tool error: {error}"
        }

    except Exception as error:
        return {
            "error": str(error),
            "routing_output": routing_output,
            "tool_output": f"Tool error: {error}"
        }


def error_node(
    state: CalculatorState
) -> CalculatorState:
    """
    Preserve an existing error or create a routing error.
    """

    error_message = state.get(
        "error",
        "The requested operation is not supported."
    )

    routing_output = (
        "LangGraph selected the error node."
    )

    tool_output = (
        "No calculator tool was executed.\n"
        f"Error: {error_message}"
    )

    print("\n--- ROUTING OUTPUT ---")
    print(routing_output)

    print("\n--- TOOL OUTPUT ---")
    print(tool_output)

    return {
        "error": error_message,
        "routing_output": routing_output,
        "tool_output": tool_output
    }


def format_response_node(
    state: CalculatorState
) -> CalculatorState:
    """
    Format the final response shown in Gradio.
    """

    if state.get("error"):
        response = (
            f"Error: {state['error']}"
        )

        formatter_output = (
            "The formatter returned the error as "
            "the final response."
        )

        print("\n--- FORMATTER OUTPUT ---")
        print(formatter_output)

        print("\n--- FINAL RESPONSE ---")
        print(response)

        return {
            "response": response,
            "formatter_output": formatter_output
        }

    operation = state.get("operation")

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
        state.get("number_1")
    )

    number_2 = format_number(
        state.get("number_2")
    )

    result = format_number(
        state.get("result")
    )

    response = (
        f"Agent selected tool: {operation}\n\n"
        f"{number_1} {symbol} {number_2} = {result}"
    )

    formatter_output = (
        f"Operation: {operation}\n"
        f"Symbol: {symbol}\n"
        f"Formatted expression: "
        f"{number_1} {symbol} {number_2} = {result}"
    )

    print("\n--- FORMATTER OUTPUT ---")
    print(formatter_output)

    print("\n--- FINAL RESPONSE ---")
    print(response)

    return {
        "response": response,
        "formatter_output": formatter_output
    }


# ---------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------

def route_operation(state: CalculatorState) -> str:
    """
    Select the next graph node.
    """

    if state.get("error"):
        return "error"

    operation = state.get("operation")

    if operation == "add":
        return "add"

    if operation == "subtract":
        return "subtract"

    if operation == "multiply":
        return "multiply"

    if operation == "divide":
        return "divide"

    return "error"


# ---------------------------------------------------------
# Build the LangGraph workflow
# ---------------------------------------------------------

graph_builder = StateGraph(CalculatorState)

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


graph_builder.add_edge(
    "add",
    "format_response"
)

graph_builder.add_edge(
    "subtract",
    "format_response"
)

graph_builder.add_edge(
    "multiply",
    "format_response"
)

graph_builder.add_edge(
    "divide",
    "format_response"
)

graph_builder.add_edge(
    "error",
    "format_response"
)

graph_builder.add_edge(
    "format_response",
    END
)


calculator_agent = graph_builder.compile()


# ---------------------------------------------------------
# Gradio backend
# ---------------------------------------------------------

def run_agent_calculator(user_request):
    """
    Pass the user's request into the LangGraph agent.
    """

    if user_request is None:
        return (
            "Error: Please enter a calculation request.",
            "",
            "",
            "",
            "",
            ""
        )

    initial_state: CalculatorState = {
        "user_request": user_request,
        "operation": None,
        "number_1": None,
        "number_2": None,
        "result": None,
        "error": None,
        "response": None,
        "llm_output": None,
        "interpretation_output": None,
        "routing_output": None,
        "tool_output": None,
        "formatter_output": None
    }

    try:
        final_state = calculator_agent.invoke(
            initial_state
        )

        return (
            final_state.get(
                "response",
                "No response was generated."
            ),
            final_state.get(
                "llm_output",
                "No LLM output was generated."
            ),
            final_state.get(
                "interpretation_output",
                "No interpretation output was generated."
            ),
            final_state.get(
                "routing_output",
                "No routing output was generated."
            ),
            final_state.get(
                "tool_output",
                "No tool output was generated."
            ),
            final_state.get(
                "formatter_output",
                "No formatter output was generated."
            )
        )

    except Exception as error:
        return (
            f"Unexpected application error: {error}",
            "",
            "",
            "",
            "",
            ""
        )


def clear_calculator():
    """Clear the interface."""

    return (
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    )


# ---------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------

with gr.Blocks(
    title="LangGraph Agent Calculator"
) as app:

    gr.Markdown(
        """
        # LangGraph Agent Calculator

        Enter a calculation in natural language.

        The Hugging Face model interprets the request.
        LangGraph routes it to the appropriate calculator tool.

        Examples:

        - `Add 25 and 17`
        - `Subtract 15 from 40`
        - `Multiply 8 by 7`
        - `What is 100 divided by 4?`
        """
    )

    request_input = gr.Textbox(
        label="Calculation Request",
        lines=2,
        value=""
    )

    with gr.Row():

        calculate_button = gr.Button(
            value="Ask Calculator Agent",
            variant="primary"
        )

        clear_button = gr.Button(
            value="Clear"
        )

    with gr.Accordion(
        "Final Answer",
        open=True
    ):
        result_output = gr.Textbox(
            label="Agent Response",
            lines=4,
            interactive=False
        )

    with gr.Accordion(
        "Raw LLM Output",
        open=False
    ):
        llm_output = gr.Textbox(
            label="Hugging Face Model Output",
            lines=6,
            interactive=False
        )

    with gr.Accordion(
        "Interpreted Request",
        open=False
    ):
        interpretation_output = gr.Textbox(
            label="Operation and Numbers",
            lines=5,
            interactive=False
        )

    with gr.Accordion(
        "LangGraph Routing",
        open=False
    ):
        routing_output = gr.Textbox(
            label="Selected Graph Node",
            lines=3,
            interactive=False
        )

    with gr.Accordion(
        "Calculator Tool Output",
        open=False
    ):
        tool_output = gr.Textbox(
            label="Tool Execution",
            lines=5,
            interactive=False
        )

    with gr.Accordion(
        "Formatter Output",
        open=False
    ):
        formatter_output = gr.Textbox(
            label="Formatted Result",
            lines=5,
            interactive=False
        )

    calculator_outputs = [
        result_output,
        llm_output,
        interpretation_output,
        routing_output,
        tool_output,
        formatter_output
    ]

    calculate_button.click(
        fn=run_agent_calculator,
        inputs=request_input,
        outputs=calculator_outputs
    )

    request_input.submit(
        fn=run_agent_calculator,
        inputs=request_input,
        outputs=calculator_outputs
    )

    clear_button.click(
        fn=clear_calculator,
        inputs=[],
        outputs=[
            request_input,
            result_output,
            llm_output,
            interpretation_output,
            routing_output,
            tool_output,
            formatter_output
        ]
    )


# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

if __name__ == "__main__":
    app.launch(
        inbrowser=True,
        share=False
    )