# -*- coding: utf-8 -*-
"""
Local Qwen2.5 calculator agent using:

- Qwen/Qwen2.5-3B-Instruct
- Hugging Face Transformers
- LangChain ChatHuggingFace
- LangChain bind_tools()
- LangGraph ToolNode and tools_condition
- Gradio

No OpenAI model or proprietary LLM API is used.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any

import gradio as gr
import torch
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
)
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def choose_dtype() -> torch.dtype:
    """
    Select a practical dtype for the available hardware.

    CUDA:
        Prefer bfloat16 when supported, otherwise float16.

    CPU:
        Use float32 for broad compatibility.
    """

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    return torch.float32


TORCH_DTYPE = choose_dtype()

print(f"Loading tokenizer: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print(f"Loading model with dtype: {TORCH_DTYPE}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)

model.eval()

print("Model loaded successfully.")


# ---------------------------------------------------------
# Hugging Face pipeline and LangChain chat model
# ---------------------------------------------------------

text_generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False,
    return_full_text=False,
)

huggingface_llm = HuggingFacePipeline(
    pipeline=text_generation_pipeline
)

llm = ChatHuggingFace(
    llm=huggingface_llm
)


# ---------------------------------------------------------
# Mathematical tools
# ---------------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.
    """

    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a.

    Args:
        a: The number from which b is subtracted.
        b: The number to subtract from a.
    """

    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: The first number.
        b: The second number.
    """

    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: The dividend.
        b: The divisor, which must not be zero.
    """

    if b == 0:
        raise ValueError(
            "Division by zero is not allowed."
        )

    return a / b


tools = [
    add,
    subtract,
    multiply,
    divide,
]


# Qwen should make one mathematical call at a time.
#
# Some LangChain chat integrations accept
# parallel_tool_calls=False, while others do not.
# ChatHuggingFace currently exposes bind_tools() but may
# reject provider-specific keyword arguments. Therefore,
# bind only the tools here and enforce one tool call in
# the assistant node below.
llm_with_tools = llm.bind_tools(
    tools
)


# ---------------------------------------------------------
# System instruction
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a calculator agent.

For every valid arithmetic request:

1. Select exactly one calculator tool.
2. Supply numeric values for arguments a and b.
3. Do not calculate the answer yourself before using a tool.
4. Use the tool result to answer the user clearly.

Operand-order rules:

- "Subtract 5 from 20" means subtract(a=20, b=5).
- "20 minus 5" means subtract(a=20, b=5).
- "100 divided by 4" means divide(a=100, b=4).
- "Divide 100 by 4" means divide(a=100, b=4).

Only use the available tools for arithmetic.
""".strip()


# ---------------------------------------------------------
# Local tool-call normalisation
# ---------------------------------------------------------

TOOL_NAMES = {
    calculator_tool.name
    for calculator_tool in tools
}


def parse_text_tool_call(
    content: Any
) -> dict[str, Any] | None:
    """
    Convert a plain-text function call such as

        add(a=14, b=9)

    into the structured tool-call dictionary expected by
    AIMessage and LangGraph ToolNode.

    ChatHuggingFace may bind the tool schemas correctly but
    still return a local pipeline generation as plain text.
    """

    if not isinstance(content, str):
        return None

    candidate = content.strip()

    # Accept an optional Markdown code fence.
    candidate = re.sub(
        r"^```(?:python)?\\s*|\\s*```$",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()

    try:
        expression = ast.parse(
            candidate,
            mode="eval",
        ).body
    except SyntaxError:
        return None

    if not isinstance(expression, ast.Call):
        return None

    if not isinstance(expression.func, ast.Name):
        return None

    tool_name = expression.func.id

    if tool_name not in TOOL_NAMES:
        return None

    # Calculator tools should use named arguments only.
    if expression.args:
        return None

    arguments: dict[str, float] = {}

    for keyword in expression.keywords:
        if keyword.arg not in {"a", "b"}:
            return None

        try:
            value = ast.literal_eval(
                keyword.value
            )
        except Exception:
            return None

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            return None

        arguments[keyword.arg] = float(value)

    if set(arguments) != {"a", "b"}:
        return None

    return {
        "name": tool_name,
        "args": arguments,
        "id": f"call_{uuid.uuid4().hex}",
        "type": "tool_call",
    }


def normalise_tool_call_response(
    response: BaseMessage
) -> BaseMessage:
    """
    Preserve native structured calls, or convert Qwen's
    textual function-call syntax into AIMessage.tool_calls.
    """

    if not isinstance(response, AIMessage):
        return response

    if response.tool_calls:
        return response

    parsed_tool_call = parse_text_tool_call(
        response.content
    )

    if parsed_tool_call is None:
        return response

    return AIMessage(
        content="",
        tool_calls=[
            parsed_tool_call
        ],
        response_metadata=response.response_metadata,
        id=response.id,
    )


# ---------------------------------------------------------
# LangGraph assistant node
# ---------------------------------------------------------

def assistant(
    state: MessagesState
) -> dict[str, list[BaseMessage]]:
    """
    Invoke Qwen to select a tool.

    After ToolNode executes the tool, the latest message is a
    ToolMessage. ChatHuggingFace local pipelines may reject
    conversations whose final message is not a HumanMessage,
    so the calculator formats the tool result directly instead
    of making an unnecessary second model call.
    """

    messages = state["messages"]

    if messages and isinstance(
        messages[-1],
        ToolMessage,
    ):
        tool_message = messages[-1]

        final_response = AIMessage(
            content=(
                f"Tool {tool_message.name} returned "
                f"{tool_message.content}"
            )
        )

        print("\n--- FINAL TOOL RESULT ---")
        print(final_response.content)

        return {
            "messages": [final_response]
        }

    raw_response = llm_with_tools.invoke(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            *messages,
        ]
    )

    response = normalise_tool_call_response(
        raw_response
    )

    # Mathematical work in this application should be
    # sequential. Reject multiple calls explicitly.
    if (
        isinstance(response, AIMessage)
        and len(response.tool_calls) > 1
    ):
        response = AIMessage(
            content=(
                "I can execute only one mathematical "
                "operation per request. Please provide "
                "one calculation at a time."
            )
        )

    print("\n--- RAW ASSISTANT MESSAGE ---")
    print(raw_response)

    print("\n--- NORMALISED ASSISTANT MESSAGE ---")
    print(response)

    if isinstance(response, AIMessage):
        print("\n--- TOOL CALLS ---")
        print(
            json.dumps(
                response.tool_calls,
                indent=2,
                default=str,
            )
        )

    return {
        "messages": [response]
    }


# ---------------------------------------------------------
# Build the ReAct graph
# ---------------------------------------------------------

builder = StateGraph(
    MessagesState
)

builder.add_node(
    "assistant",
    assistant,
)

builder.add_node(
    "tools",
    ToolNode(tools),
)

builder.add_edge(
    START,
    "assistant",
)

builder.add_conditional_edges(
    "assistant",
    tools_condition,
)

builder.add_edge(
    "tools",
    "assistant",
)

react_graph = builder.compile()


# ---------------------------------------------------------
# Gradio helpers
# ---------------------------------------------------------

def message_content_to_text(
    content: Any
) -> str:
    """
    Convert LangChain message content to readable text.
    """

    if isinstance(content, str):
        return content

    return json.dumps(
        content,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def format_trace(
    messages: list[BaseMessage]
) -> str:
    """
    Format the complete LangGraph execution trace.
    """

    sections: list[str] = []

    for index, message in enumerate(
        messages,
        start=1,
    ):
        if isinstance(message, HumanMessage):
            sections.append(
                f"Step {index} — User\n"
                f"{message_content_to_text(message.content)}"
            )

        elif isinstance(message, AIMessage):
            assistant_text = (
                message_content_to_text(message.content)
                if message.content
                else "(no natural-language content)"
            )

            tool_calls = (
                json.dumps(
                    message.tool_calls,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
                if message.tool_calls
                else "[]"
            )

            sections.append(
                f"Step {index} — Assistant\n"
                f"Content:\n{assistant_text}\n\n"
                f"Tool calls:\n{tool_calls}"
            )

        elif isinstance(message, ToolMessage):
            sections.append(
                f"Step {index} — Tool result\n"
                f"Tool: {message.name or 'unknown'}\n"
                f"Result: "
                f"{message_content_to_text(message.content)}"
            )

        else:
            sections.append(
                f"Step {index} — "
                f"{type(message).__name__}\n"
                f"{message_content_to_text(message.content)}"
            )

    return "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(
        sections
    )


def find_final_answer(
    messages: list[BaseMessage]
) -> str:
    """
    Return a readable final calculator answer.
    """

    last_tool_message = None
    selected_call = None

    for message in reversed(messages):
        if (
            last_tool_message is None
            and isinstance(message, ToolMessage)
        ):
            last_tool_message = message

        if (
            selected_call is None
            and isinstance(message, AIMessage)
            and message.tool_calls
        ):
            selected_call = message.tool_calls[0]

        if (
            last_tool_message is not None
            and selected_call is not None
        ):
            break

    if (
        last_tool_message is not None
        and selected_call is not None
    ):
        name = selected_call["name"]
        args = selected_call["args"]

        symbol_map = {
            "add": "+",
            "subtract": "-",
            "multiply": "×",
            "divide": "÷",
        }

        symbol = symbol_map.get(
            name,
            name,
        )

        a = args.get("a")
        b = args.get("b")
        result = last_tool_message.content

        return (
            f"Agent selected tool: {name}\n\n"
            f"{a} {symbol} {b} = {result}"
        )

    for message in reversed(messages):
        if (
            isinstance(message, AIMessage)
            and message.content
            and not message.tool_calls
        ):
            return message_content_to_text(
                message.content
            )

    return (
        "The graph finished without a final "
        "answer."
    )


def run_calculator(
    user_request: str
) -> tuple[str, str]:
    """
    Run one user request through the LangGraph agent.
    """

    user_request = user_request.strip()

    if not user_request:
        return (
            "Please enter a mathematical request.",
            "No graph execution occurred.",
        )

    try:
        result = react_graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=user_request
                    )
                ]
            },
            config={
                "recursion_limit": 8
            },
        )

        messages = result["messages"]

        final_answer = find_final_answer(
            messages
        )

        execution_trace = format_trace(
            messages
        )

        return (
            final_answer,
            execution_trace,
        )

    except Exception as error:
        error_text = (
            f"{type(error).__name__}: {error}"
        )

        return (
            f"Error: {error_text}",
            error_text,
        )


def clear_interface() -> tuple[str, str, str]:
    """
    Clear all Gradio components.
    """

    return "", "", ""


# ---------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------

with gr.Blocks(
    title="Qwen2.5 Tool-Calling Calculator"
) as demo:

    gr.Markdown(
        """
# Qwen2.5 LangGraph Calculator

This application runs `Qwen/Qwen2.5-3B-Instruct`
locally and uses:

- `ChatHuggingFace.bind_tools()`
- `ToolNode`
- `tools_condition`
- a LangGraph ReAct loop

Example requests:

- `Add 14 and 9`
- `Subtract 5 from 20`
- `Multiply 8 by 7`
- `Divide 100 by 4`
"""
    )

    user_input = gr.Textbox(
        label="Calculation request",
        placeholder="Example: Multiply 8 by 7",
        lines=2,
    )

    with gr.Row():
        calculate_button = gr.Button(
            "Calculate",
            variant="primary",
        )

        clear_button = gr.Button(
            "Clear"
        )

    final_output = gr.Textbox(
        label="Final answer",
        lines=4,
        interactive=False,
    )

    execution_trace = gr.Textbox(
        label="LangGraph execution trace",
        lines=20,
        interactive=False,
    )

    calculate_button.click(
        fn=run_calculator,
        inputs=user_input,
        outputs=[
            final_output,
            execution_trace,
        ],
    )

    user_input.submit(
        fn=run_calculator,
        inputs=user_input,
        outputs=[
            final_output,
            execution_trace,
        ],
    )

    clear_button.click(
        fn=clear_interface,
        inputs=[],
        outputs=[
            user_input,
            final_output,
            execution_trace,
        ],
    )


# ---------------------------------------------------------
# Launch
# ---------------------------------------------------------

if __name__ == "__main__":
    demo.launch()