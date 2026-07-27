# -*- coding: utf-8 -*-
"""
Procedural Calculator using Gradio 4.44.1

This version uses normal Python functions and
explicit if/elif control flow.

It does not use LangGraph, agents, tools, or an LLM.
"""

import gradio as gr


def format_number(value):
    """
    Format numbers cleanly.

    For example:
    5.0 becomes 5
    5.25 remains 5.25
    """

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def calculate(number_1, operation, number_2):
    """
    Perform a calculation using procedural Python logic.

    Parameters
    ----------
    number_1 : float
        First number.

    operation : str
        Selected mathematical operation.

    number_2 : float
        Second number.

    Returns
    -------
    str
        Formatted result or an error message.
    """

    if number_1 is None or number_2 is None:
        return "Error: Please enter both numbers."

    if operation is None:
        return "Error: Please select an operation."

    try:
        if operation == "Add":
            result = number_1 + number_2
            symbol = "+"

        elif operation == "Subtract":
            result = number_1 - number_2
            symbol = "-"

        elif operation == "Multiply":
            result = number_1 * number_2
            symbol = "×"

        elif operation == "Divide":
            if number_2 == 0:
                return "Error: Division by zero is not allowed."

            result = number_1 / number_2
            symbol = "÷"

        else:
            return "Error: Invalid operation selected."

        number_1 = format_number(number_1)
        number_2 = format_number(number_2)
        result = format_number(result)

        return f"{number_1} {symbol} {number_2} = {result}"

    except TypeError:
        return "Error: Please enter valid numeric values."

    except Exception as error:
        return f"Unexpected error: {error}"


def clear_calculator():
    """
    Reset the calculator fields.
    """

    return None, "Add", None, ""


with gr.Blocks(title="Procedural Python Calculator") as app:

    gr.Markdown(
        """
        # Procedural Python Calculator

        Enter two numbers and select an operation.

        This calculator uses normal Python functions and
        explicit `if/elif` control flow.
        """
    )

    with gr.Row():

        number_1_input = gr.Number(
            label="First Number"
        )

        operation_input = gr.Dropdown(
            choices=[
                "Add",
                "Subtract",
                "Multiply",
                "Divide"
            ],
            value="Add",
            label="Operation"
        )

        number_2_input = gr.Number(
            label="Second Number"
        )

    with gr.Row():

        calculate_button = gr.Button(
            value="Calculate",
            variant="primary"
        )

        clear_button = gr.Button(
            value="Clear"
        )

    result_output = gr.Textbox(
        label="Result",
        interactive=False
    )

    calculate_button.click(
        fn=calculate,
        inputs=[
            number_1_input,
            operation_input,
            number_2_input
        ],
        outputs=result_output
    )

    clear_button.click(
        fn=clear_calculator,
        inputs=[],
        outputs=[
            number_1_input,
            operation_input,
            number_2_input,
            result_output
        ]
    )

    number_2_input.submit(
        fn=calculate,
        inputs=[
            number_1_input,
            operation_input,
            number_2_input
        ],
        outputs=result_output
    )


if __name__ == "__main__":
    app.launch(share=True)