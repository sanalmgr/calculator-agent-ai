# Agent Calculator: Procedural Programming vs Agentic AI

This repository accompanies my YouTube tutorial, where we build the **same calculator application** using two different approaches:

1. **Traditional Procedural Programming**
2. **Agentic AI using LangGraph and Tool Calling**

The objective is not to build a sophisticated calculator, but to understand how the **architecture changes** when moving from a deterministic program to an AI agent.

---

## 🎥 YouTube Tutorial

📺 **Watch the complete tutorial here:**

https://youtube.com/watch?v=abc

---

## Repository Structure

```
CALCULATOR-AGENT-AI
│
├── notebooks
│   ├── procedural_calculator.ipynb
│   └── agent_calculator.ipynb
│
├── scripts
│   ├── procedural_calculator.py
│   ├── agent_calculator.py
│   ├── agent_gradio_simple.py
│   ├── agent_graph.py
│   ├── requirements.txt
│   │
│   └── langsmith_api
│       ├── agent_graph.py
│       ├── important.env
│       └── langgraph.json
│
└── README.md
```

---

# What You'll Learn

By working through these examples, you'll understand:

- How a procedural application is structured
- How an AI agent differs from a procedural program
- How tool calling works
- How LangGraph orchestrates an agent
- Why AI agents still rely on ordinary Python functions
- When to choose procedural programming versus an AI agent

---

# Project Variants

This repository contains several implementations.

## 1. Procedural Calculator

Location:

```
scripts/procedural_calculator.py
```

Notebook:

```
notebooks/procedural_calculator.ipynb
```

Features

- Traditional Python programming
- Explicit control flow
- Gradio interface
- No LLM involved

---

## 2. Basic Agent Calculator

Location:

```
scripts/agent_calculator.py
```

A minimal agent implementation showing the core concepts before introducing a user interface.

---

## 3. Agent Calculator with Gradio (Used in the Video)

Location:

```
scripts/agent_gradio_simple.py
```

This is the implementation demonstrated in the YouTube tutorial.

Features

- Natural language input
- Tool calling
- LangGraph workflow
- Local open-source LLM
- Gradio interface

Example requests:

```
Add 15 and 27

Multiply 12 by 8

What is 250 divided by 5?

Subtract 90 from 145
```

---

## 4. LangSmith Version

Location:

```
scripts/langsmith_api/
```

This version integrates with LangSmith so you can inspect the complete execution of the agent, including:

- Graph execution
- Tool calls
- State transitions
- Traces
- Debugging information

This version is useful if you want to learn how production agent workflows are observed and debugged.

---

# Python Version

The examples use different Python versions because of dependency compatibility.

| Project | Python Version |
|----------|----------------|
| Procedural Calculator | 3.9 |
| Agent Calculator | 3.9 |
| Gradio Agent | 3.9 |
| LangSmith Version | 3.11 |

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<username>/calculator-agent-ai.git
```

Move into the project directory.

```bash
cd calculator-agent-ai
```

Install dependencies.

```bash
pip install -r scripts/requirements.txt
```

---

# Running the Examples

## Procedural Calculator

```bash
python scripts/procedural_calculator.py
```

---

## Agent Calculator

```bash
python scripts/agent_calculator.py
```

---

## Gradio Agent

```bash
python scripts/agent_gradio_simple.py
```

---

## LangSmith Version

Navigate to

```
scripts/langsmith_api
```

Configure your LangSmith environment variables in:

```
important.env
```

Then run:

```bash
python agent_graph.py
```

---

# Key Takeaway

One of the biggest misconceptions about AI agents is that they replace traditional programming.

They don't.

The business logic, tools, validation and application boundaries are still written by the developer. The difference is **where decisions are made**.

- In procedural programs, decisions are explicitly written in code.
- In agentic applications, a language model decides which developer-defined tool should be used based on the user's request.

Understanding this distinction makes AI agents much easier to reason about and design.

---

# Need a Deeper AI Agents Tutorial?

This repository intentionally focuses on the practical differences between procedural programming and agentic AI.

Topics such as:

- Agent architectures
- Reasoning
- Planning
- Memory
- Tool calling internals
- Workflows vs Agents
- Model Context Protocol (MCP)
- Multi-agent systems

are intentionally kept out of this tutorial to keep the learning focused.

If you'd like a dedicated deep-dive on these topics, let me know in the comments on the YouTube video.

---

# If This Repository Helped

If you found this project useful,

⭐ Star the repository

👍 Like the YouTube video

📺 Subscribe to the channel

It really helps support future educational content.

---

# License

This project is released under the MIT License.