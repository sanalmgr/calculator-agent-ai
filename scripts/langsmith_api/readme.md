## Running the LangSmith Version

The LangSmith implementation is located in:

```
scripts/langsmith_api/
```

This version allows you to visualize the complete execution of the agent inside the LangSmith dashboard, including the graph execution, tool calls, state transitions, and traces.

### Step 1: Create a LangSmith Account

Create a free account at:

https://smith.langchain.com/

---

### Step 2: Generate an API Key

From your LangSmith account:

- Go to **Settings**
- Select **API Keys**
- Create a new API key
- Copy the generated key

---

### Step 3: Update the Environment File

Open the file:

```
scripts/langsmith_api/important.env
```

Replace the placeholder values with your own credentials.

For example:

```env
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=calculator-agent
```

Then, rename this file to:

```
.env
```

---

### Step 4: Install Dependencies

This project uses **Python 3.11**.

Install the required packages:

```bash
pip install -r ../requirements.txt
```

or, from the repository root:

```bash
pip install -r scripts/requirements.txt
```

---

### Step 5: Start the LangGraph API

From inside the `langsmith_api` directory, run:

```bash
langgraph dev
```

The local LangGraph development server will start and expose your graph.

---

### Step 6: Open LangGraph Studio

Once the server is running, open:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

You can now:

- interact with the agent
- inspect every node execution
- view tool calls
- examine state updates
- debug the workflow step by step

This provides an excellent way to understand how LangGraph executes an agent behind the scenes.