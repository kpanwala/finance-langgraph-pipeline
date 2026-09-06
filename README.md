# Indian Equity Intelligence Agent (LangGraph & Groq)

A stateful, multi-turn financial research assistant built with **LangGraph**, **LangChain**, and **ChatGroq**. The agent retrieves company fundamentals for Indian equities (NSE/BSE), scrapes recent financial news from trusted domain sources, looks up business backgrounds on Wikipedia, and manages conversation context through automated state summarization.

---

## Architecture Overview

```mermaid
flowchart TD
    START([Start]) --> tool_calling_llm[tool_calling_llm\n(ChatGroq / GPT-OSS)]
    
    tool_calling_llm -->|Tool Call Requested| tools[tools\n(ToolNode)]
    tool_calling_llm -->|Direct Reply & Messages > 2| summarize_conversation[summarize_conversation]
    tool_calling_llm -->|Direct Reply & Messages <= 2| END1([End])
    
    tools --> summarize_tool_result[summarize_tool_result\nGrounded Synthesis]
    
    summarize_tool_result -->|Messages > 2| summarize_conversation
    summarize_tool_result -->|Messages <= 2| END2([End])
    
    summarize_conversation --> END3([End])
```

The system employs a cyclical graph pattern:
1. **Dynamic Tool Calling**: Determines whether user intent requires financial data, news lookup, or background knowledge.
2. **Strict Synthesis Node (`summarize_tool_result`)**: Enforces non-hallucinatory summaries strictly adhering to tool responses.
3. **Context Trimming & Summarization (`summarize_conversation`)**: When message history exceeds a specified cutoff (e.g., > 2 messages), older messages are condensed into a persistent running summary and replaced with `RemoveMessage` to conserve LLM token windows while retaining user memory across multi-turn sessions.

---

## Features & Integrated Tools

| Tool | Source / Package | Purpose | Scope / Configuration |
| :--- | :--- | :--- | :--- |
| `get_company_financials` | `yfinance` | Fundamental statement metrics | Extracts revenue, latest net income, historical profit, operating cashflow, and total debt for NSE (`.NS`) and BSE (`.BO`) tickers. |
| `get_company_news` | `langchain_tavily` | Recent financial developments | Focused search query restricted to trusted publications: `moneycontrol.com`, `economictimes.indiatimes.com`, `livemint.com`. |
| `wikipedia` | `langchain_community` | Company overview & executive background | Brief overviews (top 1 result, max 500 chars) for founding history, founders, and corporate profile. |

---

## Project Structure

```text
.
├── demo.ipynb          # Interactive notebook with tools, state graph, and multi-turn runs
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # Documentation
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/)
- A [Tavily API Key](https://tavily.com/)

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/equity-intelligence-agent.git
cd equity-intelligence-agent
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Add your API credentials:

```env
GROQ_API_KEY="gsk_..."
TAVILY_API_KEY="tvly-..."
```

---

## Usage

You can open and execute `demo.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

### Minimal Invocation Example

```python
from langchain_core.messages import HumanMessage
from demo import graph  # or compile the graph from the notebook

# Specify a persistent thread for state checkpoints
config = {"configurable": {"thread_id": "portfolio_research_01"}}

# Turn 1: Ask an overview question
res1 = graph.invoke(
    {"messages": [HumanMessage(content="What does Infosys do in under 100 words?")]},
    config=config
)
print(res1["messages"][-1].content)

# Turn 2: Query financials (triggers automatic memory compaction)
res2 = graph.invoke(
    {"messages": [HumanMessage(content="What was their profit growth over the last 3 years?")]},
    config=config
)
print(res2["messages"][-1].content)

# Turn 3: Context recall from previous turns
res3 = graph.invoke(
    {"messages": [HumanMessage(content="Which company was I asking about?")]},
    config=config
)
print(res3["messages"][-1].content)
```

---

## Dependencies

- `langgraph`
- `langchain`
- `langchain-core`
- `langchain-groq`
- `langchain-tavily`
- `langchain-community`
- `yfinance`
- `wikipedia`
- `python-dotenv`

---

## Disclaimer

This software is for educational and informational purposes only. Retrieved financial figures, news extracts, and agent outputs do not constitute financial advice or investment recommendations.
