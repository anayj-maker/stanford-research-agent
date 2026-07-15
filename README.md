# Stanford Research Agent

An AI agent that answers questions using a local library of Stanford AI Index Report PDFs (via RAG) or a live web search — deciding on its own, per question, which source is the right fit, and combining both when needed.

## How it works
The agent has two tools: one that searches a ChromaDB vector store built from the report's PDFs, and one that searches the live web via Tavily. A ReAct-style agent reasons about each question and picks the right tool(s) before answering.
