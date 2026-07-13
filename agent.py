from tools import local_research_tool, web_search_tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent

import os

load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-4o",
    api_key=os.getenv("STANFORD_API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
    temperature=0
)
tools = [local_research_tool, web_search_tool]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful research assistant. You have two tools: one for searching a local Stanford AI report, and one for searching the live web. Use the local tool for questions about the report's contents. Use the web tool for questions about current events or anything recent. Use both if the question needs it."
)

user_question = input("Ask your question: ")
response = agent.invoke({
    "messages": [{"role": "user", "content": user_question}]
})

print(response["messages"][-1].content)

    