import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
from tools import local_research_tool, web_search_tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

st.set_page_config(page_title="Stanford Research Assistant", page_icon="🔎")
st.title("Stanford Research Assistant")

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
    system_prompt="You are a helpful research assistant with two tools: Local_Research_Tool for the Stanford AI report, and Live_Web_Search for anything current. You do NOT know today's date or any real-time information from your own memory — it is outdated. For ANY question involving today's date, current events, sports schedules, or anything time-sensitive, you MUST use Live_Web_Search first, never answer from memory. Use Local_Research_Tool for questions about the report's contents. Use both if needed."
)   

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("Ask your question:")
for msg in st.session_state.conversation:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask your question:")
if user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    response = agent.invoke({"messages": st.session_state.conversation})
    reply = response["messages"][-1].content

    st.session_state.conversation.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)