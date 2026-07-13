from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import Tool
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
import os

load_dotenv()
embeddings = OpenAIEmbeddings(
    model = "text-embedding-ada-002",
    openai_api_key=os.getenv("STANFORD_API_KEY"),
    openai_api_base=os.getenv("API_BASE_URL")
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def search_local_library(query):
    results = vectorstore.similarity_search(query, k=3)
    combined_text = "\n\n".join([result.page_content for result in results])
    return combined_text

local_research_tool = Tool(
    name="Local_Research_Tool",
    func=search_local_library,
    description="Use this to answer questions about the Stanford AI Index report. Input should be a search query about AI research, investment, policy, or trends covered in the report."
)

tavily_search = TavilySearch(max_results=3)

web_search_tool = Tool(
    name="Live_Web_Search",
    func=tavily_search.invoke,
    description="Use this to answer questions about current events, recent news, or anything that may have happened after the report's publication. Input should be a search query."
)
if __name__ == "__main__":
    print(local_research_tool.invoke("What does the report say about AI investment?"))
