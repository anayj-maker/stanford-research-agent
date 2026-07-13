import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

all_chunks=[]

files = os.listdir("data")
for filenames in files:
    if filenames.endswith('.pdf'):
        filepath = os.path.join("data", filenames)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from {filenames}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"Split {len(pages)} pages into {len(chunks)} chunks")
#save and run python3 ingest.py

stanford_api_key = os.getenv("STANFORD_API_KEY")
stanford_api_base_url = os.getenv("API_BASE_URL")

embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002", openai_api_key=stanford_api_key, openai_api_base=stanford_api_base_url)

for chunk in all_chunks:
    chunk.page_content = chunk.page_content.encode('utf-8', 'ignore').decode('utf-8')

vectorstore = Chroma.from_documents(all_chunks, embeddings, persist_directory="./chroma_db")


