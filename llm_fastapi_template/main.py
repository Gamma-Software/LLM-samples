from fastapi import FastAPI
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.cache import InMemoryCache
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain import hub
import chromadb
from pydantic import BaseModel
import dotenv
import uuid

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

dotenv.load_dotenv()

langchain.llm_cache = InMemoryCache()

app = FastAPI()

client = chromadb.Client()
collection = client.create_collection("sample_collection", embedding_function=OpenAIEmbeddings())

# RAG prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


class LLMData(BaseModel):
    prompt: str


class DocData(BaseModel):
    url: str


@app.post("/api/add_doc")
async def add_doc_url(doc_data: DocData):
    global collection
    # Load docs
    loader = WebBaseLoader(doc_data.url)
    data = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    collection.add(
        documents=[doc.text for doc in all_splits],
        metadatas=[doc.metadatas for doc in all_splits],
        ids=[uuid.uuid4() for doc in all_splits],
    )
    collection.update()


@app.get("/api/llm_completion")
async def tell_me_a_joke(data: LLMData):
    # RetrievalQA
    retriever = collection.as_retriever(search_type="mmr")

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    # The first time, it is not yet in cache, so it should take longer
    result = await qa_chain.arun({"query": data.prompt})
    return {"question": data.prompt, "completion": result["result"]}
