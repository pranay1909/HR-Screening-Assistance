from pypdf import PdfReader
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import ChatCohere
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
load_dotenv()
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def get_pdf_text(pdf):
    pdf_page = PdfReader(pdf)
    text = ""
    for page in pdf_page.pages:
        text+=page.extract_text()
    return text

def create_docs(pdf,unique_id):
    docs=[]
    for filename in pdf:

        chunks = get_pdf_text(filename)

        docs.append(Document(
            page_content = chunks,
            metadata={"name": filename.name, "type": filename.type, "size": filename.size, "unique_id":unique_id}
        ))

    return docs

def create_embeddings():
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:22m")
    return embeddings

def push_pinecone(embeddings, docs):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("test")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    vector_store.add_documents(docs)

def pull_pinecone(embeddings):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("test")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store

def similar_docs(vector_store, query, unique_id, k=2):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    similar_docs = vector_store.similarity_search_with_score(query = query,k = int(k), filter = {"unique_id":unique_id})
    return similar_docs

def get_summary(current_doc):
    llm = ChatCohere(model="command-r-plus")
    chain = load_summarize_chain(llm, chain_type = "map_reduce")
    summary = chain.run([current_doc])
    return summary
