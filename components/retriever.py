# components/retriever.py

from typing import Union, List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document


def build_retriever(
    embedding_deployment: str,
    file_path: str = None,
    docs: Union[List[Document], None] = None,
    chunk_size: int = 200,
    chunk_overlap: int = 20,
):
    """
    Build a retriever using FAISS and Azure OpenAI embeddings.
    - If file_path is provided, loads from file.
    - If docs are provided, uses them directly.
    """
    if not file_path and not docs:
        raise ValueError("Either file_path or docs must be provided.")

    # --- Load documents ---
    if file_path:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

    # --- Split documents ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    print(f"✅ Total chunks created: {len(splits)}")

    # --- Embeddings ---
    embeddings = AzureOpenAIEmbeddings(deployment=embedding_deployment)

    # --- Vector Store ---
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore.as_retriever()
