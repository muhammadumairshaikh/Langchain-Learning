# components/retriever.py


# components/retriever.py

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings


def build_retriever(file_path: str, env: dict):
    """Builds a retriever from a text file using Azure embeddings and FAISS."""

    # Step 1: Load raw content
    loader = TextLoader(file_path, encoding="utf-8")
    raw_docs = loader.load()

    # Step 2: Break content into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    chunked_docs = text_splitter.split_documents(raw_docs)
    print(f"Created {len(chunked_docs)} chunks from input file.")

    # Step 3: Generate embeddings with Azure
    embed_model = AzureOpenAIEmbeddings(
        openai_api_key=env["api_key"],
        azure_endpoint=env["endpoint"],
        deployment=env["embedding_deployment"],
        openai_api_version=env["api_version"],
    )

    # Step 4: Build FAISS index
    vectorstore = FAISS.from_documents(chunked_docs, embed_model)
    print(f"Stored {len(vectorstore.index_to_docstore_id)} embeddings in FAISS.")

    # Step 5: Return retriever
    return vectorstore.as_retriever()

