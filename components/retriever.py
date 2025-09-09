# components/retriever.py


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings


def build_retriever(file_path: str, embedding_deployment: str, chunk_size=200, chunk_overlap=20):
    """
    Build a retriever using FAISS and Azure OpenAI embeddings.
    Uses RecursiveCharacterTextSplitter for more natural chunking.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    print(f"✅ Total chunks created: {len(splits)}")

    # Initialized embedding model
    embeddings = AzureOpenAIEmbeddings(deployment=embedding_deployment)

    # Saved in FAISS Vector Store
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore.as_retriever()


