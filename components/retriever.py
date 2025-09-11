from typing import Union, List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers import MultiQueryRetriever


def build_retriever(
    embedding_deployment: str = None,
    file_path: str = None,
    docs: Union[List[Document], None] = None,
    chunk_size: int = 200,
    chunk_overlap: int = 20,
    use_multi_query: bool = False,
    chat_deployment: str = None,
):
    """
    Build a retriever using FAISS and Azure OpenAI embeddings.

    Modes supported:
    1. Normal retriever from a file (file_path).
    2. Normal retriever from pre-loaded docs (docs).
    3. Multi-query retriever (set use_multi_query=True with chat_deployment).

    Args:
        embedding_deployment: Azure embedding deployment name.
        file_path: Path to a text file.
        docs: Already loaded documents (e.g., from PDF or Web loaders).
        chunk_size: Size of each chunk.
        chunk_overlap: Overlap between chunks.
        use_multi_query: Whether to enable multi-query retriever.
        chat_deployment: Chat model deployment (required for multi-query).
    """
    
    if file_path and embedding_deployment:
        if file_path.endswith(".txt") and not embedding_deployment.endswith(".txt"):
            pass
        elif not file_path.endswith(".txt") and embedding_deployment.endswith(".txt"):
            file_path, embedding_deployment = embedding_deployment, file_path

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
    base_retriever = vectorstore.as_retriever()

    # --- Normal retriever ---
    if not use_multi_query:
        return base_retriever

    # --- Multi-query retriever ---
    if chat_deployment is None:
        raise ValueError("chat_deployment must be provided when using multi-query retriever")

    llm = AzureChatOpenAI(deployment_name=chat_deployment, temperature=0)
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
