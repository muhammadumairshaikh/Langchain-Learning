# components/retriever.py

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever


def build_retriever(
    file_path: str,
    embedding_deployment: str,
    chunk_size: int = 200,
    chunk_overlap: int = 20,
    use_multi_query: bool = False,
    chat_deployment: str = None,
):
    """
    Build a retriever on a given text file.
    - Default: Normal retriever (FAISS + embeddings)
    - Multi-query mode: Uses an LLM to generate multiple alternate queries
    """

    # Load document
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)
    print(f"✅ Total chunks created: {len(splits)}")

    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(deployment=embedding_deployment)

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    base_retriever = vectorstore.as_retriever()

    # Normal retriever
    if not use_multi_query:
        return base_retriever

    # Multi-query retriever
    if chat_deployment is None:
        raise ValueError("chat_deployment must be provided when using multi-query retriever")

    llm = AzureChatOpenAI(deployment_name=chat_deployment, temperature=0)

    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
