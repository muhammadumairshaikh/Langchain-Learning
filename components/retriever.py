# components/retriever.py


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings


def build_retriever(api_key: str, endpoint: str, deployment: str, api_version: str, file_path: str, chunk_size: int = 200, chunk_overlap: int = 20):
    """
    Build a retriever using Azure OpenAI embeddings and FAISS in-memory vector store.
    Splits the input text file into chunks and indexes them for retrieval.
    """
    # Load file with encoding specified
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # Create embeddings
    embeddings = AzureOpenAIEmbeddings(
        api_key=api_key,
        azure_endpoint=endpoint,
        deployment=deployment,
        api_version=api_version,
    )

    # Build FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Create retriever
    retriever = vector_store.as_retriever()
    return retriever

