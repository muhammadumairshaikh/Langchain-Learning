# components/retriever.py

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

def build_retriever(file_path, api_key, endpoint, embedding_deployment, api_version, chunk_size=200, overlap=20, k=3):
   
   # Load Text File
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = splitter.split_documents(documents)
    print(f" Total chunks created: {len(docs)}")

    # Create embeddings
    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        deployment=embedding_deployment,   
        openai_api_version=api_version,
    )

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f" Total embedding vectors stored: {len(vectorstore.index_to_docstore_id)}")

    # Retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever, docs
