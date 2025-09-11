# components/qa_chain.py

from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

def build_qa_chain(deployment_name: str, temperature: float = 0):
    """
    Build a Question-Answering chain using the new RunnableSequence style.
    """

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Answer the question using only the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer concisely:"
        ),
    )

    # LLM
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=temperature,
    )

    # Runnable chain
    qa_chain = prompt | llm

    # Add verbose logging to the chain
    return qa_chain.with_config(verbose=True)
