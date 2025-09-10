# components/qa_chain.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI


def build_qa_chain(deployment_name: str, temperature: float = 0):
    """
    Build a Question-Answering chain.
    The chain takes:
      - context (the text to answer from)
      - question (the user query)
    and returns a concise answer.
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

    # QA Chain
    qa_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True 
    )

    return qa_chain
