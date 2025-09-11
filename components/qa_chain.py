# components/qa_chain.py

from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

def build_qa_chain(deployment_name: str, temperature: float = 0):
    """
    Build a Question-Answering chain that returns:
      - a concise answer
      - a short, non-sensitive rationale (bullet list; NOT chain-of-thought)
    The chain is returned with verbose enabled so LangChain runtime logs
    (requests/responses) are printed.
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Answer the question using only the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Output requirements (IMPORTANT):\n"
            "1) Provide a concise direct Answer (1-2 sentences) prefixed with `Answer:`.\n"
            "2) Then provide a brief, non-sensitive Rationale (max 3 bullet points) prefixed with `Rationale:`.\n"
            "   - The rationale should list the pieces of context you used and the high-level steps/evidence.\n"
            "   - Do NOT reveal internal chain-of-thought or verbatim private reasoning.\n\n"
            "Example Output:\n"
            "Answer: <concise answer>\n"
            "Rationale:\n"
            "- Bullet 1\n"
            "- Bullet 2\n"
        ),
    )

    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=temperature,
    )

    qa_chain = prompt | llm
    return qa_chain.with_config(verbose=True)
