import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate


def build_summarizer(deployment_name: str, sentences: int = 3):
    """
    Build a summarizer chain using AzureChatOpenAI.
    """
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=0,
        # api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are an expert summarizer. Summarize the following text "
            f"in exactly {sentences} sentence(s). Avoid redundancy and repetition.\n\n"
            "Text: {text}"
        ),
    )

    return prompt | llm

