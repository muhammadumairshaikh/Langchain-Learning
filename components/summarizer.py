# components/summarizer.py

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# For Task2
def build_summarizer(api_key, endpoint, deployment, api_version, sentences=3, temperature=0.1):
    """
    Returns an LLMChain that summarizes text into a given number of sentences.
    """
    llm = AzureChatOpenAI(
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        deployment_name=deployment,
        openai_api_version=api_version,
        temperature=temperature
    )

    prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
You are an expert summarizer.

Task: Summarize the following text into exactly 3 sentences.
- Do not write more or fewer sentences.
- Each sentence should be concise, factual, and focused.
- Avoid introductions, commentary, or meta text.

Text:
{text}
"""
)

    return LLMChain(llm=llm, prompt=prompt_template)
