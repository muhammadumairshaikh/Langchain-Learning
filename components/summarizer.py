# components/summarizer.py

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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
        template=f"Summarize the following text into exactly {sentences} sentence(s):\n\n{{text}}"
    )

    return LLMChain(llm=llm, prompt=prompt_template)
