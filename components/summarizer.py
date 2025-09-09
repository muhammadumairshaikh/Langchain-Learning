from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def build_summarizer(api_key: str, endpoint: str, deployment: str, api_version: str, sentences: int = 3) -> LLMChain:
    """
    Build a strict summarization chain using Azure OpenAI.
    Summarizes input text into the exact number of sentences specified,
    avoiding repetition or irrelevant details.
    """

    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        deployment_name=deployment,
        api_version=api_version,
    )

    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are an expert summarizer. "
            "Summarize the following text in exactly {summary_sentences} sentence(s). "
            "Be concise, factual, and avoid repetition.\n\n"
            "Text:\n{text}\n\n"
            "Summary:"
        ),
        partial_variables={"summary_sentences": str(sentences)}
    )

    return LLMChain(llm=llm, prompt=prompt_template)
