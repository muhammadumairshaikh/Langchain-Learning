from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def build_summarizer(api_key: str, endpoint: str, deployment: str, api_version: str, summary_sentences: int = 3) -> LLMChain:
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
<<<<<<< HEAD
        template=(
            "You are an expert summarizer. "
            "Summarize the following text in exactly {summary_sentences} sentence(s). "
            "Be concise, factual, and avoid repetition.\n\n"
            "Text:\n{text}\n\n"
            "Summary:"
        ),
        partial_variables={"summary_sentences": str(summary_sentences)}
    )
=======
                template=f"""
                You are an expert summarizer.

                Task: Summarize the following text into exactly {sentences} sentence(s).
                - Do not write more or fewer sentences.
                - Each sentence should be concise, factual, and focused.
                - Avoid introductions, commentary, or meta text.

                Text:
                {{text}}
                """
            )
>>>>>>> tasks/task_3

    return LLMChain(llm=llm, prompt=prompt_template)