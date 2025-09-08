# components/summarizer.py

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from components.config_loader import load_config


# 🔹 Task 2: Basic Summarizer (no memory)
def build_summarizer(api_key, endpoint, deployment, api_version, sentences=3, temperature=0):
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

    return LLMChain(llm=llm, prompt=prompt_template)


# 🔹 Task 6: Extended Summarizer (with optional memory)
def build_summarizer_with_memory(sentences: int = 3, memory_type: str | None = None):
    """
    Returns a summarizer chain that can optionally use memory.

    Args:
        sentences (int): Number of sentences for the summary.
        memory_type (str | None): Memory option:
            - "buffer": ConversationBufferMemory
            - "summary": ConversationSummaryMemory
            - None: No memory

    Returns:
        LLMChain: Configured summarizer chain.
    """
    creds = load_config()

    llm = AzureChatOpenAI(
        openai_api_key=creds["api_key"],
        azure_endpoint=creds["endpoint"],
        deployment_name=creds["deployment"],
        openai_api_version=creds["api_version"],
        temperature=0,
    )

    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=f"""
        You are an expert summarizer.

        Task: Summarize the following text into exactly {sentences} sentence(s).
        - Do not write more or fewer sentences.
        - Each sentence should be concise, factual, and focused.
        - Avoid introductions, commentary, or meta text.
        - If prior conversation exists, incorporate it into the summary.

        Text:
        {{text}}
        """
    )

    memory = None
    if memory_type == "buffer":
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="text",
            k=3,
            return_messages=True
        )
    elif memory_type == "summary":
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            input_key="text",
            return_messages=True
        )

    if memory:
        return LLMChain(llm=llm, prompt=prompt_template, memory=memory, verbose=True)

    return LLMChain(llm=llm, prompt=prompt_template, verbose=True)
