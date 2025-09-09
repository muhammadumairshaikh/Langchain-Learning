# components/summarizer.py

import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory


def build_summarizer(deployment_name: str, sentences: int = 3, memory=None):
    """
    Build a summarizer chain using AzureChatOpenAI.
    - If memory is None -> return simple chain (prompt | llm).
    - If memory is provided -> return an LLMChain with memory.
    """
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=0,
    )

    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are an expert summarizer. Summarize the following text "
            f"in exactly {sentences} sentence(s). Avoid redundancy and repetition.\n\n"
            "Text: {text}"
        ),
    )

    if memory is None:
        # Default pipe chain
        return prompt | llm
    else:
        # Memory-enabled chain
        return LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True,
        )


def create_buffer_memory(k: int = 3):
    """Helper to create ConversationBufferMemory storing last k interactions."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        k=k,
    )


def create_summary_memory(llm):
    """Helper to create ConversationSummaryMemory using given LLM."""
    return ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
    )
