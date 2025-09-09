# components/summarizer.py

import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


def build_summarizer(
    deployment_name: str,
    sentences: int = 3,
    memory=None,
    use_parser: bool = False,
):
    """
    Build a summarizer chain.
    - Case 1: With parser → JSON output { "summary": str, "length": int }
    - Case 2: With memory → conversation-aware summarizer
    - Case 3: Default → plain summarizer (no memory, no parser)
    """

    # Create base LLM
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=0,
        
    )

   
    # Structured Output Parser
    if use_parser:
        response_schemas = [
            ResponseSchema(
                name="summary",
                description="The summary of the provided text"
            ),
            ResponseSchema(
                name="length",
                description="The character length of the summary"
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are an expert summarizer. Summarize the following text "
                f"in exactly {sentences} sentence(s).\n\n"
                "Text: {text}\n\n"
                "{format_instructions}\n\n"
                "Important: Return ONLY valid JSON, no extra commentary, no markdown fences."
            ),
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )
        return prompt | llm | output_parser

    # Default or Memory Summarizer

    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are an expert summarizer. Summarize the following text "
            f"in exactly {sentences} sentence(s). Avoid redundancy and repetition.\n\n"
            "Text: {text}"
        ),
    )

    if memory is None:
        # Default summarizer 
        return prompt | llm
    else:
        # Memory-enabled summarizer 
        return prompt | llm  # memory can still be used externally if needed



# Helper Functions for Memory

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
