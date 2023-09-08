from io import StringIO
import asyncio

import streamlit as st
import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import AnalyzeDocumentChain
from langchain.chains import SequentialChain

from lib.output_parsers import code_parser, theme_parser


# --- Summarizing qualitative data ---
def summary_chain():
    """--- Summarizing qualitative data ---"""
    summary_template = """Summarize the transcript in {max_limit_summary} words.
    transcript: {transcript}
    summary:"""
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    summary_prompt_template = PromptTemplate(input_variables=["max_limit_summary", "transcript"],
                                             template=summary_template)
    return LLMChain(llm=llm, prompt=summary_prompt_template, output_key="summary")

def summary_qa_chain():
    """--- Summarizing qualitative data with research question ---"""
    summary_qa_template = """Summarize the transcript based on the research question in {max_limit_summary} words.
    transcript: {transcript}
    question: {question}
    summary_qa:"""
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    summary_qa_prompt_template = PromptTemplate(input_variables=["max_limit_summary",
                                                                 "transcript",
                                                                 "question"],
                                                template=summary_qa_template)
    return LLMChain(llm=llm, prompt=summary_qa_prompt_template, output_key="summary_qa")


def generate_codes_chain():
    prompt_codes = """Review the given transcript to identify relevant excerpts that address the research question.
    Generate between {min_limit_codes} and {max_limit_codes} phrases (or codes) that best represent the excerpts identified. Each code must be between two to five words long.
    {format_instructions}

    <transcript>
    {transcript}
    <transcript>

    <question>
    {question}
    </question>

    codes:"""
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-16k")
    extract_code_prompt_template = PromptTemplate(
        input_variables=["min_limit_codes", "max_limit_codes", "transcript", "question"],
        template=prompt_codes,
        partial_variables={"format_instructions": code_parser.get_format_instructions()},)
    return LLMChain(llm=llm, prompt=extract_code_prompt_template, output_key="codes")


def generate_themes_chain():
    prompt_themes = """Based on the summary you generated, develop 5 or 6 themes by categorizing the codes and addressing the research question.
    Each themes must in between 5 to 6 words long. {format_instructions}

    <code>
    {codes}
    </code>

    <summary>
    {summary}
    </summary>

    <question>
    {question}
    </question>

    themes:"""
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-16k")
    extract_themes_prompt_template = PromptTemplate(
        input_variables=["codes", "summary", "question"],
        template=prompt_themes,
        partial_variables={"format_instructions": theme_parser.get_format_instructions()},)
    return LLMChain(llm=llm, prompt=extract_themes_prompt_template, output_key="themes")


def overall_chain():
    return SequentialChain(
        chains=[summary_qa_chain(), generate_codes_chain(), generate_themes_chain()],
        input_variables=["max_limit_summary", "min_limit_codes", "max_limit_codes",
                         "transcript", "question"],
        output_variables=["summary_qa", "codes", "themes"],
        verbose=True)
