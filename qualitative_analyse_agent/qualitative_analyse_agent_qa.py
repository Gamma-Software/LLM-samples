from io import StringIO
import asyncio

import streamlit as st
import langchain
from langchain.llms import OpenAI

from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import AnalyzeDocumentChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import SequentialChain

from dotenv import load_dotenv
load_dotenv()

langchain.debug = True

st.sidebar.title("Qualitative Analyse Agent")
st.sidebar.write("This is a tool to help you analysis qualitative data")

# Fetching Qualitative Data
qualitative_docs = []
st.title("Upload Qualitative Data")
if files := st.file_uploader("Upload files", accept_multiple_files=True, type=['txt', 'md']):
    # Read the qualitative data
    # TODO force a format
    for file in files:
        # To read file as bytes:
        bytes_data = file.getvalue()

        # To convert to a string based IO:
        stringio = StringIO(file.getvalue().decode("utf-8"))

        # To read file as string:
        string_data = stringio.read()
        qualitative_docs.append("\n".join([string_data]))
    qualitative_docs_string = "\n\n".join([f"Qualitative Data {files[i].name}:\n{d}" for i, d in enumerate(qualitative_docs)])

    for i, docs in enumerate(qualitative_docs):
        with st.expander(files[i].name):
            st.markdown(qualitative_docs[i])

    # --- Summarizing qualitative data ---
    """if qualitative_docs:
        async def summarize_datas():
            async def async_generate(chain: LLMChain, doc):
                resp = await chain.arun(doc)
                print(resp.generations[0][0].text)

            llm = OpenAI(temperature=0)
            summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
            summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
            tasks = [async_generate(summarize_document_chain, d) for d in qualitative_docs]
            await asyncio.gather(*tasks)

        if st.button("Summarize data"):
            asyncio.run(summarize_datas())
            st.write()


# Summarize
if st.button("Summarize"):
    asyncio.run(summarize_datas())"""

template = """Summarize the transcript in 1000 words.
transcript: {transcript}
summary:"""
llm = OpenAI(temperature=0)
summary_prompt_template = PromptTemplate(input_variables=["transcript"], template=template)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt_template, output_key="summary")

if not qualitative_docs:
    st.write("Please upload some qualitative data in the Data menu")
    st.stop()

# --- Summarizing qualitative data based on a research question ---
# This step should use the RAG method to repond to the user based on the question

"""async def question_datas(question):
    async def async_generate(chain: LLMChain, doc, question):
        resp = await chain.arun(input_document=doc, question=question)
        print(resp.generations[0][0].text)

    llm = OpenAI(temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
    tasks = [async_generate(qa_document_chain, d, question) for d in qualitative_docs]
    await asyncio.gather(*tasks)

# Ask the chain a question
if question := st.text_input("Ask a question"):
    asyncio.run(summarize_datas(question))"""

template = """Summarize the transcript based on the research question in 1000 words.
transcript: {transcript}
question: {question}
summary_qa:"""
llm = OpenAI(temperature=0)
summary_qa_prompt_template = PromptTemplate(input_variables=["transcript", "question"], template=template)
summary_qa_chain = LLMChain(llm=llm, prompt=summary_qa_prompt_template, output_key="summary_qa")

# --- Generating initial codes ---
template = """Review the given transcript to identify relevant excerpts that address the research question.
Generate phrases (or codes) that best represent the excerpts identified. Each code must be between two to five words long.
Format your output in json format like such:
{
    "excepts":
    [
        "excepts1": "The excerpt that was identified",
        "excepts2": "The excerpt that was identified",
        ...
    ]
    "codes":
    [
        "code1": "The code that corresponds to the excerpt1",
        "code2": "The code that corresponds to the excerpt2",
        ...
    ]
}

transcript: {transcript}
question: {question}
output (in json):"""
llm = OpenAI(temperature=0.5)
extract_code_prompt_template = PromptTemplate(input_variables=["transcript", "question"], template=template)
extract_code_chain = LLMChain(llm=llm, prompt=extract_code_prompt_template, output_key="codes")

# --- (Double check) Verify the codes generated ---
# TODO

# --- Generating themes ---

template = """Based on the summary you have generated, develop 5 or 6 themes by categorizing the codes and addressing the research question.
Each themes must in between 5 to 6 words long.
Format your output in json format like such:
{
    "themes":[
        "theme1": "The first theme that was identified",
        "theme2": "The second theme that was identified",
        ...
    ]
    "codes":
    [
        "theme1": [
            "code1": "The first code that corresponds to the theme1",
            "code2": "The second code that corresponds to the theme1",
            ...
        ]
        "theme2": [
            "code1": "The first code that corresponds to the theme2",
            "code2": "The second code that corresponds to the theme2",
            ...
        ]
        ...
    ]
}

codes: {codes}
transcript: {summary_qa}
question: {question}
output (in json):"""
llm = OpenAI(temperature=0.5)
extract_themes_prompt_template = PromptTemplate(input_variables=["code", "summary_qa", "question"], template=template)
extract_themes_chain = LLMChain(llm=llm, prompt=extract_themes_prompt_template, output_key="themes")

# --- (Double check) Verify the themes generated ---
# TODO

# --- Execute LLM Chain ---
overall_chain = SequentialChain(
    chains=[summary_qa_chain, extract_code_chain, extract_themes_chain],
    input_variables=["transcript", "question"],
    output_variables=["codes", "themes"],
    verbose=True)

if question := st.text_input("Research question") and st.button("Analyse"):
    output = overall_chain.run({"transcript": qualitative_docs_string, "question": question})

# --- Generating a report in a table form ---
# TODO
output