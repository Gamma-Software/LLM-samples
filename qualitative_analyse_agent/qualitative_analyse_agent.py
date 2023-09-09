from io import StringIO
import asyncio
import json

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

from lib.report import summary, codes, themes, parse_codes
from lib.chains import summary_chain, summary_qa_chain, generate_codes_chain, generate_themes_chain, overall_chain

from dotenv import load_dotenv
load_dotenv()

langchain.debug = True

st.title("Qualitative Analysis 📝 Agent")
st.caption("Using the power of LLMs")

st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Select pages", ["Raw data", "Qualitative Analysis"], index=1)

api_key = st.sidebar.container()
api_key.title("OpenAI API Key")
api_key.text_input("API Key", key="api_key", placeholder="openai api key")

data_upload = st.sidebar.container()
data_upload.title("Upload your data")
files = data_upload.file_uploader("Upload text", accept_multiple_files=True, type=['txt', 'md'])

options = st.sidebar.container()
options.title("Options ⚙️")
max_limit_summary_words = options.select_slider(
    'Select a limit of words for the summary',
    options=range(100, 2000, 100),
    value=(1000))
min_limit_codes, max_limit_codes = options.select_slider(
    'Select a range of codes to generate',
    options=range(2, 21, 1),
    value=(14, 20))
min_limit_theme_words, max_limit_theme_words = options.select_slider(
    'Each themes must between min/max words long',
    options=range(2, 10, 1),
    value=(5, 6))

description_container = st.empty()
if files:
    description_container = st.expander("Description")
else:
    description_container = st.container()

description_container.write("This is a tool to help you do your qualitative data analysis. This can for instance take your transcripts and generate codes and themes for you")
description_container.markdown("To perfom the qualitative analysis, you will need to upload your transcripts and ask a research question. The tool will then generate:\n1. summary of the transcripts \n2. Identify excerts and generate codes\n3. Generate themes based on the research question you asked.")
description_container.markdown("Learn more about [Qualitative Analysis](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiB2pWUhZyBAxV2WqQEHRW-A54QFnoECCUQAw&url=https%3A%2F%2Fwww.investopedia.com%2Fterms%2Fq%2Fqualitativeanalysis.asp&usg=AOvVaw09Xoebi_k9lmD1zCFIS2Bn&opi=89978449)")

st.sidebar.divider()
st.sidebar.markdown("This tool is powered by:\n- [Streamlit](https://streamlit.io)\n- [Langchain](https://langchain.com)")
st.sidebar.divider()
st.sidebar.markdown("Made with ❤️ by [Valentin Rudloff](https://www.linkedin.com/in/rudloffvalentin/)")


if not files:
    st.info("Please upload some qualitative data in the sidebar")
    st.stop()

# Fetching Qualitative Data
qualitative_docs = []
if files:
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
    qualitative_docs_string = "\n\n".join(
        [f"Qualitative Data {files[i].name}:\n{d}" for i, d in enumerate(qualitative_docs)])

    if qualitative_docs and menu == "Raw data":
        st.subheader("Show Raw data")
        tabs = st.tabs([file.name for file in files])
        for i, tab in enumerate(tabs):
            with tab:
                tab.markdown(qualitative_docs[i])

# TODO Add audio transcription to text

# --- Summarizing qualitative data ---
if menu == "Raw data":
    if st.button("Summarize transcripts", key="summarize"):
        with st.spinner("Summarizing..."):
            chain = summary_chain()
            summarize = chain({"max_limit_summary": max_limit_summary_words,
                               "transcript": qualitative_docs_string})
            with st.expander("Summary"):
                st.markdown(summarize["summary"])

# --- Summarizing qualitative data based on a research question ---
# This step should use the RAG method to repond to the user based on the question
if menu == "Qualitative Analysis":
    question = st.text_area("Research question", placeholder="How do students perceive the quality "
                            "and accessibility of food services on campus, and what factors "
                            "influence their dining choices and satisfaction?")

    # --- Summarizing transcripts ---
    # DONE

    # --- Generating initial codes ---
    # DONE

    # --- (Double check) Verify the codes generated ---
    # TODO

    # --- Generating themes ---
    # DONE

    # --- (Double check) Verify the themes generated ---
    # TODO

    # --- Get source of the excerpts ---
    # TODO

    # --- Execute LLM Chain ---
    output = None
    if not question:
        st.info("Please enter a question to continue analysis...")
    elif st.button("🚄 Perform Qualitative Analysis"):
        with st.spinner("Summarizing..."):
            output = overall_chain()({"max_limit_summary": max_limit_summary_words,
                                      "min_limit_codes": min_limit_codes,
                                      "max_limit_codes": max_limit_codes,
                                      "transcript": qualitative_docs_string,
                                      "question": question}, return_only_outputs=True)
            generated_themes = output["themes"]
            generated_codes = output["codes"]
            generated_summary = output["summary_qa"]

    # --- Generating a report in a table form ---
    if output:
        st.header("Analysis report 📈")

        st.subheader("Summary")
        st.caption(generated_summary)

        st.subheader("Table")
        table = parse_codes(generated_codes, generated_themes)
        st.table(table)
