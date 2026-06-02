from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
#os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
## Lnagsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
#os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

# Streamlit UI
st.title('Langchain Demo with Groq model')

with st.form("search_form_groq"):
    input_text = st.text_input("Search the topic you want",key="groq_input")
    submitted = st.form_submit_button("Search", key="groq_search")

## Groq model

llm = ChatGroq(model="qwen/qwen3-32b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if submitted and input_text:
    try:
        with st.spinner("Searching..."):
            result = chain.invoke({"question": input_text})
        st.write(result)
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    if submitted:
        st.warning("Please enter a question")