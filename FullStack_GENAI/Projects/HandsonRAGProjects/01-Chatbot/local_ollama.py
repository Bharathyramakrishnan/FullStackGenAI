import langchain_ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

## Lnagsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title('Langchain Demo with OLLama Local')
input_text=st.text_input("Search the topic u want")
search_button = st.button("Search")

## OLLama LLAma-3 model

llm=OllamaLLM(model="llama3")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if search_button:
    if input_text:
        st.write(chain.invoke({"question":input_text}))
    else:
        st.warning("Please enter a question to search")