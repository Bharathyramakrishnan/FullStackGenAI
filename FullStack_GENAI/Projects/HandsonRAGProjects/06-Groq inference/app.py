import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv

load_dotenv()

# load the Groq API key
groq_api_key=os.getenv('GROQ_API_KEY') 

if "vector" not in st.session_state:
    st.session_state.embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    loader=WebBaseLoader("https://docs.smith.langchain.com/")
    docs = loader.load()
    
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    final_documents=splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectors=FAISS.from_documents(final_documents, embeddings)
    st.session_state.vectors = vectors

st.title("Chat-Groq Agent")
llm=ChatGroq(groq_api_key=groq_api_key, model='llama-3.3-70b-versatile')

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate respobse based o the question
<context>
{context}
</context>
Question:{input} 
"""
)

document_chain= create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain= create_retrieval_chain(retriever, document_chain)

user_prompt = st.text_input(
    "Ask a question",
    key="question_input"
)


if user_prompt:
    start=time.time()
    response=retrieval_chain.invoke({"input":user_prompt})
    print("Response time :", time.time()-start)
    st.write(response['answer'])
    
    st.caption(
        f"Response Time: {time.time()-start:.2f}s"
    )

    #with a streamlit expander
    with st.expander("Documnet Similarity Search"):
        #find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(
                f"Chunk {i+1}"
            )

            st.write(doc.page_content)
            st.divider()
            st.write("---------------------------------------------")