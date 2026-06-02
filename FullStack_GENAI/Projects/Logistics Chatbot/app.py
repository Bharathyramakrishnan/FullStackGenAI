import streamlit as st

from rag import ask


st.title(
    "Logistics RAG Chatbot"
)

if "messages" not in st.session_state:

    st.session_state.messages=[]


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


question=st.chat_input(
    "Ask logistics questions"
)


if question:

    st.session_state.messages.append(

        {

            "role":"user",

            "content":question

        }

    )

    with st.chat_message("user"):

        st.write(question)


    response=ask(

        question,

        st.session_state.messages

    )


    st.session_state.messages.append(

        {

            "role":"assistant",

            "content":response

        }

    )

    with st.chat_message("assistant"):

        st.write(response)