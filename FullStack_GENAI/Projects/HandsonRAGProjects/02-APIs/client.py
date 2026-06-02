import requests
import streamlit as st

# -------------------------
# GROQ CHAT (/chat)
# -------------------------
def get_groq_response(input_text):
    try:
        response = requests.post(
            "http://localhost:8000/chat/invoke",
            json={"input": {"topic": input_text}},
            timeout=10
        )
        response.raise_for_status()

        data = response.json()

        # safe parsing
        return (
            data.get("output")
            or data.get("output", {}).get("content")
            or "No output found"
        )

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------
# OLLAMA POEM (/poem)
# -------------------------
def get_ollama_response(input_text):
    try:
        response = requests.post(
            "http://localhost:8000/poem/invoke",
            json={"input": {"topic": input_text}},  # 🔥 IMPORTANT FIX
            timeout=60
        )
        response.raise_for_status()

        data = response.json()

        return (
            data.get("output")
            or data.get("output", {}).get("content")
            or "No output found"
        )

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------
# STREAMLIT UI
# -------------------------
st.title("LangChain Demo with Llama3 & Groq API")

input_text1 = st.text_input("Write an essay on")
input_text2 = st.text_input("Write a poem on")

if input_text1:
    st.subheader("Essay Result")
    st.write(get_groq_response(input_text1))

if input_text2:
    st.subheader("Poem Result")
    st.write(get_ollama_response(input_text2))