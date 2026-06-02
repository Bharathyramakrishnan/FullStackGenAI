from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API Server"
)

# -------------------------
# GROQ MODEL
# -------------------------
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_model = ChatGroq(model="llama-3.1-70b-versatile")

# -------------------------
# OLLAMA MODEL
# -------------------------
llm = Ollama(model="llama3")

# -------------------------
# PROMPTS
# -------------------------
prompt_essay = ChatPromptTemplate.from_template(
    "Write an essay about {topic} in 100 words"
)

prompt_poem = ChatPromptTemplate.from_template(
    "Write a poem about {topic} in 100 words"
)

# -------------------------
# ROUTES
# -------------------------

# Direct chat
chat_prompt = ChatPromptTemplate.from_template("{input}")

chat_chain = chat_prompt | groq_model

add_routes(app, chat_chain, path="/chat")

# Essay (Groq)
add_routes(app, prompt_essay | groq_model, path="/essay")

# Poem (Ollama)
add_routes(app, prompt_poem | llm, path="/poem")

# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)