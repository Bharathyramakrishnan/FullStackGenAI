from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from utils.prompts import LOGISTICS_PROMPT

import os
from dotenv import load_dotenv
load_dotenv()
#from langchain.chat_models import init_chat_model
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

emb=HuggingFaceEmbeddings(
 model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db=Chroma(
 persist_directory="vectorstore",
 embedding_function=emb
)

retriever=db.as_retriever(
 search_kwargs={"k":5}
)

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

prompt = LOGISTICS_PROMPT

def ask(question,chat_history):

    docs = retriever.invoke(question)

    context="\n".join(
        [d.page_content for d in docs]
)

    history="\n".join(

        [

            f"{m['role']}:{m['content']}"

            for m in chat_history[-6:]

        ]

    )

    final = prompt.format(
        context=context,
        question=question,
        history=history
    )

    return llm.invoke(final).content