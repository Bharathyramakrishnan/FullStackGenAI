
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY') 

## Data Ingestion
loader=TextLoader("speech.txt")
text_documets=loader.load()
#text_documets

# web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4

## load, hunk and index the content of the html page
loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header") 
                     )),)
text_documets=loader.load()

## PDF reader-----> Load
loader=PyPDFLoader('Outcome.pdf')
docs=loader.load()

## split -------> Transform
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documnets=text_splitter.split_documents(docs)
#documnets[:5]

## Vector Embedding And Vector Strore ------->Embed
#embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=Chroma.from_documents(documnets[:40], HuggingFaceEmbeddings())

## Vector DataBase
query="show me the table of contents"
result=db.similarity_search(query)
print("Chroma:",result)

## FAISS Vector DataBase
from langchain_community.vectorstores import FAISS
db1=FAISS.from_documents(documnets[:40],HuggingFaceEmbeddings())

# FAISS Query
query="show me the table of contents"
result=db1.similarity_search(query)
print("FAISS:",result)