import os
from langchain.vectorstores import Chroma
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTORSTORE = os.getenv("VECTORSTORE")
EMBEDDINGS = os.getenv("EMBEDDINGS")

