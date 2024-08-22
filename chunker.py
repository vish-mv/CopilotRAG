# chunker.py

import os
import re
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Define the directory containing the markdown files
directory = "Docs"

# Define the heading patterns to split by (only # and ##)
heading_patterns = [r"^# ", r"^## "]

# Initialize Google embeddings
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
VECTORSTORE = os.environ.get("VECTORSTORE")


# Function to split content into chunks based on # and ##
def split_into_chunks(content):
    chunks = []
    current_chunk = []
    current_headers = []

    lines = content.splitlines()

    for line in lines:
        if any(re.match(pattern, line) for pattern in heading_patterns):
            if current_chunk:
                chunks.append({
                    'headers': current_headers,
                    'content': "\n".join(current_chunk)
                })
            current_headers = [line]
            current_chunk = []
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append({
            'headers': current_headers,
            'content': "\n".join(current_chunk)
        })

    return chunks


# Function to process all markdown files in the directory
def process_markdown_files(directory):
    all_chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                chunks = split_into_chunks(content)
                all_chunks.extend(chunks)
    return all_chunks


# Function to create and save vector store
def create_and_save_vector_store(chunks, embedding_model, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    texts = [chunk['content'] for chunk in chunks]
    vector_store = Chroma.from_texts(texts, embedding_model, persist_directory=directory)

    # Save the vector store to disk
    vector_store.persist()

    # Convert to retriever
    vector_index = vector_store.as_retriever(search_kwargs={"k": 5})
    return vector_index


def main():
    # Process the markdown files and get all chunks
    chunks = process_markdown_files(directory)

    # Create and save the vector store
    vector_index = create_and_save_vector_store(chunks, embedding_model, VECTORSTORE)

    print("Chunks have been processed and saved to the vector store.")


if __name__ == "__main__":
    main()
