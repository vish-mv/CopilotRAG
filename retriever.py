import os
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)



def load_vector_store(directory, embedding_model):
    if os.path.exists(directory):
        return Chroma(persist_directory=directory, embedding_function=embedding_model).as_retriever(search_kwargs={"k": 5})
    else:
        raise FileNotFoundError(f"The vector store directory '{directory}' does not exist.")


def retrieve_relevant_info(question, vector_index, cohere_api_key, num_to_rerank=20, num_to_return=5):

    # Set the environment variable for the API key
    os.environ['COHERE_API_KEY'] = cohere_api_key

    # Retrieve more documents than needed for reranking
    relevant_docs = vector_index.get_relevant_documents(question, k=num_to_rerank)

    # Extract the content of relevant documents
    relevant_info = [doc.page_content for doc in relevant_docs]

    # Initialize CohereRerank without passing the API key
    reranker = CohereRerank()

    # Rerank the documents
    reranked = reranker.rerank(
        query=question,
        documents=relevant_info,
        top_n=num_to_return
    )

    # Retrieve the actual text of the top reranked documents using their indices
    reranked_info = [relevant_info[item['index']] for item in reranked]

    return reranked_info







