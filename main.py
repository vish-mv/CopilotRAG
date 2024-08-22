import os
import shutil
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunker import main
from retriever import retrieve_relevant_info, load_vector_store
import csv
from openai import OpenAI
from groq import Groq

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)
VECTORSTORE = os.getenv("VECTORSTORE")
EMBEDDINGS = os.getenv("EMBEDDINGS")
CHUNK_DATA = os.getenv("CHUNK_DATA")
cohere_api_key = os.getenv("COHERE_API_KEY")
model = os.getenv("MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
SAVEFILE = os.getenv("SAVEFILE")
print(model)


embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)


if CHUNK_DATA == 'True':
    # Delete the directory specified in VECTORSTORE
    if os.path.exists(VECTORSTORE) and os.path.isdir(VECTORSTORE):
        shutil.rmtree(VECTORSTORE)
        print(f"Deleted directory: {VECTORSTORE}")
    else:
        print(f"Directory not found: {VECTORSTORE}")

    # Run the chunker main process
    main()


try:
    VECTORSTORE = os.getenv("VECTORSTORE")
    vector_index = load_vector_store(VECTORSTORE, embedding_model)
    print("Vector store loaded successfully.")
except FileNotFoundError as e:
    print(e)
    vector_index = None
if vector_index is None:
    print("Vector index is not loaded. Please load the vector store first.")
    exit()

csv_file = 'dataset.csv'

prompt = """You are a docbot that help users to understand more about Choreo Product usage by answering
    their questions. Information from docs are given to you help you answer the questions. If the information from docs
    are not relevant to your answer do not use it to construct your answer. You do not have to use all the information
    in the docs to construct your answer. Do not make up your own answer.If you don't have enough information to answer,
    refuse politely to answer the question. Do not hallucinate! Some docs might be from the same file. You can verify
    this through the filename or the doc_link in the metadata. If the docs are from the same file then you can use the
    step numbers to get an idea of the workflow.
    Choreo has Choreo console UI for a user to create projects and create components by connecting their github
    repository and build them and deploy them. By component term in choreo we mean services, webapps, webhooks and other
    types. In the Choreo console UI we can test, observe, view insights and perform devops actions for the deployed
    components. We can also create connections between different components to consume a service. The other UI is Choreo
    Dev portal. In this Dev portal UI we can view the components published from the Choreo Console UI. In the Dev portal
    UI user can perform actions like subscribing to components and invoking them for other applications and they can
    also perform actions like api rate-limiting stuffs. So since we have these two UIs the docs you have are chunks from
    performing actions in both the UIs. SO go through the doc VERY CAREFULLY to identify which UI the steps belong to
    and provide a comprehensive answer. You are most likely to mix the steps in both UIs. So be very careful not to do
    it. Construct you answer ACCURATELY! ALWAYS construct your answer with generic names for the components or services
    without using specific names like 'Reading List Service'(use the term 'your service' in this case).
    The information given contains markdown images, bullet-points and tables etc. You can make use of them by adding
    them to the response in markdown format. Make sure answers are structured enough to follow through and descriptive.
    In your answer always give the links of the most relevant doc from which you got the answer. You can use the
    doc_link metadata in the docs you are provided. Do not give fake links of your own. Do not always ask the user to
    refer the docs. Give a comprehensive answer to how to do it or what it is before you direct the user to the docs
    with the links.
    The answer you give is very critical. Choreo may not support all type of applications and user might ask those. So
    Strictly if you do not have enough information from the docs, refuse to answer politely. Don't include steps to sign
    in to choreo console. User is already in Choreo console while asking you this question so you don't have to direct
    him to choreo console again. You can use the Question's context to understand more about the User's question.
"""

with open(csv_file, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    # Loop through each row in the CSV
    for row in reader:
        question = row["question"]
        ground_truth = row["ground_truth"]

        # Retrieve relevant context using the question
        context = retrieve_relevant_info(question, vector_index, cohere_api_key)
        relevant_context = str(context)

        # You can now use `question`, `ground_truth`, and `relevant_context` variables
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Relevant Context: {relevant_context}\n")

        if model == 'gpt-4o' or model =='gpt-4o-mini':
            client = OpenAI(
                api_key=OPENAI_API_KEY
            )
        elif model == 'llama-3.1-70b-versatile':
            client = Groq(
                api_key=GROQ_API_KEY,
            )
        else:
            print("Invalid Model")
            exit()
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": relevant_context
                }
            ],
            model=model
        )
        result = chat_completion.choices[0].message.content
        save_file = SAVEFILE

        # Check if the file exists
        file_exists = os.path.isfile(save_file)

        # Open the file in append mode
        with open(save_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write header only if the file does not exist
            if not file_exists:
                writer.writerow(['question', 'answer', 'contextOrg', 'ground_truth'])

            # Write the data
            writer.writerow([question, result, relevant_context, ground_truth])

        print("Data has been appended to", save_file)

with open("evaluate.py") as f:
    code = f.read()
    exec(code)










