import csv
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import  answer_similarity, answer_relevancy,faithfulness,context_precision,context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
SAVEFILE = os.getenv("SAVEFILE")
print(SAVEFILE)
EVALUATE = os.getenv("EVALUATE")

# Initialize the LLM model
llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=1e-8)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# CSV file names
input_csv_file = SAVEFILE
output_csv_file = EVALUATE
    

# Function to evaluate a single question and return the evaluation score
def evaluate_single_question(question, answer, contexts, ground_truth):
    # Prepare the data in the required format
    data_samples = {
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],  # contexts should be a list
        'ground_truth': [ground_truth]
    }

    # Convert the data into a Hugging Face Dataset
    dataset = Dataset.from_dict(data_samples)

    # Evaluate the dataset using the specified metrics
    score = evaluate(dataset, metrics=[answer_similarity,answer_relevancy,faithfulness,context_precision,context_recall], llm=llm, embeddings=embeddings)

    # Convert the evaluation results to a DataFrame
    df = score.to_pandas()

    # Extract the scores

    answer_similarity_score = df['answer_similarity'].iloc[0]
    answer_relevancy_score= df['answer_relevancy'].iloc[0]
    faithfulness_score = df['faithfulness'].iloc[0]
    context_precision_score = df['context_precision'].iloc[0]
    context_recall_score = df['context_recall'].iloc[0]

    return answer_relevancy_score, answer_similarity_score, faithfulness_score, context_precision_score, context_recall_score


# Read the input CSV file and process each entry
with open(input_csv_file, mode='r') as input_file:
    reader = csv.reader(input_file)

    # Check if output CSV file exists to determine if header should be written
    file_exists = os.path.isfile(output_csv_file)

    # Open the output CSV file in append mode
    with open(output_csv_file, mode='a', newline='') as output_file:
        csv_writer = csv.writer(output_file)

        # Write the header if the file doesn't exist
        if not file_exists:
            csv_writer.writerow(['question', 'answer_similarity','answer_relevancy','faithfulness','context_precision','context_recall'])

        # Skip the header row in the input file
        next(reader)

        # Process each row
        for row in reader:
            # Extract the relevant data
            question, answer, contextOrg, ground_truth = row

            # Evaluate the current entry
            answer_similarity_score, answer_relevancy_score,faithfulness_score, context_precision_score, context_recall_score = evaluate_single_question(question,
                                                                                         answer, [
                                                                                         contextOrg],
                                                                                         ground_truth)

            # Append the results to the CSV file
            csv_writer.writerow([question, answer_similarity_score, answer_relevancy_score,faithfulness_score,context_precision_score,context_recall_score])

print("Evaluation scores have been saved to", output_csv_file)
