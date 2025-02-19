from openai import OpenAI
import json
import time
import re
from tqdm import tqdm
import os

# Set up third-party API and keys
url = 'https://api.together.xyz/v1/'
llm_api_key = 'your_api_key'  

# Initialize the OpenAI client, specifying the third-party API's base_url and API key
client = OpenAI(
    base_url=url,
    api_key=llm_api_key
)

PROGRESS_FILE = "evaluation_progress.json"  # Save progress information
OUTPUT_FILE = "output_results.jsonl"  # Final results file

def read_jsonl(file_path):
    """
    Read the JSONL file line by line, parsing each line as a question (JSON object)
    """
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

def save_progress(last_processed_idx, results):
    """
    Save progress information
    """
    progress = {
        "last_processed_idx": last_processed_idx,
        "results": results
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)

def load_progress():
    """
    Load the previously saved progress information; if no progress is saved, return None
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def online_chat_completion(messages, model, max_tokens=1000):
    """
    Call the third-party service's model API and return the model's response
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,  
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def process_question(question, idx, runs=3):
     """
    Call the model multiple times for a single question (default runs=3), and return:
    - The answer, response time (latency), and tokens consumed for each run,
    - The average response time for the question,
    - Whether the answers are consistent (if all answers have the same letter),
    - The aggregated answer (take the answer returned in the first run),
    - Total token consumption (the sum of tokens consumed in all runs for this question)
    """
    base_prompt = (
        "Please answer the following questions and explain your reasoning. Please respond in the following format:\n"
        "Answer: <Option Letter>\n"
        "Explanation: <Detailed Explanation>\n\n"
        f"Question {idx + 1}: {question['question']}\n"
        "Options:\n"
    )
    for option, content in question['options'].items():
        base_prompt += f"  {option}: {content}\n"

    results = []      # Store the result of each call
    total_latency = 0 # Record the total response time of all calls
    tokens_list = []  # Record the number of tokens consumed by each call

    for i in range(runs):
        messages = [{"role": "user", "content": base_prompt}]
        start_time = time.time()
        content = online_chat_completion(
            messages=messages,
            model="model_name",  # Enter the model name
            max_tokens=1000
        )
        latency = time.time() - start_time
        total_latency += latency

        # Assuming the return format is similar to OpenAI, retrieve the complete answer text returned by the model
        answer_text = content.strip()

        # Use regular expressions to extract the answer letter (A to E, case-insensitive)
        letter_match = re.search(r"答案[:：]\s*([A-E])", answer_text, re.IGNORECASE)
        answer_letter = letter_match.group(1).upper() if letter_match else None

        # Simulate token consumption; the actual consumption needs to be adjusted based on the API response
        token_usage = len(content.split())  # Use word count as a substitute

        tokens_list.append(token_usage)

        results.append({
            "answer_letter": answer_letter,
            "latency": latency,
            "tokens": token_usage
        })

    avg_latency = total_latency / runs if runs > 0 else 0
    total_tokens = sum(tokens_list)

    # Determine whether all returned answers are consistent (excluding None cases)
    letters = [res["answer_letter"] for res in results if res["answer_letter"] is not None]
    consistency = (len(set(letters)) == 1) if letters else False

    aggregated_answer = letters[0] if letters else None

    return {
        "results": results,
        "aggregated_answer": aggregated_answer,
        "avg_latency": avg_latency,
        "total_tokens": total_tokens,
        "consistency": consistency
    }

def main():
    file_path = "test.jsonl"  # Enter the correct file path
    questions = read_jsonl(file_path)
    total_questions = len(questions)

    # Load the previous progress (if available)
    progress = load_progress()
    start_idx = progress["last_processed_idx"] if progress else 0
    evaluated_questions = progress["results"] if progress else []

    correct_count = 0         # Correct question count
    consistency_count = 0     # Consistent answer question count
    total_latency_all = 0     # Total response time
    total_tokens_all = 0      # Total token consumption (sum of all questions and all runs)

    runs_per_question = 3     # Number of model calls for each question

    # Use tqdm to display a progress bar
    for idx, question in enumerate(tqdm(questions[start_idx:], desc="Evaluating Questions", unit="question", initial=start_idx)):
        result = process_question(question, idx, runs=runs_per_question)
        # Retrieve the preset correct answer for the question (remove spaces and convert to uppercase)
        correct_answer = question.get("answer_idx", "").strip().upper()
        model_answer = result["aggregated_answer"]
        is_correct = (model_answer == correct_answer)
        if is_correct:
            correct_count += 1
        if result["consistency"]:
            consistency_count += 1

        total_latency_all += result["avg_latency"]
        total_tokens_all += result["total_tokens"]

        evaluated_questions.append({
            "question": question["question"],
            "ground_truth": correct_answer,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "consistency": result["consistency"],
            "avg_latency": result["avg_latency"],
            "total_tokens": result["total_tokens"],
            "detailed_runs": result["results"]
        })

        # Save progress after processing a certain number of questions
        if (idx + 1) % 10 == 0:
            save_progress(idx + 1, evaluated_questions)

    overall_accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    overall_consistency = (consistency_count / total_questions) * 100 if total_questions > 0 else 0
    average_latency = (total_latency_all / total_questions) if total_questions > 0 else 0

    # Save the final evaluation results to a JSONL file
    output_file = "other_model_evaluation_results.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        # Save overall statistics
        summary_obj = {
            "type": "summary",
            "total_questions": total_questions,
            "accuracy": overall_accuracy,
            "consistency": overall_consistency,
            "average_latency": average_latency,
            "total_tokens": total_tokens_all
        }
        f.write(json.dumps(summary_obj, ensure_ascii=False) + "\n")
        # Save the detailed evaluation results for each question
        for eval_data in evaluated_questions:
            eval_data["type"] = "question_evaluation"
            f.write(json.dumps(eval_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
