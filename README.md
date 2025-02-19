# LLMs_Medical_Evaluation
Evaluation of Large Language Models in Female Malignancies Specialty Q&amp;A
This repository provides a set of automated scripts to evaluate various large models for answering professional medical questions.

Files:
test.jsonl: Contains 205 multiple-choice questions and their corresponding correct answers on female malignancies.
OpenAI Model Evaluation.py: Script for evaluating OpenAI models (e.g., GPT-3, GPT-4) on the provided test set.
Other Model Evaluation.py: Script for evaluating other large language models through third-party APIs.
Usage:
Prerequisites:

Ensure you have Python installed (preferably Python 3.7+).
Install necessary Python libraries:
bash
复制
pip install openai tqdm
Configuration:

In both OpenAI Model Evaluation.py and Other Model Evaluation.py, update the following variables:
api_key: Set this to your API key.
model_name: Set the model name you wish to evaluate.
Run the Evaluation:

Place test.jsonl and both .py files in the same directory.
Execute the appropriate script for evaluation:
For OpenAI models:
bash
复制
python OpenAI Model Evaluation.py
For other large models:
bash
复制
python Other Model Evaluation.py
Results:

After running the scripts, the evaluation results will be saved as .jsonl files:
OpenAI_model_evaluation_results.jsonl for OpenAI models.
other_model_evaluation_results.jsonl for other models.
These files contain both the summary statistics and detailed evaluation for each question, including accuracy, consistency, and token usage.

Feel free to modify the scripts for your specific needs, such as adjusting the number of runs for each question or changing the output format.
