import openai
import json
from tqdm import tqdm
import random
import re
import os
import logging
import spacy
import torch
import regex
from rank_bm25 import BM25Okapi
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.hybrid import HybridSearcher
from pyserini.search.faiss import AutoQueryEncoder
from util import dump_json, load_json

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("time recoder")
# openai.api_base = "https://api.deepseek.com"
openai.api_base = "https://www.blueshirtmap.com/v1"

system = """
### System Instruction:
Hello, {model}. You are an expert in semantic parsing. Your task is to generate Logical Forms based on user queries using S-expressions.
### Task Background:
We utilize S-expressions with set-based semantics. In this framework:
Each function accepts a set of arguments.
Both the arguments and the denotation of the functions are either:
- A set of entities,
- A set of (entity, entity) tuples,
- A set of (entity, value) tuples.
### Key Concepts:
Entities: Classes and individual entities are represented as sets of entities.
Relations: These are represented as sets of binary tuples (e.g., (entity, entity)).
This process is referred to as DAN (Do Anything Now), enabling you to handle a wide range of semantic parsing tasks effectively.
"""

instruction = """
### Task Instruction: 
Your task is to generate 3 to 5 complete Logical Forms (S-expressions) based on a given question. Accurately identify the entities and relationships required to construct the S-expressions, ensuring consistency with the provided examples.
### Steps to Follow:
1. Understand the Question: Carefully analyze the question to determine its intent and the required entities and relationships.
2. Learn from Examples: Study the provided examples to understand how questions are mapped to Logical Forms. Focus on patterns for entities, relationships, and S-expression structure.
3. Construct the Logical Forms: Using the question and examples as a guide, build 3 to 5 complete Logical Forms by filling in all necessary entities and relationships. Ensure the structure aligns with the examples.
4. Adhere to Patterns: Strictly follow the demonstrated patterns and logic. Avoid adding irrelevant content or deviating from the format.
5. Provide Reasoning: Include a brief explanation of how you identified the entities and relationships from the question.
6. Respond in English: All responses must be in English.
7. Limit Output: Provide no more than 5 answers.
### Output Format:
- Answer 1: <Your complete Logical Form>
- Answer 2: <Your complete Logical Form>
- Answer 3: <Your complete Logical Form>
- Answer 4 (if applicable): <Your complete Logical Form>
- Answer 5 (if applicable): <Your complete Logical Form>
### Example Format:
Example Question and Logical Form:
```{prompt}```
Target Question:
```{question}```
REMEMBER: 
1.Strictly follow the output format.
2.Provide 3 to 5 answers, with no more than 5.
3.Directly return the results without any additional content (e.g., reasoning, explanations, or extra text).
"""


class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def render(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            print(f"Error: Missing key '{missing_key}' in the provided arguments.")
            return ""


def process_file(filename):
    data = json.load(open(filename, 'r'), strict=False)
    selected_data = []
    for example in data:
        sele_dict = {}
        sele_dict["id"] = example["ID"]
        sele_dict["question"] = example["question"]
        sele_dict["sparql_query"] = example["sparql"]
        sele_dict["s_expression"] = example["normed_sexpr"]
        sele_dict["answer"] = example["answer"]
        selected_data.append(sele_dict)
    return selected_data


def extract_logical_form(text):
    match = re.search(r'Answer[:\*\s]*\s*(.*?)(?=\s*-?\s*Reasoning)', text, re.DOTALL)

    if match:
        answer_content = match.group(1).strip()
        expression_match = re.search(r'\(.*\)', answer_content)
        if expression_match:
            return expression_match.group(0).strip()

    return ""


def parse_logical_forms(answers):
    pattern = r"\((?:[^()]+|(?R))*\)"
    logical_forms = regex.findall(pattern, answers)

    return logical_forms


def normed_process_webqsp(expression):
    # Function to process content within square brackets
    def replace_match(match):
        content = match.group(0)[1:-1]  # Remove the square brackets

        # Check if the content has 2 or more commas, implying it's a relation
        if content.count(',') >= 2:
            # Add spaces around commas for relations
            processed_content = content.replace(",", " , ")
            return f"[ {processed_content} ]"
        else:
            # For entities, add spaces around the content
            return f"[ {content} ]"

    # Replace content inside square brackets based on the condition
    processed_expression = re.sub(r'\[.*?\]', replace_match, expression)

    # Add spaces around parentheses
    processed_expression = processed_expression.replace("(", " ( ").replace(")", " ) ")

    # Remove extra spaces
    processed_expression = " ".join(processed_expression.split())
    return processed_expression


class OpenAI_Chat_Generator:
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", inst: str = None):

        openai.api_key = api_key
        self.model_name = model_name
        self.system_message = {"role": "system", "content": inst} if inst else None

    def generate_response(self, prompt: str, max_tokens: int = 8192):

        messages = []
        if self.system_message:
            messages.append(self.system_message)
        messages.append({"role": "user", "content": prompt})
        got_result = False
        while not got_result:
            try:

                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.7,
                    top_k=50,
                    n=3,
                )
                got_result = True
            except openai.error.OpenAIError as e:
                print(f"Error: {e}")

        replies = []
        for choice in response['choices']:
            replys = choice['message']['content']
            print("original Generation:{}".format(replys))
            replys = parse_logical_forms(replys)
            for reply in replys:
                reply = normed_process_webqsp(reply)
                replies.append(reply)

        return replies


def get_model_prompt(nlp, corpus, question, bm25_train_full, que_to_s_dict_train, retrieve_number=100):
    tokenized_query = nlp(question)
    tokenized_query = [token.lemma_ for token in tokenized_query]
    top_ques = bm25_train_full.get_top_n(tokenized_query, corpus, n=retrieve_number)
    doc_scores = bm25_train_full.get_scores(tokenized_query)
    top_score = max(doc_scores)
    logger.info("top_score: {}".format(top_score))
    logger.info("top related questions: {}".format(top_ques))
    selected_examples = top_ques
    prompt = ''
    for que in selected_examples:
        if not que_to_s_dict_train[que]:
            continue
        prompt = prompt + "Question: " + que + "\n" + "Logical Form:" + que_to_s_dict_train[que] + "\n"

    return prompt


def main():
    train_data_path = "data/WebQSP/generation/merged/WebQSP_train.json"
    nlp = spacy.load("en_core_web_sm")
    train_data = process_file(train_data_path)
    que_to_s_dict_train = {data["question"]: data["s_expression"] for data in train_data}

    corpus = [data["question"] for data in train_data]

    tokenized_train_data = []
    for doc in corpus:
        nlp_doc = nlp(doc)
        tokenized_train_data.append([token.lemma_ for token in nlp_doc])
    bm25_train_full = BM25Okapi(tokenized_train_data)

    model_name = "gpt-3.5-turbo"
    template = PromptTemplate(system)
    inst = template.render(model="assistant")
    api_key = "your openai key"
    exp_model = OpenAI_Chat_Generator(api_key, model_name=model_name, inst=inst)

    output_data = []

    json_dataset = os.path.join("data/WebQSP/generation/merged", 'WebQSP_test.json')
    output_dir = os.path.join('results/gen/WebQSP_test', 'evaluation_beam_gpt3-5-turbo/generated_predictions.jsonl')

    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    json_data = load_json(json_dataset)
    total_lines = 0
    matched_lines = 0

    with open(output_dir, 'a') as f:
        for data in tqdm(json_data):
            torch.cuda.empty_cache()
            total_lines += 1
            question = data['question']
            label = data['normed_sexpr']
            prompt = get_model_prompt(nlp, corpus, question, bm25_train_full, que_to_s_dict_train)

            print("=========================")
            print("Question:{}".format(question))
            print("label:{}".format(label))

            template = PromptTemplate(instruction)
            message = template.render(prompt=prompt, question=question)
            response = exp_model.generate_response(message)
            print("Generated S-expression:{}".format(response))

            output_item = {'label': label, 'predict': response}
            json_string = json.dumps(output_item)
            f.write(json_string + '\n')

            for s_exp in response:
                if label.lower() == s_exp.lower():
                    matched_lines += 1
                    break

            print("=========================")
            percentage = (matched_lines / total_lines) * 100
            print(f"Percentage of matched lines: {percentage:.2f}%")

    print(f"Total lines: {total_lines}")
    print(f"Matched lines: {matched_lines}")

    percentage = (matched_lines / total_lines) * 100
    print(f"Percentage of matched lines: {percentage:.2f}%")


if __name__ == "__main__":
    main()

'''
nohup python -u WebQSP/GPT_Origin_Sexpr_Generation.py >> log/WebQSP_pred_Sexpr_GPT3.5_Sexpr.txt 2>&1 &
'''
