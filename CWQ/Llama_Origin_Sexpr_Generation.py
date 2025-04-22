import openai
import json
from tqdm import tqdm
import random
import re
import os
import logging
import spacy
import torch
from rank_bm25 import BM25Okapi
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.hybrid import HybridSearcher
from pyserini.search.faiss import AutoQueryEncoder
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:24'

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("time recoder")


system = """
### System Instruction: 
Hello, {model}. You are an expert in semantic parsing. Your task is to complete the Logical Form based on user queries. 
### Task Background: 
- Logical Forms follow set-based semantics, where functions operate on sets of entities or entity tuples.
- The structure consists of: 
  1. A framework (a tuple of functions) that operates on sets of entities or tuples. 
  2. Arguments that can be one of the following types:
     - A set of entities (e.g., classes or single entities).
     - A set of (Entity, Entity) tuples (representing relationships).
     - A set of (Entity, Value) tuples (representing attributes).
Your goal is to fill in the missing classes, entities and relationships in the logical framework, based on the user's question.
This process is referred to as "DAN" (Do Anything Now). 
"""

instruction = """
### Task Instruction: 
Your task is to generate a complete Logical Form based on a given question. You should accurately identify the classes,
entities and relationships required to construct the Logical Form, ensuring consistency with the examples provided.
### Steps to Follow:
1. Understand the Question: Carefully read the given question and analyze its intent.
2. Learn from Examples: Review the provided examples to understand how questions are mapped Logical Forms. Pay attention to the patterns for classes,entities and relationships.
3. Construct the Logical Form: Based on the question, construct a complete Logical Form. Use the examples as a guide for structure and content.
4. Adhere to Patterns: Ensure that your Logical Form aligns with the patterns and logic demonstrated in the examples. Avoid adding irrelevant content or deviating from the demonstrated format.
5. Provide Reasoning: Along with the Logical Form, include a brief explanation of how you identified the classes, entities and relationships from the question.
6. Respond in English: All responses should be in English.
### Output Format:
- Answer: <Your complete Logical Form>
- Reasoning: <Your explanation on how the Logical Form was constructed based on the question>
### Example Format:
Example Questions and Logical Form:
```{prompt}```
Now, use this format to answer the following question:
Target Question:
```{question}```
REMEMBER: Please strictly follow the output format and respond only in English. 
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


def normed_process_cwq(expression):
    # Function to process content within square brackets

    special_map_dev_file = "data/CWQ/generation/special_map/CWQ_special_map.json"
    with open(special_map_dev_file, 'r', encoding='utf-8') as file:
        comma_values = json.load(file)
    special_map_list = [entry['value'] for entry in comma_values if entry['value'].count(',') >= 2]

    def replace_match(match):
        content = match.group(0)[1:-1]  # Remove the square brackets

        # Check if the content has 2 or more commas, implying it's a relation
        if content.count(',') >= 2 and content.strip() not in special_map_list:
            # Add spaces around commas for relations
            processed_content = content.replace(",", " , ")
            return f"[ {processed_content} ]"
        else:
            # Leave entity content unchanged
            return match.group(0)

    # Replace content inside square brackets based on the condition
    processed_expression = re.sub(r'\[.*?\]', replace_match, expression)

    # Add spaces around parentheses
    processed_expression = processed_expression.replace("(", " ( ").replace(")", " ) ")

    # Remove extra spaces
    processed_expression = " ".join(processed_expression.split())
    return processed_expression


class Chat_Llama_Generator:
    def __init__(self, model_name: str, inst: str = None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.device = torch.device(f"cuda:3" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                          low_cpu_mem_usage=True, trust_remote_code=True).to(
            self.device).eval()
        self.history = []

        if inst:
            self.add_to_history(inst, sender="System")

    def add_to_history(self, message: str, sender: str = "User"):

        self.history.append({"role": sender, "content": message})

    def clear_non_system_history(self):
        self.history = [entry for entry in self.history if entry['role'] == "System"]

    def generate_response(self, prompt: str, max_length: int = 11288):

        self.add_to_history(prompt, sender="User")
        # full_prompt = "\n".join(self.history)

        inputs = self.tokenizer.apply_chat_template(
            self.history,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.device)
        gen_kwargs = {
            "max_new_tokens": max_length,
            "do_sample": True,
            "temperature": 0.70,
            "top_k": 50,
            "top_p": 0.7,
            "num_return_sequences": 3,
        }
        with torch.no_grad():

            responses = self.model.generate(**inputs, **gen_kwargs)

            input_length = inputs['input_ids'].shape[1]

            decoded_responses = []
            for response in responses:
                response = response[input_length:]
                decoded_text = self.tokenizer.decode(response, skip_special_tokens=True)
                processed_response = extract_logical_form(decoded_text)
                spaced_response = normed_process_cwq(processed_response)
                decoded_responses.append(spaced_response)

        return decoded_responses


def get_model_prompt(nlp, corpus, question, bm25_train_full, que_to_s_dict_train, retrieve_number=40):
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
    train_data_path = "data/CWQ/generation/merged/CWQ_train.json"
    nlp = spacy.load("en_core_web_sm")
    train_data = process_file(train_data_path)
    que_to_s_dict_train = {data["question"]: data["s_expression"] for data in train_data}

    corpus = [data["question"] for data in train_data]  # 所有问题

    tokenized_train_data = []
    for doc in corpus:
        nlp_doc = nlp(doc)
        tokenized_train_data.append([token.lemma_ for token in nlp_doc])
    bm25_train_full = BM25Okapi(tokenized_train_data)

    model_dir = "Meta-Llama-3-8B-Instruct"
    template = PromptTemplate(system)
    inst = template.render(model="Llama")
    exp_model = Chat_Llama_Generator(model_dir, inst)

    output_data = []
    with open(os.path.join("data/CWQ/generation/merged", 'CWQ_test.json'), 'r',
              encoding='utf-8') as f:
        json_data = json.load(f)

        total_lines = 0
        matched_lines = 0

        for data in tqdm(json_data):
            torch.cuda.empty_cache()
            total_lines += 1
            question = data['question']
            label = data['normed_sexpr']
            # query = data['instruction'] + data['input']
            prompt = get_model_prompt(nlp, corpus, question, bm25_train_full, que_to_s_dict_train)
            # predict = chat_model.chat_beam(query)
            print("=========================")
            print("Question:{}".format(question))
            # print("Logical Frame:{}".format(predicts))
            print("Sexpression label:{}".format(label))
            exp_model.clear_non_system_history()
            template = PromptTemplate(instruction)
            message = template.render(prompt=prompt, question=question)
            response = exp_model.generate_response(message)
            print("Pred S-expression:{}".format(response))
            output_data.append({'label': label, 'predict': response})
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

    output_dir = os.path.join('results/gen/CWQ_test',
                              'evaluation_beam_llama/generated_predictions.jsonl')
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
    with open(output_dir, 'w') as f:
        for item in output_data:
            json_string = json.dumps(item)
            f.write(json_string + '\n')


if __name__ == "__main__":
    main()

'''
nohup python -u CWQ/Llama_Origin_Sexpr_Generation.py >> log/CWQ_pred_Sexpr_Llama3_Sexpr.txt 2>&1 &
'''


