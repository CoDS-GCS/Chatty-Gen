"""
webnlg_qa.py

question answer pair generation from given description of webnlg example.
it uses openai's gpt-3.5 for generation with <Q></Q><A></A> format.
the generated output data will be used to question-asnwer task specific finetuning of LLM.
"""

import ast
import os
import re
import random
import jsonlines
import time
import argparse
import json
import openai
from openai.error import (
    RateLimitError,
    APIError,
    APIConnectionError,
    Timeout,
    ServiceUnavailableError,
)
import backoff
from typing import List, Dict

# import vertexai
# from vertexai.preview.language_models import TextGenerationModel

openai.api_key = os.getenv("OPENAI_API_KEY")


ChatMessage = Dict[str, str]
ChatPrompt = List[ChatMessage]


def chat_prompt_to_text_prompt(prompt: ChatPrompt, for_completion: bool = True):
    """
    Refer OpenAIChatMessage to OpenAICreateChatPrompt in openai/eval
    """
    chat_to_prefixes = {
        "system": "system: ",
        "user": "user: ",
        "assistant": "assistant: ",
    }
    if len(prompt) == 1:
        return prompt[0]["content"]

    text = ""
    for msg in prompt:
        role = msg["name"] if "name" in msg else msg["role"]
        prefix = chat_to_prefixes.get(role, role.capitalize() + ":")
        content = msg["content"]
        text += f"{prefix}{content}\n"

    if for_completion:
        text += "assistant: "
    return text.lstrip()


def chat_prompt_template(question, text_format: bool = False):
    prompt = [
        {
            "role": "system",
            "content": "You are good assistant that follows the instruction,Given a question you must answer all possible answers in list format that can be converted to python list using 'eval()'. In case of single word answer use `['your answer']` format. In case of no specific answer return `[]`",
        },
        {
            "role": "system",
            "content": "Don’t justify your answers. Don’t give information not mentioned in the CONTEXT INFORMATION.",
        },
        {"role": "user", "content": f"Question: {question}"},
    ]
    if text_format:
        prompt = chat_prompt_to_text_prompt(prompt)
    return prompt


def transform_fn(text: str):
    """
    text starting with word `Answer: []`
    apply eval on string after striping the `Answer:`
    """
    answer_list = []
    try:
        answer_list = ast.literal_eval(text)
        if type(answer_list) == str:
            answer_list = [answer_list]
    except (TypeError, MemoryError, SyntaxError, ValueError):
        if (
            text.startswith("'")
            and text.endswith("'")
            or text.startswith('"')
            and text.endswith('"')
        ):
            answer_list = [answer_list]
        else:
            answer_list = []

    return answer_list


def test_transform_fn():
    # text = 'Answer:     ["hwl","fdsa","fdsafds", "fdasfd"]'
    text = '"hwl"'
    print(text)
    out = transform_fn(text)
    expected = ["hwl"]
    print(out)
    assert expected == out, "transform_fn failed"


def transform_llm_answer(answers: List[str], transform_fn):
    answers = [transform_fn(answer) for answer in answers]
    return answers


@backoff.on_exception(
    backoff.expo,
    (RateLimitError, APIError, APIConnectionError, Timeout, ServiceUnavailableError),
)
def openai_chat_completion_api_call(prompt, post_transform_fn):
    """
    prompt: List[dict] chat format
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=0,
        # max_tokens=4096,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    generated_texts = [
        choice.message["content"].strip() for choice in response["choices"]
    ]
    generated_texts = transform_llm_answer(generated_texts, post_transform_fn)
    return generated_texts


def load_benchmark_file(filepath: str):
    data = []
    with open(filepath, "r") as f:
        filelines = f.__iter__()
        data = [json.loads(line) for line in filelines]
    return data


@backoff.on_exception(
    backoff.expo,
    (RateLimitError, APIError, APIConnectionError, Timeout, ServiceUnavailableError),
)
def vertexai_llm_textgen(
    content: str,
    project_id: str = "musix-308909",
    model_name: str = "text-bison@001",
    temperature: float = 0.2,
    max_decode_steps: int = 256,
    top_p: float = 0.8,
    top_k: int = 40,
    location: str = "us-central1",
    tuned_model_name: str = "",
):
    """Predict using large language model"""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
        model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,
    )
    return response


def vertexai_chat_completion_api_call(prompt, post_transform_fn):
    response = vertexai_llm_textgen(prompt)
    generated_texts = [response.text.strip()]
    generated_texts = transform_llm_answer(generated_texts, post_transform_fn)
    return generated_texts


def get_llm_answers(question, llm="openai"):
    """
    use template_fn to get the text prompt
    """
    answers = []
    if llm == "openai":
        prompt = chat_prompt_template(question, text_format=False)
        answers = openai_chat_completion_api_call(prompt, transform_fn)
    elif llm == "google":
        prompt = chat_prompt_template(question, text_format=True)
        # print(prompt)
        answers = vertexai_chat_completion_api_call(prompt, transform_fn)

    return answers


def evaluate(answers, ground_truth):
    pass


# def main():
#     supported_args = {
#         "model": ["openai", "google"],
#         "benchmark": ["yago", "mag", "dblp", "qald"],
#     }
#     args_parser = argparse.ArgumentParser()
#     args_parser.add_argument(
#         "--model_name",
#         choices=supported_args["model"],
#         help=f'allowed values: {",".join(supported_args["model"])}',
#         required=True,
#     )
#     args_parser.add_argument(
#         "--benchmark_name",
#         choices=supported_args["benchmark"],
#         help=f'allowed values: {",".join(supported_args["benchmark"])}',
#         required=True,
#     )
#     args_parser.add_argument(
#         "--data_directory",
#         help="directory path for benchmark's json file",
#         default="data",
#         required=True,
#     )
#     args_parser.add_argument(
#         "--out_directory",
#         help="directory path for llm answer for benchmark's question json file",
#         default="llm_out",
#         required=True,
#     )
#     args = args_parser.parse_args()
#
#     if not os.path.exists(args.data_directory):
#         print("data directory path does not exists!!")
#
#     filepath = os.path.join(args.data_directory, f"{args.benchmark_name}.jsonl")
#
#     data = load_benchmark_file(filepath)
#     os.makedirs(args.out_directory, exist_ok=True)
#     out_file = f"{args.benchmark_name}_{args.model_name}_{time.time()}.jsonl"
#     out_file_path = os.path.join(args.out_directory, out_file)
#     with open(out_file_path, "a") as f:
#         for idx, q in enumerate(data):
#             qid = q["id"]
#             question = q["question"]
#             ground_truth = q["answers"]
#             print(f"Question Id: {qid}")
#             answers = get_llm_answers(question, args.model_name)
#             jsondump = json.dumps(
#                 {
#                     "benchmark": args.benchmark_name,
#                     "id": qid,
#                     "question": question,
#                     "model": args.model_name,
#                     "answers": answers,
#                     "gtruth": ground_truth,
#                     "time": time.time(),
#                 }
#             )
#             f.write(jsondump + "\n")
#             time.sleep(5)
#             if idx % 15 == 0:
#                 time.sleep(20)

@backoff.on_exception(
    backoff.expo,
    (RateLimitError, APIError, APIConnectionError, Timeout, ServiceUnavailableError),
)
def openai_chat_generate(prompt):
    """
    prompt: List[dict] chat format
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=0,
        # max_tokens=4096,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    generated_texts = [
        choice.message["content"].strip() for choice in response["choices"]
    ]
    print(generated_texts)
    return generated_texts

def webnlg_chat_prompt_template(input, example, text_format: bool = False):
    prompt = [
        {
            "role": "system",
            "content": "You are good assistant that follows the instruction, given the input generate one comma separated Question-Answer pair from it.",
        },
        {
            "role": "system",
            "content": "Don’t justify your answers. Don’t give information not mentioned in the CONTEXT INFORMATION.",
        },
        {
            "role": "user",
            "content": f'Input: {example["input"]}',
        },
        {
            "role": "assistant",
            "content": f'Output: <Q>{example["question"]}</Q><A>{example["answer"]}</A>',
        },
        {
            "role": "user",
            "content": f'Input: {input}',
        },
        {
            "role": "assistant",
            "content": 'Output:',
        },
    ]
    if text_format:
        prompt = chat_prompt_to_text_prompt(prompt)
    return prompt


def get_random_example():
    examples = [
            ("Bakewell pudding 's main ingredients are almond , jam , butter and eggs . It is a dessert from the Derbyshire Dales region .", 
             "What are the main ingredients of Bakewell pudding?",
             "almond, jam, butter, and eggs"),
            ("The city of Lahore , Pakistan , is served by Allama Iqbal International airport . It is operated by the Pakistan Civil Aviation Authority . It has a runway length of 2900 .",
             "Which organization operates Allama Iqbal International Airport in Lahore, Pakistan?",
             "The Pakistan Civil Aviation Authority"),
            ("The Acharya Institute of Technology is in Bangalore . It has 700 postgraduate students and is affiliated with the Visvesvaraya Technological University .",
             "Where is the Acharya Institute of Technology located?",
             "Bangalore")]
    example_list = [
            {
                "input": t[0],
                "question": t[1],
                "answer": t[2],
            }
            for t in examples
    ]
    # print(example_list)

    return example_list[random.randint(0, len(example_list)-1)]


def extract_qa_tuples(text_list):
    result_tuples = []

    for item in text_list:
        question_match = re.search(r"<Q>(.*?)<\/Q>", item)
        answer_match = re.search(r"<A>(.*?)<\/A>", item)

        if question_match and answer_match:
            question = question_match.group(1)
            answer = answer_match.group(1)
            result_tuples.append((question, answer))
    return result_tuples


def generate_webnlg():
    # first read the file
    with open("webnlg_train.json", "r") as f:
        data = json.load(f)
    sample_size = int(len(data) * 0.1)
    sampled = random.sample(data, sample_size)
    generated_data = []
    for idx, q in enumerate(sampled):
        t_input = q["input"]
        example = get_random_example()
        prompt = webnlg_chat_prompt_template(t_input, example, text_format=False)
        response = openai_chat_generate(prompt)
        qa_tuple = extract_qa_tuples(response)
        d = {
                "input": t_input,
                "response": response,
                "qa": qa_tuple
                }
        generated_data.append(d)
        with jsonlines.open("webnlg_gpt_question_answer.jsonl", "a") as f:
            f.write(d)
        if idx % 100 == 0:
            time.sleep(20)
        print(idx, t_input, response)


if __name__ == "__main__":
    generate_webnlg()
    # main()
