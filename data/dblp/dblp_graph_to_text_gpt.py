"""
dblp_graph_to_text.py

dblp subgraphs to text using gpt-3.5
the generated output data will be used to domain-specific(dblp) finetuning of LLM.
"""

import ast
import os
import re
import random
import jsonlines
import time
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
    return generated_texts

def dblp_chat_prompt_template(input, example, text_format: bool = False):
    prompt = [
        {
            "role": "system",
            "content": "You are good assistant that follows the instruction, Given a subgraph extracted from a knowledge graph in the form of a list of triplets, synthesize a comprehensive textual summary."
        },
        {
            "role": "system",
            "content": "Don’t justify your answers. Don’t give information not mentioned in the CONTEXT INFORMATION.",
        },
        {
            "role": "user",
            "content": f'Input: triplets: {json.dumps(example["triples"])}',
        },
        {
            "role": "assistant",
            "content": f'Output: {example["summary"]}',
        },
        {
            "role": "user",
            "content": f'Input: triplets: {json.dumps(input)}',
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
            {"triples": [{"subject": "Parisa Memarmoshrefi", "predicate": "primary affiliation", "object": "University of Göttingen, Institute for Computer Science, Germany"}, {"subject": "Hang Zhang et al.: Investigating the Learning Phase of an Autonomous Authentication in Mobile Ad-hoc Networks. (2016)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "William Casey et al.: Identity Deception and Game Deterrence via Signaling Games. (2016)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "William Casey et al.: Identity Deception and Game Deterrence via Signaling Games. (2015)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "Parisa Memarmoshrefi et al.: Investigation of a bio-inspired security mechanism in Mobile Ad hoc Networks. (2013)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "Parisa Memarmoshrefi et al.: Autonomous Ant-based Public Key Authentication Mechanism for Mobile Ad-hoc Networks. (2016)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "Parisa Memarmoshrefi: A Bio-Inspired Autonomous Authentication Mechanism in Mobile Ad Hoc Networks. (2012)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "William Casey et al.: Deception, identity, and security: the game theory of sybil attacks. (2019)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "Emmanuel Charleson Dapaah et al.: An AI-Based Transmission Power-Control Certificate Omission in Vehicular Ad-Hoc Networks. (2021)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}], "summary": """Parisa Memarmoshrefi is affiliated with the University of Göttingen, Institute for Computer Science, located in Germany. Her research contributions encompass various aspects of computer science and security, particularly in the field of mobile ad-hoc networks. She has authored several notable papers, shedding light on authentication mechanisms, identity deception, and security challenges within such networks. In 2012, Parisa Memarmoshrefi introduced a bio-inspired autonomous authentication mechanism for mobile ad hoc networks. This innovative approach aimed to enhance security and authentication processes within these dynamic and decentralized networks. Her research continued with investigations into a bio-inspired security mechanism for mobile ad hoc networks, further expanding the understanding of security considerations in this context. A significant contribution came in 2016 when she co-authored a paper titled "Investigating the Learning Phase of an Autonomous Authentication in Mobile Ad-hoc Networks." This work explored the learning phase of autonomous authentication, delving into the complexities of ensuring secure communication in dynamic and potentially unreliable network environments. Parisa Memarmoshrefi collaborated on another 2016 paper titled "Autonomous Ant-based Public Key Authentication Mechanism for Mobile Ad-hoc Networks." This research proposed a novel authentication mechanism inspired by ant behavior, aimed at enhancing the security of communications in ad-hoc networks. In 2015, she was involved in research related to identity deception and game deterrence via signaling games, as evidenced by her co-authorship of the paper titled "Identity Deception and Game Deterrence via Signaling Games." Her work continued to evolve, and in 2019, she contributed to a paper titled "Deception, identity, and security: the game theory of sybil attacks," which further explored the intersections of game theory, identity deception, and security. Parisa Memarmoshrefi's research interests extended beyond ad-hoc networks. In 2021, she co-authored a paper titled "An AI-Based Transmission Power-Control Certificate Omission in Vehicular Ad-Hoc Networks," demonstrating her engagement with security challenges in vehicular ad-hoc networks and the application of AI-based approaches to address these issues. Through her various papers and collaborations, Parisa Memarmoshrefi has demonstrated a deep commitment to advancing the field of computer science and security, particularly in the domains of mobile and vehicular ad-hoc networks. Her research has provided valuable insights into authentication mechanisms, identity deception, security protocols, and the application of innovative techniques inspired by natural systems."""},
            {"triples": [{"subject": "Bernhard Schätz", "predicate": "primary affiliation", "object": "TU Munich, Department of Informatics, Germany"}, {"subject": "Klaus Becker et al.: Deployment Calculation and Analysis for a Fail-Operational Automotive Platform. (2014)", "predicate": "authored by", "object": "Bernhard Schätz"}, {"subject": "Bernhard Schätz: Bericht des AK Requirements Engineering für eingebettete Systeme (REES). (2008)", "predicate": "authored by", "object": "Bernhard Schätz"}, {"subject": "Bernhard Schätz et al.: Anforderungsanalyse in der modellbasierten Entwicklung am Beispiel von AutoFocus. (2004)", "predicate": "authored by", "object": "Bernhard Schätz"}, {"subject": "Tamás Szabó et al.: mbeddr - Extensible Languages for Embedded Software Development. (2014)", "predicate": "authored by", "object": "Bernhard Schätz"}, {"subject": "Andreas Bayha et al.: Model-based software in-the-loop-test of autonomous systems. (2012)", "predicate": "authored by", "object": "Bernhard Schätz"}, {"subject": "Dagmar Koß et al.: Establishing a smart grid node architecture and demonstrator in an office environment using the SOA approach. (2012)", "predicate": "authored by", "object": "Bernhard Schätz"}, {"subject": "Martin Törngren et al.: Education and training challenges in the era of Cyber-Physical Systems: beyond traditional engineering. (2015)", "predicate": "authored by", "object": "Bernhard Schätz"}, {"subject": "Eva Geisberger and Bernhard Schätz: Modellbasierte Anforderungsanalyse mit AutoRAID. (2007)", "predicate": "authored by", "object": "Bernhard Schätz"}], "summary": """In the realm of computer science and informatics, Bernhard Schätz is associated with TU Munich's Department of Informatics in Germany. His work has significantly impacted the field, particularly in the domain of embedded systems development and requirements engineering. Among his contributions, Bernhard Schätz co-authored a paper titled "Deployment Calculation and Analysis for a Fail-Operational Automotive Platform" in 2014. This work explored the intricacies of deploying fail-operational automotive platforms, delving into the calculations and analyses required to ensure their reliability and robustness. In 2008, Bernhard Schätz authored a report for the AK Requirements Engineering für eingebettete Systeme (REES), demonstrating his involvement in advancing requirements engineering practices for embedded systems. His expertise extended to the model-based development realm, as evidenced by his paper "Anforderungsanalyse in der modellbasierten Entwicklung am Beispiel von AutoFocus" in 2004. This work exemplified how model-based techniques could enhance the requirements analysis process, using the AutoFocus platform as a case study. Collaboration played a vital role in Bernhard Schätz's research journey. In conjunction with Tamás Szabó and others, he contributed to "mbeddr - Extensible Languages for Embedded Software Development" in 2014. This collaborative effort aimed to create extensible programming languages tailored for embedded software development. Furthermore, his involvement in "Model-based software in-the-loop-test of autonomous systems" in 2012, authored with Andreas Bayha and colleagues, showcased his commitment to enhancing testing methodologies for autonomous systems through model-based approaches. Bernhard Schätz's influence extended beyond traditional boundaries. In partnership with Dagmar Koß and others, he established a smart grid node architecture and demonstrator using the Service-Oriented Architecture (SOA) approach, as documented in "Establishing a smart grid node architecture and demonstrator in an office environment using the SOA approach" in 2012. His contributions to education and training in the era of Cyber-Physical Systems were evident in the paper "Education and training challenges in the era of Cyber-Physical Systems: beyond traditional engineering" co-authored with Martin Törngren and collaborators in 2015. In collaboration with Eva Geisberger, Bernhard Schätz presented "Modellbasierte Anforderungsanalyse mit AutoRAID" in 2007, further highlighting his prowess in model-based requirements analysis, this time using the AutoRAID methodology. Bernhard Schätz's endeavors have left an indelible mark on the field of informatics, particularly in the areas of embedded systems, requirements engineering, and model-based development. His contributions, both individual and collaborative, continue to shape the landscape of computer science research and practice."""},
            {"triples": [{"subject": "Helmut Dietrich 0001", "predicate": "primary affiliation", "object": "University of Mainz, Germany"}, {"subject": "Helmut Dietrich: Ergebnisse und Bewertungen computererfasster Funktionsstaten der Poliklinik für zahnärztliche Prothetik. (1994)", "predicate": "authored by", "object": "Helmut Dietrich 0001"}, {"subject": "Helmut Dietrich 0001", "predicate": "primary affiliation", "object": "University of Mainz, Germany"}], "summary": """In the realm of academia and research, Helmut Dietrich 0001 is associated with the University of Mainz in Germany. His contributions span various domains, particularly within the field of dental prosthesis and related research. Notably, Helmut Dietrich authored a paper titled "Ergebnisse und Bewertungen computererfasster Funktionsstaten der Poliklinik für zahnärztliche Prothetik" in 1994. This work delved into the results and assessments of computer-captured functional states within the Poliklinik für zahnärztliche Prothetik, shedding light on advancements in dental prosthesis through computer-aided techniques. Helmut Dietrich's affiliation with the University of Mainz underscores his academic involvement and commitment to research in his chosen field. His work reflects a dedication to advancing the understanding and practice of dental prosthesis, potentially contributing to improvements in patient care and treatment approaches. While specific details are limited, Helmut Dietrich's affiliation and research output within the University of Mainz suggest an individual deeply engaged in dental research, particularly in the context of prosthesis and related areas."""}
            ]

    return examples[random.randint(0, len(examples)-1)]


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


def generate_dblp():
    # first read the file
    with open("dblp_subgraphs.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    sample_size = int(len(data) * 1)
    sampled = random.sample(data, sample_size)
    generated_data = []
    for idx, graph in enumerate(sampled):
        t_input = graph["triples"]
        example = get_random_example()
        prompt = dblp_chat_prompt_template(t_input, example, text_format=False)
        response = openai_chat_generate(prompt)
        graph["summary"] = response[0]
        generated_data.append(graph)
        with jsonlines.open("dblp_graph_to_text_gpt.jsonl", "a") as f:
            f.write(graph)
        if idx % 100 == 0:
            time.sleep(20)


if __name__ == "__main__":
    generate_dblp()
    # main()
