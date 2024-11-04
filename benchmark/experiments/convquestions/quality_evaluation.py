import json
import os

from datetime import datetime
import langchain
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.output_parsers import (
    ResponseSchema,
    StructuredOutputParser,
    PydanticOutputParser,
)
from pydantic import BaseModel

import time

langchain.debug = True

llm = None
key = ""


def create_prompt_chatty_gen_setA(llm):
    prompt = PromptTemplate(
        input_variables=[
            "conv_questions",
            "chatty_gen_questions",
        ],
        template="""
        You are given two sets of questions: Set A and Set B. Both sets ask questions about the same topic. Consider factors such as fluency, clarity, and variety in the questions. Based on these criteria, determine which set has better overall quality. Return only Set A or Set B.
        
        Set A: 
        
        {chatty_gen_questions}
        
        Set B:
        
        {conv_questions}
        """,
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )
    return chain


def create_prompt_chatty_gen_setB(llm):
    prompt = PromptTemplate(
        input_variables=[
            "conv_questions",
            "chatty_gen_questions",
        ],
        template="""
        You are given two sets of questions: Set A and Set B. Both sets ask questions about the same topic. Consider factors such as fluency, clarity, and variety in the questions. Based on these criteria, determine which set has better overall quality. Return only Set A or Set B.

        Set A: 

        {conv_questions}

        Set B:

        {chatty_gen_questions}
        """,
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )
    return chain


def read_files(exp_name):
    with open(f"{exp_name}/chatty_gen_data.json", "r") as f:
        chatty_gen_questions = json.load(f)
    with open(f"{exp_name}/conv_data.json", "r") as f:
        conv_questions = json.load(f)
    return chatty_gen_questions, conv_questions


def read_files_2():
    with open("modified-convex.json", "r") as f:
        data = json.load(f)
    conv_questions = []
    chatty_gen_questions = []
    for x in data["Results"]:
        conv_questions.append(x["ConvQuestions_dialogue_modified"])
        chatty_gen_questions.append(x["Chatty_Gen_dialogue"])
    return chatty_gen_questions, conv_questions


exp_name = "multi-llm-new-7q"
# exp_name = "codellama-13b-new"
# exp_name = "gpt"


def evaluate(evaluation_name, llm, sleep):
    chatty_gen_questions, convquestions = read_files(exp_name)
    tie = 0
    chatty_gen_score = 0
    conv_score = 0
    counter = 0
    logging_result = list()

    for chatty_gen, conv in zip(chatty_gen_questions, convquestions):
        counter += 1
        log_obj = {"Chatty_Gen_dialogue": chatty_gen, "ConvQuestions_dialogue": conv}
        chatty_gen_first_chain = create_prompt_chatty_gen_setA(llm)
        chatty_gen_last_chain = create_prompt_chatty_gen_setB(llm)

        result_1 = chatty_gen_first_chain.generate(
            [{"conv_questions": conv, "chatty_gen_questions": chatty_gen}], None
        )
        result_1 = result_1.generations[0][0].text.lower()
        if sleep:
            time.sleep(30)
        result_2 = chatty_gen_last_chain.generate(
            [{"conv_questions": conv, "chatty_gen_questions": chatty_gen}], None
        )
        result_2 = result_2.generations[0][0].text.lower()
        if sleep:
            time.sleep(30)
        if result_1 == result_2:
            tie = tie + 1
            log_obj["result"] = "Tie"
        elif "set a" in result_1 and "set b" in result_2:
            chatty_gen_score = chatty_gen_score + 1
            log_obj["result"] = "Chatty_Gen"
        elif "set b" in result_1 and "set a" in result_2:
            conv_score = conv_score + 1
            log_obj["result"] = "ConvQuestions"
        else:
            print("Error ", result_1, result_2)
        print(counter)
        logging_result.append(log_obj)

        with open(f"Evaluation-{evaluation_name}.json", "w") as f:
            output = {
                "Results": logging_result,
                "Analysis": {
                    "Tie": tie,
                    "ConvQuestions": conv_score,
                    "Chatty_Gen": chatty_gen_score,
                },
            }
            json.dump(output, f, indent=4)
    return tie, chatty_gen_score, conv_score


class Output(BaseModel):
    answer_set: str
    answer_reason: str


op_parser = PydanticOutputParser(pydantic_object=Output)
json_format_instructions = op_parser.get_format_instructions()


def create_prompt_human(llm):
    prompt = PromptTemplate(
        input_variables=[
            "first_set_questions",
            "second_set_questions",
        ],
        partial_variables={"format_instructions": json_format_instructions},
        template="""
        You are given two sets of questions: Set A and Set B. Both sets ask questions about the same topic. Your task is to identify which set of questions are written by Human. Return only Set A or Set B. And also give the reason for it.
        
        Set A: 
        
        {first_set_questions}
        
        Set B: 
        
        {second_set_questions}
        
        {format_instructions}
        output:
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True, output_parser=op_parser)
    return chain


def create_prompt_variation(llm, prompt):
    prompt = PromptTemplate(
        input_variables=[
            "first_set_question",
            "second_set_question",
        ],
        template=prompt,
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True, output_parser=op_parser)
    return chain


def evaluate_2(evaluation_name, llm, sleep):
    chatty_gen_questions, convquestions = read_files()
    tie = 0
    chatty_gen_score = 0
    conv_score = 0
    counter = 0
    logging_result = list()
    total = 0
    fp = 0

    for chatty_gen, conv in zip(chatty_gen_questions, convquestions):
        counter += 1
        log_obj = {"Chatty_Gen_dialogue": chatty_gen, "ConvQuestions_dialogue": conv}
        llm_chain = create_prompt_human(llm)

        result_1 = llm_chain.generate(
            [{"first_set_questions": conv, "second_set_questions": chatty_gen}], None
        )
        result_1 = result_1.generations[0]
        result_1 = op_parser.parse_result(result_1).dict()
        answer1 = result_1["answer_set"]
        answer1_reason = result_1["answer_reason"]
        print(type(answer1), answer1_reason)
        if sleep:
            time.sleep(30)
        result_2 = llm_chain.generate(
            [{"first_set_questions": chatty_gen, "second_set_questions": conv}], None
        )
        result_2 = result_2.generations[0]
        result_2 = op_parser.parse_result(result_2).dict()
        answer2 = result_2["answer_set"]
        answer2_reason = result_2["answer_reason"]
        if sleep:
            time.sleep(30)

        if "set b" in answer1.lower() or "set a" in answer2.lower():
            fp += 1
        total += 2

        log_obj["answer1"] = answer1
        log_obj["answer1_reason"] = answer1_reason
        log_obj["answer2"] = answer2
        log_obj["answer2_reason"] = answer2_reason
        logging_result.append(log_obj)

        with open(f"Evaluation-{evaluation_name}.json", "w") as f:
            output = {
                "Results": logging_result,
                "Analysis": {
                    "total": total,
                    "True Positive": total - fp,
                    "False Positive": fp,
                },
            }
            json.dump(output, f, indent=4)
    return tie, chatty_gen_score, conv_score


def evaluate_prompt_variations(llm, sleep):
    chatty_gen_questions, convquestions = read_files_2()
    with open("experiment-prompts.json", "r") as f:
        prompts = json.load(f)

    for p in prompts:
        prompt = p.get("prompt")
        prompt_name = p.get("prompt_name")
        llm_chain = create_prompt_variation(llm, prompt)
        logging_result = list()
        tie = 0
        chatty_gen_score = 0
        conv_score = 0
        counter = 0
        for chatty_gen, conv in zip(chatty_gen_questions, convquestions):
            counter += 1
            log_obj = {
                "Chatty_Gen_dialogue": chatty_gen,
                "ConvQuestions_dialogue": conv,
            }
            result_1 = llm_chain.generate(
                [{"first_set_question": conv, "second_set_question": chatty_gen}], None
            )
            result_1 = result_1.generations[0][0].text.lower()
            if sleep:
                time.sleep(30)
            result_2 = llm_chain.generate(
                [{"first_set_question": chatty_gen, "second_set_question": conv}], None
            )
            result_2 = result_2.generations[0][0].text.lower()

            if sleep:
                time.sleep(30)
            if result_1 == result_2:
                tie = tie + 1
                log_obj["result"] = "Tie"
            elif "set a" in result_1 and "set b" in result_2:
                conv_score = conv_score + 1
                log_obj["result"] = "ConvQuestions"
            elif "set b" in result_1 and "set a" in result_2:
                chatty_gen_score = chatty_gen_score + 1
                log_obj["result"] = "Chatty_Gen"
            else:
                print("Error ", result_1, result_2)
            logging_result.append(log_obj)

        with open(
            f"prompt-variations-modified/evaluation-{prompt_name}.json", "w"
        ) as f:
            output = {
                "prompt_name": prompt_name,
                "Prompt": prompt,
                "Results": logging_result,
                "Analysis": {
                    "Tie": tie,
                    "ConvQuestions": conv_score,
                    "Chatty_Gen": chatty_gen_score,
                },
            }
            json.dump(output, f, indent=4)


def modify_results():
    files = os.listdir("prompt-variations")
    for fname in files:
        fname = f"prompt-variations/{fname}"
        with open(fname) as f:
            data = json.load(f)

        new_results = []
        results = data.get("Results")
        for r in results:
            if r["result"] == "Chatty_Gen":
                r["result"] = "ConvQuestions"
            elif r["result"] == "ConvQuestions":
                r["result"] = "Chatty_Gen"
            new_results.append(r)
        data["Results"] = new_results
        old_analysis = data["Analysis"]
        new_analysis = {
            "Tie": old_analysis["Tie"],
            "ConvQuestions": old_analysis["Chatty_Gen"],
            "Chatty_Gen": old_analysis["ConvQuestions"],
        }
        data["Analysis"] = new_analysis
        with open(fname, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    # # llm_type = 'gpt'
    llm_type = "gemini"
    current_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    # experiment_name = f"gemini-{current_time_str}"
    experiment_name = f"{exp_name}-{current_time_str}"

    # experiment_name = f"codellama-13b-{current_time_str}"
    if llm_type == "gpt":
        llm = OpenAI(model_name="gpt-4o", temperature=0, api_key=key, max_retries=0)
    elif llm_type == "gemini":
        llm = VertexAI(model_name="gemini-1.5-pro-001", temperature=0, streaming=False)
    t, c, cq = evaluate(experiment_name, llm, llm_type == "gemini")
    print("Tie: ", t, "\nChatty-Gen: ", c, "\nConv-Questions: ", cq)

    # first_set = "Set A"
    # second_set = "Set B"
    # llm_type = "gpt"
    # current_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    # experiment_name = f"e2-gemini-{current_time_str}"
    # if llm_type == "gpt":
    #     llm = OpenAI(model_name="gpt-4o", temperature=0, api_key=key, max_retries=0)
    # elif llm_type == "gemini":
    #     llm = VertexAI(model_name="gemini-1.5-pro-001", temperature=0, streaming=False)
    # t, c, cq = evaluate_2(experiment_name, llm, llm_type == "gemini")
    # print("Tie: ", t, "\nChatty-Gen: ", c, "\nConv-Questions: ", cq)

    # llm_type = "gpt"
    # if llm_type == "gpt":
    #     llm = OpenAI(model_name="gpt-4o", temperature=0, api_key=key, max_retries=0)
    # elif llm_type == "gemini":
    #     llm = VertexAI(model_name="gemini-1.5-pro-001", temperature=0, streaming=False)
    # evaluate_prompt_variations(llm, llm_type == "gemini")
