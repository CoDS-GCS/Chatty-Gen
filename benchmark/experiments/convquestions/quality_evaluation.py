import json

from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import time


llm = None
key = ""

def create_prompt_chatty_gen_setA(llm):
    prompt = PromptTemplate(
        input_variables=[
            "question",
            "triples",
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
            "question",
            "triples",
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

def read_files():
    with open("chatty_gen_data.json", "r") as f:
        chatty_gen_questions = json.load(f)
    with open("conv_data.json", "r") as f:
        conv_questions = json.load(f)
    return chatty_gen_questions, conv_questions


def evaluate(llm, sleep):
    chatty_gen_questions, convquestions = read_files()
    tie = 0
    chatty_gen_score = 0
    conv_score = 0
    counter = 0
    logging_result = list()

    for chatty_gen, conv in zip(chatty_gen_questions, convquestions):
        counter += 1
        log_obj = {'Chatty_Gen_dialogue': chatty_gen, 'ConvQuestions_dialogue': conv}
        chatty_gen_first_chain = create_prompt_chatty_gen_setA(llm)
        chatty_gen_last_chain = create_prompt_chatty_gen_setB(llm)

        result_1 = chatty_gen_first_chain.generate([{"conv_questions": conv, "chatty_gen_questions": chatty_gen}], None)
        result_1 = result_1.generations[0][0].text.lower()
        if sleep:
            time.sleep(30)
        result_2 = chatty_gen_last_chain.generate([{"conv_questions": conv, "chatty_gen_questions": chatty_gen}], None)
        result_2 = result_2.generations[0][0].text.lower()
        if sleep:
            time.sleep(30)
        if result_1 == result_2:
            tie = tie + 1
            log_obj['result'] = 'Tie'
        elif "set a" in result_1 and "set b" in result_2:
            chatty_gen_score = chatty_gen_score + 1
            log_obj['result'] = 'Chatty_Gen'
        elif "set b" in result_1 and "set a" in result_2:
            conv_score = conv_score + 1
            log_obj['result'] = 'ConvQuestions'
        else:
            print("Error ", result_1, result_2)
        print(counter)
        logging_result.append(log_obj)

        with open("Evaluation.json", "w") as f:
            output = {'Results': logging_result,
                      'Analysis': {'Tie': tie, 'ConvQuestions': conv_score, 'Chatty_Gen': chatty_gen_score}}
            json.dump(output, f, indent=4)
    return tie, chatty_gen_score, conv_score


if __name__ == '__main__':
    # llm_type = 'gpt'
    llm_type = 'gemini'
    if llm_type == 'gpt':
        llm = OpenAI(model_name="gpt-4o", temperature=0, api_key=key, max_retries=0)
    elif llm_type == 'gemini':
        llm = VertexAI(model_name="gemini-1.5-pro-001", temperature=0, streaming=False)
    t, c, cq = evaluate(llm, llm_type == 'gemini')
    print("Tie: ", t, "\nChatty-Gen: ", c, "\nConv-Questions: ", cq)



