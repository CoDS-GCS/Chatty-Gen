import json
from utils import read_json
# import datasets

templates = read_json("templates.json")
template_id_list = [1,2,3,5]
for t in templates:
    if t["id"] is in template_id_list:
        print()
    else:
        continue


# first grab the format of the qrcc's question

# {
#     "Context": [],
#     "Question": "What is a physician's assistant?",
#     "Rewrite": "What is a physician's assistant?",
#     "Answer": "physician assistants are medical providers who are licensed to diagnose and treat illness and disease and to prescribe medication for patients",
#     "Answer_URL": "https://explorehealthcareers.org/career/medicine/physician-assistant/",
#     "Conversation_no": 1,
#     "Turn_no": 1,
#     "Conversation_source": "trec"
# },
# {
#     "Context": [
#         "What is a physician's assistant?",
#         "physician assistants are medical providers who are licensed to diagnose and treat illness and disease and to prescribe medication for patients"
#     ],
#     "Question": "What are the educational requirements required to become one?",
#     "Rewrite": "What are the educational requirements required to become a physician's assistant?",
#     "Answer": "Complete your bachelor's degree (a science or healthcare related major is usually best); Gain experience either working or volunteering in a healthcare setting; Apply to ARC-PA accredited physician assistant programs; Complete a 2-3 year, master's level PA program;",
#     "Answer_URL": "https://www.geteducated.com/careers/how-to-become-a-physician-assistant",
#     "Conversation_no": 1,
#     "Turn_no": 2,
#     "Conversation_source": "trec"


# context = []
# prev_context = []
# for i in range(10):
#     id = i
#     curr_context = None
#     context.append(prev_context + [curr_context])
#     prev_context = context
#
#     # subgraph_type = english description for the type of a subgraph
#
#     # sparql_template = how did you got the sparql, what was the template
#     # sparql = get the sparql of an answer
#     # NNQT_q = natural language question template for sparql template
#     # paraphrased = get the paraphrased version of NNQT question, its different from the standalone question
#     # question = paraphrased is the question
#     # the problem is we know llm can do given the template and entities it can genrate related question in conversation flow, yes
#     # how?
#     # - we need some type of NNQT templates from sparql template - easy use the one from lcqald 
#     #     - prompt-task : "given the sparql template and triplet generate NNQT template", use the example from the lc-qald data #TODO: verify
#     #     - wait first clear what do you mean by NNQT template - do you need it filled entity or just the pure template, or the one filled with entity
#     #         - i think for now lets go with the filled one, later on we will revise,
#     #         - prompt-task : "given the example of sparql template and triplet, generate natural language filled - NNQT question"
#     #     - prompt-task : "paraphrase the NNQT-question" - this is lc-qald standalone question genration
#     #         - now in qrcc - the task was to rewrite the question given the context and not standalone question
#     #         - here our task will be "given the context and a standalone question rewrite it into non-standalone question" #TODO: verify this
#     # - now in the retrieval type we will have answers with real information, how to handle this in case of KG
#     # - well what can you do, your answer should be based on KG - get the label for the answer entity and add to context
#     # - will this work, maybe but at what extent, :thinking
#
#     # lets start with the simple
#     # - task 1: 
#     #
#
#     # standalone = get the rewrite of a paraphrased question
#
#     # answer = get an answer - (it was text in case of qrcc - what will be here :thinking)
#
#
#
#
#
"""
forgot about fully automated benchmark generation, lets start with small, by tomorrow if you have 3 types of question from dblp
0- who is the author of "paper", spo, s
1- it is authored by which author?, spo, o
2- he/she is affiliated with which university?, spo, o

0- who is the author of "paper" and affiliated with university?
1- who else is affiliated with it?
2- it is authored by which author?


- task - given the context and rewritten question come up with the conv-question. (done- need to be tested)
- task - generate various standalone question from the subgraph

two ways to do it, 
1. extract the subgraph and create its text summary - prompt based question generation
    problems - too big prompt, lack the sparql ground truths, can't control the generation process(diversity and conversation flow)

2. extract the subgraph, 
    how? - controls the templates of questions (sparql and NNQT )
    task - TODO: given the sparql template and entity-predicate types or classes create NNQT template or NNQT question. -> verbelize it
    task1 - given the sparql template, generate NNQT template.
    task2 - given NNQT question verbelize it.

for this task if we do it manually what we need, we need different types of questions

"""

"""
lets solve the one type of question

sparql- select ?x {?x ?p ?o}
"""


