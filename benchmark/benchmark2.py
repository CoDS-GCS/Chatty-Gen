import sys
import random
import pathlib
import json
from typing import List
from dataclasses import asdict
from rdflib import Graph, URIRef, Literal, RDF
from llm.prompt_chains import get_prompt_chains
import langchain
from langchain.callbacks import get_openai_callback
from logger import logger
from tracer import Tracer
from kg.kg.kg import DblpKG, YagoKG, DbpediaKG, Node


langchain.debug = True

prompt_chains = get_prompt_chains()
pronoun_identification_chain = prompt_chains.get("pronoun_identification_chain")
pronoun_substitution_chain = prompt_chains.get("pronoun_substitution_chain")
n_question_from_subgraph_chain_without_example = prompt_chains.get("n_question_from_subgraph_chain_without_example")
n_question_from_schema_chain_without_example = prompt_chains.get("n_question_from_schema_chain_without_example")

dblp_dummy_seeds = {
    "https://dblp.org/rdf/schema#Person": [
        "https://dblp.org/pid/z/ZhongfeiMarkZhang",
        "https://dblp.org/pid/z/YoutaoZhang",
        "https://dblp.org/pid/z/ArkadyBZaslavsky"
    ],
    "https://dblp.org/rdf/schema#Publication":[
        "https://dblp.org/rec/series/sist/BhattacharjeeRB21",
        "https://dblp.org/rec/series/sist/IpinaFSZC16",
        "https://dblp.org/rec/series/smpai/SchenkerKBL05"
    ]
}
yago_dummy_seeds = {
    "http://schema.org/Movie": [
        "http://yago-knowledge.org/resource/Dabangg",
        "http://yago-knowledge.org/resource/Maine_Pyar_Kiya",
        "http://yago-knowledge.org/resource/The_Way_of_the_Dragon"
    ]
}
dbpedia_dummy_seeds = {
    "http://dbpedia.org/ontology/Place": [
        "http://dbpedia.org/resource/Caballo,_New_Mexico", 
        "http://dbpedia.org/resource/Cabin_Lake_(California)", 
        "http://dbpedia.org/resource/Cabool,_Missouri"
    ],
    "http://dbpedia.org/ontology/Actor": [
        "http://dbpedia.org/resource/Camilla_Power", 
        "http://dbpedia.org/resource/Sarika", 
        "http://dbpedia.org/resource/David_Arnott"
    ]
}

def get_dummy_seeds(kg_name) -> List[Node]:
    seeds = []
    if kg_name == "dblp":
        dummy_seeds = dblp_dummy_seeds
    elif kg_name == "yago":
        dummy_seeds = yago_dummy_seeds
    elif kg_name == "dbpedia":
        dummy_seeds = dbpedia_dummy_seeds
    for k, v in dummy_seeds.items():
        for s in v:
            seeds.append(Node(uri=URIRef(s), nodetype=URIRef(k)))
    return seeds

def get_kg_instance(kg_name):
    kgs = {"yago": YagoKG(), "dblp": DblpKG(), "dbpedia": DbpediaKG()}
    kg = kgs.get(kg_name, None)
    if kg is None:
        raise ValueError(f"kg : {kg_name} not supported")
    return kg

def categorize_questions(original_questions):
    # Dictionary to store categorized questions
    categorized_questions = {}

    # Categorize questions based on question type
    for question in original_questions:
        # Convert "Can you list" question to "List" only (case-insensitive)
        if question.lower().startswith("can you list"):
            question = question.replace("Can you list", "List", 1)
            question = question.replace("?", ".", 1)

        # Extract the question type (e.g., "what," "where," "did," "when," etc.)
        question_type = question.split()[0].lower()

        # Add the question to the corresponding category
        if question_type not in categorized_questions:
            categorized_questions[question_type] = [question]
        else:
            categorized_questions[question_type].append(question)

    return categorized_questions


def filter_and_select_questions(original_questions):
    categorized_questions = categorize_questions(original_questions)

    # List to store selected questions
    selected_questions = []

    # Iterate through question types and select one random question per type
    for question_type, questions in categorized_questions.items():
        selected_question = random.choice(questions)
        selected_questions.append(selected_question)

    return selected_questions

def generate_dialogues_from_subgraph(kg_name, dataset_size=3, dialogue_size=3):
    """
    kgname
    benchmark size 
    dialogue size

    """
    exp_name = f"{kg_name}_e1_{dataset_size}_{dialogue_size}"
    output_file = f"results/{exp_name}.json"
    tracer_instance = Tracer(f'traces/{exp_name}.jsonl')

    seed_nodes = get_dummy_seeds(kg_name)
    # seed_nodes = [] # will be added by @reham
    # suggestion to use kg.get_seed_nodes(dataset_size)

    # seed = seed_nodes[0]
    benchmark_sample = []
    kg = get_kg_instance(kg_name)
    for idx, seed in enumerate(seed_nodes):
        # seed = Node(uri=URIRef("https://dblp.org/rec/phd/Dobry87"))
        
        logger.info(f"INDEX : {idx} -- start --")
        tracer_instance.add_data(idx, "seed", asdict(seed))

        subgraph = kg.subgraph_extractor(seed)
        subgraph = kg.filter_subgraph(subgraph)
        subgraph_uri_str = subgraph.__str__(representation='uri')
        
        tracer_instance.add_data(idx, "subgraph", subgraph_uri_str)

        # subgraph_uri_label = subgraph.__str__(representation='label')
        question_set_dialogue = None
        question_set = None
        filtered_set = None
        cb_dict = None
        try:
            with get_openai_callback() as cb:
                logger.info(f"INDEX : {idx} -- question set generation chain start --")
                n = dialogue_size
                output = n_question_from_subgraph_chain_without_example.get("chain").run(
                    {"subgraph": subgraph_uri_str, "n": n}
                )
                question_set = output["output"]
                logger.info(f"INDEX : {idx} -- question set generation chain end --")
                tracer_instance.add_data(idx, "questions", question_set)

                logger.info(f"INDEX : {idx} -- question set : {question_set} --")
                question_0 = question_set[0]

                logger.info(f"INDEX : {idx} -- pronoun identify chain start --")
                payload_dict = pronoun_identification_chain.get("payload")
                ent_pronoun = pronoun_identification_chain.get("chain").run(
                    {"query": question_0, **payload_dict}
                )
                question_0_ent_pron = ent_pronoun["output"]
                logger.info(f"INDEX : {idx} -- pronoun identify chain end --")
                tracer_instance.add_data(idx, "pron_indetify", question_0_ent_pron)

                logger.info(f"INDEX : {idx} -- pronoun substitute chain start --")
                query_dict = {
                    "query_inp": question_set[1:],
                    "query_entity": question_0_ent_pron[0],
                    "query_pronouns": question_0_ent_pron[1],
                }
                payload_dict = pronoun_substitution_chain.get("payload")
                output = pronoun_substitution_chain.get("chain").run(
                    {**query_dict, **payload_dict}
                )
                transformed_questions = output["output"]
                logger.info(f"INDEX : {idx} -- pronoun substitute chain end --")
                tracer_instance.add_data(idx, "pron_sub", transformed_questions)

                question_set_dialogue = [question_0, *transformed_questions]
                filtered_set = filter_and_select_questions(question_set_dialogue)
                cb_dict = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    # "successful_requests": cb.successful_requests,
                    # "total_cost": cb.total_cost
                }
        except Exception as e:
            logger.info(f"INDEX : {idx} -- ERROR: {idx} : {e} --")
            tracer_instance.save_to_file()

        dialogue = {
            "dialogue": question_set_dialogue,
            "original": question_set,
            "filtered": filtered_set,
            "cost": cb_dict,
        }
        print(dialogue)
        benchmark_sample.append(dialogue)

    directory = pathlib.Path(output_file).parent
    directory.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(benchmark_sample, f, indent=4)



def generate_dialogues_from_schema(kg_name, dataset_size=3, dialogue_size=3):
    """
    kgname
    benchmark size 
    dialogue size

    """
    exp_name = f"{kg_name}_e3_{dataset_size}_{dialogue_size}"
    output_file = f"results/{exp_name}.json"
    tracer_instance = Tracer(f'traces/{exp_name}.jsonl')

    seed_nodes = get_dummy_seeds(kg_name)
    # seed_nodes = [] # will be added by @reham
    # suggestion to use kg.get_seed_nodes(dataset_size)

    benchmark_sample = []
    kg = get_kg_instance(kg_name)
    for idx, seed in enumerate(seed_nodes):
        logger.info(f"INDEX : {idx} -- start --")
        tracer_instance.add_data(idx, "seed", asdict(seed))

        seed_schema = kg.schema_extractor(seed)
        seed_schema_uri_str = seed_schema.__str__(representation='uri')
        # seed_schema_label_str = seed_schema.__str__(representation='label')
        
        tracer_instance.add_data(idx, "schema", seed_schema_uri_str)

        question_set_dialogue = None
        question_set = None
        filtered_set = None
        cb_dict = None
        try:
            with get_openai_callback() as cb:
                logger.info(f"INDEX : {idx} -- question set generation chain start --")
                n = dialogue_size
                output = n_question_from_schema_chain_without_example.get("chain").run(
                    {"seed": str(seed), "schema": seed_schema_uri_str, "n": n}
                )
                question_set = output["output"]
                logger.info(f"INDEX : {idx} -- question set generation chain end --")
                tracer_instance.add_data(idx, "questions", question_set)

                logger.info(f"INDEX : {idx} -- question set : {question_set} --")
                question_0 = question_set[0]

                logger.info(f"INDEX : {idx} -- pronoun identify chain start --")
                payload_dict = pronoun_identification_chain.get("payload")
                ent_pronoun = pronoun_identification_chain.get("chain").run(
                    {"query": question_0, **payload_dict}
                )
                question_0_ent_pron = ent_pronoun["output"]
                logger.info(f"INDEX : {idx} -- pronoun identify chain end --")
                tracer_instance.add_data(idx, "pron_indetify", question_0_ent_pron)

                logger.info(f"INDEX : {idx} -- pronoun substitute chain start --")
                query_dict = {
                    "query_inp": question_set[1:],
                    "query_entity": question_0_ent_pron[0],
                    "query_pronouns": question_0_ent_pron[1],
                }
                payload_dict = pronoun_substitution_chain.get("payload")
                output = pronoun_substitution_chain.get("chain").run(
                    {**query_dict, **payload_dict}
                )
                transformed_questions = output["output"]
                logger.info(f"INDEX : {idx} -- pronoun substitute chain end --")
                tracer_instance.add_data(idx, "pron_sub", transformed_questions)

                question_set_dialogue = [question_0, *transformed_questions]
                filtered_set = filter_and_select_questions(question_set_dialogue)
                cb_dict = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    # "successful_requests": cb.successful_requests,
                    # "total_cost": cb.total_cost
                }
        except Exception as e:
            logger.info(f"INDEX : {idx} -- ERROR: {idx} : {e} --")
            tracer_instance.save_to_file()

        dialogue = {
            "dialogue": question_set_dialogue,
            "original": question_set,
            "filtered": filtered_set,
            "cost": cb_dict,
        }
        print(dialogue)
        benchmark_sample.append(dialogue)

    directory = pathlib.Path(output_file).parent
    directory.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(benchmark_sample, f, indent=4)