import sys
sys.path.append('../')
import os
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
from analysis import analyze_benchmark_sample
from tracer import Tracer
from kg.kg.kg import DblpKG, YagoKG, DbpediaKG, Node, defrag_uri
from seed_node_extractor.sampling import get_seed_nodes
from answer_creation import get_answer_LLM_based, get_answer_query_from_graph, get_LLM_based_postprocessed, updated_get_answer_query_from_graph
from seed_node_extractor import utils

langchain.debug = True

prompt_chains = get_prompt_chains()
pronoun_identification_chain = prompt_chains.get("pronoun_identification_chain")
pronoun_substitution_chain = prompt_chains.get("pronoun_substitution_chain")
n_question_from_subgraph_chain_without_example = prompt_chains.get("n_question_from_subgraph_chain_without_example")
n_question_from_summarized_subgraph_chain_without_example = prompt_chains.get("n_question_from_summarized_subgraph_chain_without_example")
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
    kgs = {"yago": YagoKG, "dblp": DblpKG, "dbpedia": DbpediaKG}
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

def decouple_questions_and_answers(input_obj, seed_node, subgraph, approach):
    questions = list()
    answer_queries = list()
    for element in input_obj["output"]:
        questions.append(element["question"])
        # 1- LLM Based
        # answer_query = get_answer_LLM_based(element["question"], element["triples"], subgraph)
        # 2-Rule based
        # answer_query = get_answer_query_from_graph(element["triples"], seed_node, subgraph, element["question"])
        # 3- LLM based modified
        answer_query = get_LLM_based_postprocessed(element["question"], element["triples"], subgraph, approach)
        # 4- Rule based modified
        # answer_query = updated_get_answer_query_from_graph(element["triples"], subgraph, element["question"])
        answer_queries.append(answer_query)
    return questions, answer_queries



def generate_dialogues(kg_name, dataset_size=2, dialogue_size=2, approach=['subgraph'], label_predicate=None, out_dir='./results'):
    """
    Generate the dialogues given the following inputs:
    kg_name: name of the required knowledge graph
    dataset_size: Number of dialogues in the final benchmark
    dialogue_size: Maximum number of generated question in each dialogue
    approach: The approach to be used to generate the dialogue
              (0: subgraph approach, 1: schema based approach, 2. summarized subgraph approach, 3: All approaches)
    """
    seed_nodes = get_seed_nodes(kg_name, dataset_size)
    if label_predicate is not None:
        utils.excluded_predicates.append(label_predicate)
    # seed_nodes = get_dummy_seeds(kg_name)
    # seed_nodes = [] # will be added by @reham
    # suggestion to use kg.get_seed_nodes(dataset_size)
    if "subgraph" in approach:
        exp_name = f"{kg_name}_e1_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}.json")
        tracer_instance = Tracer(os.path.join(out_dir, 'traces', f'{exp_name}.jsonl'))
        generate_dialogues_from_subgraph(kg_name, seed_nodes, label_predicate, tracer_instance, dialogue_size, output_file)
    if "subgraph-summarized" in approach:
        exp_name = f"{kg_name}_e3_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}.json")
        tracer_instance = Tracer(os.path.join(out_dir, 'traces', f'{exp_name}.jsonl'))
        generate_dialogues_from_schema(kg_name, seed_nodes, label_predicate, tracer_instance, dialogue_size, output_file)
    if "schema" in approach:
        exp_name = f"{kg_name}_e11_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}.json")
        tracer_instance = Tracer(os.path.join(out_dir, 'traces', f'{exp_name}.jsonl'))
        generate_dialogues_from_summarized_subgraph(kg_name, seed_nodes, label_predicate, tracer_instance, dialogue_size, output_file)


def generate_dialogues_from_subgraph(kg_name, seed_nodes, label_predicate, tracer_instance, dialogue_size, output_file):
    """
    kgname
    benchmark size 
    dialogue size

    """
    benchmark_sample = []
    KG = get_kg_instance(kg_name)
    kg = KG(label_predicate)
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
        answer_queries = None
        try:
            with get_openai_callback() as cb:
                logger.info(f"INDEX : {idx} -- question set generation chain start --")
                n = dialogue_size
                output = n_question_from_subgraph_chain_without_example.get("chain").run(
                    {"subgraph": subgraph_uri_str, "n": n}
                )
                question_set, answer_queries = decouple_questions_and_answers(output.dict(), seed, subgraph, "subgraph")
                logger.info(f"INDEX : {idx} -- question set generation chain end --")
                tracer_instance.add_data(idx, "questions", question_set)

                logger.info(f"INDEX : {idx} -- question set : {question_set} --")
                question_0 = question_set[0]

                logger.info(f"INDEX : {idx} -- pronoun identify chain start --")
                payload_dict = pronoun_identification_chain.get("payload")
                ent_pronoun = pronoun_identification_chain.get("chain").run(
                    {"query": question_0, **payload_dict}
                )
                question_0_ent_pron = ent_pronoun.dict()["output"]
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
                transformed_questions = output.dict()["output"]
                logger.info(f"INDEX : {idx} -- pronoun substitute chain end --")
                tracer_instance.add_data(idx, "pron_sub", transformed_questions)

                question_set_dialogue = [question_0, *transformed_questions]
                filtered_set = [question_0, *filter_and_select_questions(transformed_questions)]
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
            "queries": answer_queries,
            "filtered": filtered_set,
            "cost": cb_dict,
        }
        print(dialogue)
        benchmark_sample.append(dialogue)

    benchmark_analysis = analyze_benchmark_sample(benchmark_sample)
    benchmark = {
        "data": benchmark_sample,
        "analysis" : benchmark_analysis,
    }
    directory = pathlib.Path(output_file).parent
    directory.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(benchmark, f, indent=4)


def generate_dialogues_from_summarized_subgraph(kg_name, seed_nodes, label_predicate, tracer_instance, dialogue_size, output_file):
    """
    kgname
    benchmark size 
    dialogue size

    """
    benchmark_sample = []
    KG = get_kg_instance(kg_name)
    kg = KG(label_predicate)
    for idx, seed in enumerate(seed_nodes):
        # seed = Node(uri=URIRef("https://dblp.org/rec/phd/Dobry87"))
        
        logger.info(f"INDEX : {idx} -- start --")
        tracer_instance.add_data(idx, "seed", asdict(seed))

        subgraph = kg.subgraph_extractor(seed)
        subgraph = kg.filter_subgraph(subgraph)
        # subgraph_str = subgraph.__str__(representation='uri')
        # subgraph_str = subgraph.get_quadruple_summary(representation='label')
        subgraph_str = subgraph.get_quadruple_summary(representation='uri')
        
        tracer_instance.add_data(idx, "subgraph", subgraph_str)

        # subgraph_uri_label = subgraph.__str__(representation='label')
        question_set_dialogue = None
        question_set = None
        filtered_set = None
        cb_dict = None
        answer_queries = None
        try:
            with get_openai_callback() as cb:
                logger.info(f"INDEX : {idx} -- question set generation chain start --")
                n = dialogue_size
                output = n_question_from_summarized_subgraph_chain_without_example.get("chain").run(
                    {"subgraph": subgraph_str, "n": n}
                )
                question_set, answer_queries = decouple_questions_and_answers(output.dict(), seed, subgraph, "optimized")
                logger.info(f"INDEX : {idx} -- question set generation chain end --")
                tracer_instance.add_data(idx, "questions", question_set)

                logger.info(f"INDEX : {idx} -- question set : {question_set} --")
                question_0 = question_set[0]

                logger.info(f"INDEX : {idx} -- pronoun identify chain start --")
                payload_dict = pronoun_identification_chain.get("payload")
                ent_pronoun = pronoun_identification_chain.get("chain").run(
                    {"query": question_0, **payload_dict}
                )
                question_0_ent_pron = ent_pronoun.dict()["output"]
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
                transformed_questions = output.dict()["output"]
                logger.info(f"INDEX : {idx} -- pronoun substitute chain end --")
                tracer_instance.add_data(idx, "pron_sub", transformed_questions)

                question_set_dialogue = [question_0, *transformed_questions]
                filtered_set = [question_0, *filter_and_select_questions(transformed_questions)]
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
            "queries": answer_queries,
            "filtered": filtered_set,
            "cost": cb_dict,
        }
        print(dialogue)
        benchmark_sample.append(dialogue)

    benchmark_analysis = analyze_benchmark_sample(benchmark_sample)
    benchmark = {
        "data": benchmark_sample,
        "analysis" : benchmark_analysis,
    }
    directory = pathlib.Path(output_file).parent
    directory.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(benchmark, f, indent=4)



def generate_dialogues_from_schema(kg_name, seed_nodes, label_predicate, tracer_instance, dialogue_size, output_file):
    """
    kgname
    benchmark size 
    dialogue size

    """
    benchmark_sample = []
    KG = get_kg_instance(kg_name)
    kg = KG(label_predicate)
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
                question_0_ent_pron = ent_pronoun.dict()["output"]
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
                transformed_questions = output.dict()["output"]
                logger.info(f"INDEX : {idx} -- pronoun substitute chain end --")
                tracer_instance.add_data(idx, "pron_sub", transformed_questions)

                question_set_dialogue = [question_0, *transformed_questions]
                filtered_set = [question_0, *filter_and_select_questions(transformed_questions)]
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

    benchmark_analysis = analyze_benchmark_sample(benchmark_sample)
    benchmark = {
        "data": benchmark_sample,
        "analysis" : benchmark_analysis,
    }
    directory = pathlib.Path(output_file).parent
    directory.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(benchmark, f, indent=4)