import sys
import time

sys.path.append('../')
import os
import random
import pathlib
import json
from typing import List
from dataclasses import asdict
from rdflib import Graph, URIRef, Literal, RDF
from llm.prompt_chains import get_prompt_chains, get_num_tokens
import langchain
from langchain.callbacks import get_openai_callback
from logger import logger
from analysis import analyze_benchmark_sample
from tracer import Tracer
from kg.kg.kg import DblpKG, YagoKG, DbpediaKG, Node, defrag_uri
from seed_node_extractor.sampling import get_seed_nodes
from answer_creation import get_answer_LLM_based, get_answer_query_from_graph, get_LLM_based_postprocessed, updated_get_answer_query_from_graph
from seed_node_extractor import utils
from answer_validation import validate_query
from seed_node_extractor.seed_node_selector import SeedNodeSelector
from prepare_nodes_subgraph import retrieve_seed_nodes_with_subgraphs, retrieve_one_node_with_subgraph, retrieve_seed_nodes_with_subgraphs_new
import ast
import re

langchain.debug = True

prompt_chains = get_prompt_chains()
pronoun_identification_chain = prompt_chains.get("pronoun_identification_chain")
pronoun_substitution_chain = prompt_chains.get("pronoun_substitution_chain")
n_question_from_subgraph_chain_without_example = prompt_chains.get("n_question_from_subgraph_chain_without_example")
n_question_from_summarized_subgraph_chain_without_example = prompt_chains.get("n_question_from_summarized_subgraph_chain_without_example")
n_question_from_schema_chain_without_example = prompt_chains.get("n_question_from_schema_chain_without_example")
n_question_from_subgraph_chain_using_seed_entity = prompt_chains.get("get_n_question_from_subgraph_chain_using_seed_entity")
n_question_from_subgraph_chain_using_seed_entity_and_type = prompt_chains.get("get_n_question_from_subgraph_chain_using_seed_entity_and_type")
pronoun_identification_and_substitution_chain = prompt_chains.get("get_pronoun_identification_and_substitution_chain_without_example")

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

def decouple_questions_and_answers(input_obj, subgraph, approach, endpoint, seed_node_uri):
    questions = list()
    answer_queries = list()
    triples = list()
    query_results = dict()
    for element in input_obj["output"]:
        # 1- LLM Based
        # answer_query = get_answer_LLM_based(element["question"], element["triples"], subgraph)
        # 2-Rule based
        # answer_query = get_answer_query_from_graph(element["triples"], seed_node, subgraph, element["question"])
        # 3- LLM based modified
        answer_query = get_LLM_based_postprocessed(element["question"], element["triples"], subgraph, approach)
        # 4- Rule based modified
        # answer_query = updated_get_answer_query_from_graph(element["triples"], subgraph, element["question"])
        status = validate_query(answer_query, element["triples"], endpoint, subgraph, seed_node_uri, approach)
        if status in query_results:
            query_results[status] += 1
        else:
            query_results[status] = 1

        if status == 'Correct':
            questions.append(element["question"])
            answer_queries.append(answer_query)
            triples.append(element["triples"])

    return questions, answer_queries, triples, query_results



def generate_dialogues(kg_name, dataset_size=2, dialogue_size=2, approach=['subgraph'],  out_dir='./results', prompt=1, use_label=True, seed_nodes_file=None):
    """
    Generate the dialogues given the following inputs:
    kg_name: name of the required knowledge graph
    dataset_size: Number of dialogues in the final benchmark
    dialogue_size: Maximum number of generated question in each dialogue
    approach: The approach to be used to generate the dialogue
              (0: subgraph approach, 1: schema based approach, 2. summarized subgraph approach, 3: All approaches)
    """
    start = time.time()
    sampler = SeedNodeSelector(kg_name, seed_nodes_file)
    # seed_nodes, seednode_to_subgraph_map, kg = retrieve_seed_nodes_with_subgraphs(kg_name, dataset_size, sampler, use_label)
    seed_nodes, seednode_to_subgraph_map, kg = retrieve_seed_nodes_with_subgraphs_new(kg_name, dataset_size, sampler, use_label)
    end = time.time()
    print(f"Seed node selection took {end - start} seconds")
    # seed_nodes, sample_distribution = sampler.retrieve_initial_list_top_k(dataset_size)
    # KG = get_kg_instance(kg_name)
    # kg = KG()
    # type_to_predicate_map = dict()
    # if use_label:
    #     file_name = f"{kg_name}_types_representative.json"
    #     if os.path.exists(file_name):
    #         file = open(file_name, 'r')
    #         type_to_predicate_map = json.load(file)
    #     type_to_predicate_map = get_representative_label_per_node_type(kg.sparql_endpoint, sample_distribution, seed_nodes, type_to_predicate_map, file_name)
    # kg.set_type_to_predicate_map(type_to_predicate_map)
    # kg.set_use_label(use_label)
    # print(type_to_predicate_map)
    
    # seed_nodes = get_dummy_seeds(kg_name)
    # seed_nodes = [] # will be added by @reham
    # suggestion to use kg.get_seed_nodes(dataset_size)
    if "subgraph" in approach:
        exp_name = f"{kg_name}_e1_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}.json")
        tracer_instance = Tracer(os.path.join(out_dir, 'traces', f'{exp_name}.jsonl'))
        generate_dialogues_from_subgraph(seed_nodes, kg, tracer_instance, dialogue_size, output_file, prompt, sampler, seednode_to_subgraph_map)
    if "schema" in approach:
        exp_name = f"{kg_name}_e3_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}.json")
        tracer_instance = Tracer(os.path.join(out_dir, 'traces', f'{exp_name}.jsonl'))
        generate_dialogues_from_schema(seed_nodes, kg, tracer_instance, dialogue_size, output_file, prompt, sampler, seednode_to_subgraph_map)
    if "subgraph-summarized" in approach:
        exp_name = f"{kg_name}_e11_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}.json")
        tracer_instance = Tracer(os.path.join(out_dir, 'traces', f'{exp_name}.jsonl'))
        generate_dialogues_from_summarized_subgraph(seed_nodes, kg, tracer_instance, dialogue_size, output_file, prompt, sampler, seednode_to_subgraph_map)


def generate_dialogues_from_subgraph(initial_seed_nodes, kg, tracer_instance, dialogue_size, output_file, prompt, sampler, seednode_to_subgraph_map):
    """
    kgname
    benchmark size 
    dialogue size

    """

    benchmark_sample = []

    seed_nodes = initial_seed_nodes.copy()
    total_time = 0
    processed_seeds = 0
    context_length_limit_error = 0
    question_validation_error = 0
    triple_validation_error = 0
    dialogue_validation_error = 0
    for idx, seed in enumerate(seed_nodes):
        start_time = time.time()
        logger.info(f"INDEX : {idx} -- start --")
        tracer_instance.add_data(idx, "seed", asdict(seed))
        key = seed.label if seed.label else seed.uri
        subgraph = seednode_to_subgraph_map[key]

        # subgraph = kg.subgraph_extractor(seed)
        # print("Subgraph Extracted")
        # subgraph = kg.filter_subgraph(subgraph, seed)
        subgraph_uri_str = subgraph.__str__(representation='uri')
        tracer_instance.add_data(idx, "subgraph", subgraph_uri_str)

        # subgraph_uri_label = subgraph.__str__(representation='label')
        question_set_dialogue = None
        question_set = None
        filtered_set = None
        cb_dict = None
        answer_queries = None
        triples_used = None
        answer_status_dict = None
        try:
            with get_openai_callback() as cb:
                logger.info(f"INDEX : {idx} -- question set generation chain start --")
                n = dialogue_size
                seed_entity = seed.label if seed.label else seed.uri
                valid_question = False
                valid_triples = False
                retry = 0
                while not valid_question and not valid_triples:
                    output = execute_question_generation_prompt("subgraph", prompt, subgraph_uri_str, n, seed)
                    if output is None:
                        context_length_limit_error += 1
                        break
                    valid_question = validate_questions_output(seed_entity, output)
                    valid_triples = validate_triples_output(subgraph, output, "subgraph")
                    retry += 1
                    if retry == 3:
                        break
                if not valid_question:
                    question_validation_error += 1
                elif not valid_triples:
                    triple_validation_error += 1

                if output is not None and valid_question and valid_triples:
                    question_set, answer_queries, triples_used, answer_status_dict = decouple_questions_and_answers(
                        output, subgraph, "subgraph", kg.sparql_endpoint, seed.uri)

                if question_set and len(question_set) > 2:
                    logger.info(f"INDEX : {idx} -- question set generation chain end --")
                    tracer_instance.add_data(idx, "questions", question_set)

                    logger.info(f"INDEX : {idx} -- question set : {question_set} --")
                    question_0 = question_set[0]

                    logger.info(f"INDEX : {idx} -- pronoun identify and substitute chain start --")
                    seed_entity = seed.label if seed.label else seed.uri
                    valid = False
                    retry = 0
                    while not valid:
                        transformed_questions = execute_dialogue_generation_prompt(seed_entity, question_set)
                        valid = validate_dialogue_output(seed_entity, transformed_questions)
                        retry += 1
                        if retry == 3:
                            break

                    logger.info(f"INDEX : {idx} -- pronoun identify and substitute chain end --")
                    tracer_instance.add_data(idx, "pron_sub", transformed_questions)
                    if valid:
                        question_set_dialogue = [question_0, *transformed_questions]
                    else:
                        dialogue_validation_error += 1
                    # filtered_set = [question_0, *filter_and_select_questions(transformed_questions)]
                    cb_dict = {
                        "total_tokens": cb.total_tokens,
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        # "successful_requests": cb.successful_requests,
                        # "total_cost": cb.total_cost
                    }
                else:
                    # This is a trigger to sample the new node
                    question_set = None
        except Exception as e:
            logger.info(f"INDEX : {idx} -- ERROR: {idx} : {e} --")
            tracer_instance.save_to_file()

        if question_set is None or question_set_dialogue is None or len(question_set) != len(question_set_dialogue):
            # Sample a new node and add it to seed nodes
            new_seed, subgraph = retrieve_one_node_with_subgraph(sampler, seed.nodetype, kg)
            key = new_seed.label if new_seed.label else new_seed.uri
            seed_nodes.append(new_seed)
            seednode_to_subgraph_map[key] = subgraph
        else:
            dialogue = {
                "seed_entity": str(seed.uri),
                "seed_label": str(seed.label) if seed.label else defrag_uri(str(seed.uri)),
                "dialogue": question_set_dialogue,
                "original": question_set,
                "queries": answer_queries,
                "triples": triples_used,
                # "filtered": filtered_set,
                "cost": cb_dict,
                "query_status": answer_status_dict
            }
            print(dialogue)
            benchmark_sample.append(dialogue)
            end_time = time.time()
            total_time += (end_time - start_time)
            processed_seeds += 1
    benchmark_analysis = analyze_benchmark_sample(benchmark_sample)
    benchmark = {
        "data": benchmark_sample,
        "analysis" : benchmark_analysis,
        "total_time": total_time,
        "average_time": total_time / processed_seeds,
        "Skipped Context Length": context_length_limit_error,
        "Question Validation Error": question_validation_error,
        "Triples Validation Error": triple_validation_error,
        "Dialogue Validation Error": dialogue_validation_error,
    }
    directory = pathlib.Path(output_file).parent
    directory.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(benchmark, f, indent=4)

def validate_dialogue_output(seed, dialogue):
    for question in dialogue:
        if seed in question:
            return False
    return True

def validate_questions_output(seed, questions):
    if seed[-1] == '.':
        seed = seed[:-1]
    for question in questions["output"]:
        if seed not in question["question"]:
            return False
    return True

def validate_triples_output(subgraph, output, approach):
    for instance in output["output"]:
        triples = instance["triples"]
        if len(triples) > 0 and '(' not in triples[0] and ')' not in triples[0]:
            if len(triples) == 3:
                triples = [str(tuple(triples))]
            elif len(triples) == 2:
                triples = [str((triples[0], triples[1], ''))]
            instance["triples"] = triples

        for triple in triples:
            if not subgraph.contain_triple(triple, approach):
                return False
    return True


def execute_dialogue_generation_prompt(seed_entity, question_set):
    try:
        output = pronoun_identification_and_substitution_chain.get("chain").run(
            {"entity": seed_entity, "questions": question_set[1:]}
        )
        transformed_questions = output.dict()["output"]
    except Exception as e:
        response = str(e)
        if response.startswith("Failed to parse SchemaInput from completion"):
            start_index = response.index('[')
            end_index = response.index(']')
            transformed_questions = ast.literal_eval(response[start_index:end_index + 1])
    return transformed_questions


def generate_dialogues_from_summarized_subgraph(initial_seed_nodes, kg, tracer_instance, dialogue_size, output_file, prompt, sampler, seednode_to_subgraph_map):
    """
    kgname
    benchmark size 
    dialogue size

    """

    benchmark_sample = []
    seed_nodes = initial_seed_nodes.copy()
    total_time = 0
    processed_seeds = 0
    context_length_limit_error = 0
    question_validation_error = 0
    triple_validation_error = 0
    dialogue_validation_error = 0
    for idx, seed in enumerate(seed_nodes):
        start_time = time.time()
        logger.info(f"INDEX : {idx} -- start --")
        tracer_instance.add_data(idx, "seed", asdict(seed))
        key = seed.label if seed.label else seed.uri
        subgraph = seednode_to_subgraph_map[key]
        # subgraph = kg.subgraph_extractor(seed)
        # subgraph = kg.filter_subgraph(subgraph, seed)
        # subgraph_str = subgraph.__str__(representation='uri')
        # subgraph_str = subgraph.get_quadruple_summary(representation='label')
        # subgraph_str = subgraph.get_quadruple_summary(representation='uri')
        subgraph_str = subgraph.get_summarized_graph_str(approach='no_object')
        tracer_instance.add_data(idx, "subgraph", subgraph_str)

        # subgraph_uri_label = subgraph.__str__(representation='label')
        question_set_dialogue = None
        question_set = None
        filtered_set = None
        cb_dict = None
        answer_queries = None
        triples_used = None
        answer_status_dict = None
        try:
            with get_openai_callback() as cb:
                logger.info(f"INDEX : {idx} -- question set generation chain start --")
                n = dialogue_size
                seed_entity = seed.label if seed.label else seed.uri
                seed_label = seed.label if seed.label else defrag_uri(str(seed.uri))
                valid_question = False
                valid_triples = False
                retry = 0
                while not valid_question and not valid_triples:
                    output = execute_question_generation_prompt("summarized", prompt, subgraph_str, n, seed)
                    if output is None:
                        context_length_limit_error += 1
                        break
                    valid_question = validate_questions_output(seed_label, output)
                    valid_triples = validate_triples_output(subgraph, output, "optimized")
                    retry += 1
                    if retry == 3:
                        break
                if not valid_question:
                    question_validation_error += 1
                elif not valid_triples:
                    triple_validation_error += 1

                if output is not None and valid_question and valid_triples:
                    question_set, answer_queries, triples_used, answer_status_dict = decouple_questions_and_answers(output, subgraph, "optimized", kg.sparql_endpoint, seed.uri)

                if question_set and len(question_set) > 2:
                    logger.info(f"INDEX : {idx} -- question set generation chain end --")
                    tracer_instance.add_data(idx, "questions", question_set)

                    logger.info(f"INDEX : {idx} -- question set : {question_set} --")
                    question_0 = question_set[0]

                    logger.info(f"INDEX : {idx} -- pronoun identify and substitute chain start --")

                    valid = False
                    retry = 0
                    while not valid:
                        transformed_questions = execute_dialogue_generation_prompt(seed_entity, question_set)
                        valid = validate_dialogue_output(seed_label, transformed_questions)
                        retry += 1
                        if retry == 3:
                            break

                    logger.info(f"INDEX : {idx} -- pronoun identify and substitute chain end --")
                    tracer_instance.add_data(idx, "pron_sub", transformed_questions)

                    if valid:
                        question_set_dialogue = [question_0, *transformed_questions]
                    else:
                        dialogue_validation_error += 1
                    # filtered_set = [question_0, *filter_and_select_questions(transformed_questions)]
                    cb_dict = {
                        "total_tokens": cb.total_tokens,
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        # "successful_requests": cb.successful_requests,
                        # "total_cost": cb.total_cost
                    }

                else:
                    # This is a trigger to sample the new node
                    question_set = None
        except Exception as e:
            logger.info(f"INDEX : {idx} -- ERROR: {idx} : {e} --")
            tracer_instance.save_to_file()

        if question_set is None or question_set_dialogue is None:
            # Sample a new node and add it to seed nodes
            new_seed, subgraph = retrieve_one_node_with_subgraph(sampler, seed.nodetype, kg)
            key = new_seed.label if new_seed.label else new_seed.uri
            seed_nodes.append(new_seed)
            seednode_to_subgraph_map[key] = subgraph
        else:
            dialogue = {
                "seed_entity": str(seed.uri),
                "seed_label": seed_label,
                "dialogue": question_set_dialogue,
                "original": question_set,
                "queries": answer_queries,
                "triples": triples_used,
                # "filtered": filtered_set,
                "cost": cb_dict,
                "query_status": answer_status_dict
            }
            print(dialogue)
            benchmark_sample.append(dialogue)
            end_time = time.time()
            total_time += (end_time - start_time)
            processed_seeds += 1
    benchmark_analysis = analyze_benchmark_sample(benchmark_sample)
    benchmark = {
        "data": benchmark_sample,
        "analysis" : benchmark_analysis,
        "total_time": total_time,
        "average_time": total_time / processed_seeds,
        "Context Length Error": context_length_limit_error,
        "Question Validation Error": question_validation_error,
        "Triples Validation Error": triple_validation_error,
        "Dialogue Validation Error": dialogue_validation_error,
    }
    directory = pathlib.Path(output_file).parent
    directory.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(benchmark, f, indent=4)





def generate_dialogues_from_schema(seed_nodes, kg, tracer_instance, dialogue_size, output_file):
    """
    kgname
    benchmark size 
    dialogue size

    """
    benchmark_sample = []

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


def execute_question_generation_prompt(subgraph_approach, prompt, subgraph_str, n, seed):
    output = None
    try:
        if prompt == 1:
            if subgraph_approach == "subgraph":
                prompt = n_question_from_subgraph_chain_without_example.get("prompt").format(
                    subgraph=subgraph_str, n=n)
                num_tokens = get_num_tokens(prompt)
                if num_tokens > 4097:
                    return None
                output = n_question_from_subgraph_chain_without_example.get("chain").run(
                    {"subgraph": subgraph_str, "n": n}
                )
            elif subgraph_approach == "summarized":
                prompt = n_question_from_summarized_subgraph_chain_without_example.get("prompt").format(subgraph=subgraph_str, n=n)
                num_tokens = get_num_tokens(prompt)
                if num_tokens > 4097:
                    return None
                output = n_question_from_summarized_subgraph_chain_without_example.get("chain").run(
                    {"subgraph": subgraph_str, "n": n}
                )
        elif prompt == 2:
            seed_entity = seed.label if seed.label else seed.uri
            output = n_question_from_subgraph_chain_using_seed_entity.get("chain").run(
                {"e": seed_entity, "subgraph": subgraph_str, "n": n}
            )
        elif prompt == 3:
            seed_entity = seed.label if seed.label else seed.uri
            seed_entity_type = defrag_uri(str(seed.nodetype))
            output = n_question_from_subgraph_chain_using_seed_entity_and_type.get("chain").run(
                {"e": seed_entity, "e_type": seed_entity_type, "subgraph": subgraph_str, "n": n}
            )
        output = output.dict()
    except Exception as e:
        response = str(e)
        if response.startswith("Failed to parse LLMInput from completion"):
            start_index = response.index('[')
            end_index = response.index('Got:')
            data = ast.literal_eval(response[start_index:end_index - 3])
            output = {"output": data}
    return output
