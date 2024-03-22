import sys
import time

sys.path.append('../')
import os
import re
import traceback
import random
import pathlib
import json
from dataclasses import asdict
from llm.prompt_chains import get_prompt_chains
from llm.llms import get_num_tokens, llms_dict
import langchain
from langchain.callbacks import get_openai_callback
from logger import logger
from analysis import analyze_benchmark_sample
from tracer import Tracer
from kg.kg.kg import defrag_uri
from answer_creation import get_LLM_based_postprocessed
from answer_validation import validate_query
from seed_node_extractor.seed_node_selector import SeedNodeSelector
from prepare_nodes_subgraph import retrieve_one_node_with_subgraph, retrieve_seed_nodes_with_subgraphs_new
import ast
import re
from appconfig import config
from errors import JsonParsingError, ContextLengthError

langchain.debug = True


questions_triple_llm = llms_dict["question_generation_model"]
questions_answer_llm = llms_dict["sparql_generation_model"]
questions_dialogue_llm = llms_dict["dialogue_generation_model"]

prompt_chains = get_prompt_chains()
n_question_from_subgraph_chain_without_example = prompt_chains.get("n_question_from_subgraph_chain_without_example")(questions_triple_llm)
n_question_from_summarized_subgraph_chain_without_example = prompt_chains.get("n_question_from_summarized_subgraph_chain_without_example")(questions_triple_llm)
n_question_from_subgraph_chain_using_seed_entity = prompt_chains.get("get_n_question_from_subgraph_chain_using_seed_entity")(questions_triple_llm)
n_question_from_subgraph_chain_using_seed_entity_and_type = prompt_chains.get("get_n_question_from_subgraph_chain_using_seed_entity_and_type")(questions_triple_llm)
pronoun_identification_and_substitution_chain = prompt_chains.get("get_pronoun_identification_and_substitution_chain_without_example")(questions_dialogue_llm)
get_triple_for_question_given_subgraph_chain_without_example = prompt_chains.get("get_triple_for_question_given_subgraph_chain_without_example")(questions_triple_llm)
n_question_from_summarized_subgraph_chain_without_example_without_triple = prompt_chains.get("n_question_from_summarized_subgraph_chain_without_example_without_triple")(questions_triple_llm)



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
        print("element-", element)
        try:
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
        except JsonParsingError:
            continue

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
    seed_nodes, seednode_to_subgraph_map, kg = retrieve_seed_nodes_with_subgraphs_new(kg_name, dataset_size, sampler, use_label)
    end = time.time()
    print(f"Seed node selection took {end - start} seconds")
    
    if "subgraph" in approach:
        exp_name = f"{kg_name}_e1_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}.json")
        tracer_instance = Tracer(os.path.join(out_dir, 'traces', f'{exp_name}.jsonl'))
        generate_dialogues_from_subgraph(seed_nodes, kg, tracer_instance, dialogue_size, output_file, prompt, sampler, seednode_to_subgraph_map)
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
    json_parsing_error = 0
    question_validation_error = 0
    triple_validation_error = 0
    dialogue_validation_error = 0
    failed_stage = {
        "question_triple": 0,
        "sparql_generation": 0,
    }
    for idx, seed in enumerate(seed_nodes):
        if idx>500:
            break
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
        output = None
        try:
            with get_openai_callback() as cb:
                logger.info(f"INDEX : {idx} -- question set generation chain start --")
                n = dialogue_size
                seed_entity = seed.label if seed.label else seed.uri
                valid_question = False
                valid_triples = False
                retry = 0
                while not (valid_question and valid_triples):
                    try:
                        output = execute_question_generation_prompt("subgraph", prompt, subgraph, n, seed)
                        valid_question = validate_questions_output(seed_entity, output)
                        valid_triples = validate_triples_output(subgraph, output, "subgraph")
                    except ContextLengthError:
                        context_length_limit_error += 1
                        break
                    except JsonParsingError:
                        json_parsing_error += 1
                        break

                    retry += 1
                    if retry == 3:
                        break

                if output is not None:
                    if not valid_question:
                        print("invalid question error")
                        question_validation_error += 1
                    elif not valid_triples:
                        print("invalid triple error")
                        triple_validation_error += 1
                    else:
                        question_set, answer_queries, triples_used, answer_status_dict = decouple_questions_and_answers(
                            output, subgraph, "subgraph", kg.sparql_endpoint, seed.uri)
                else:
                    failed_stage["question_triple"] += 1

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
                    if question_set and len(question_set) < 3:
                        failed_stage["sparql_generation"] += 1
                    # This is a trigger to sample the new node
                    question_set = None
        except Exception as e:
            traceback.print_exc()
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
            "seeds_used": idx,
            "data": benchmark_sample,
            "analysis" : benchmark_analysis,
            "total_time": total_time,
            "average_time": 0 if processed_seeds == 0 else (total_time / processed_seeds),
            "failed_stage": failed_stage,
            "Context Length Error": context_length_limit_error,
            "Json parsing Error": json_parsing_error,
            "Question Validation Error": question_validation_error,
            "Triples Validation Error": triple_validation_error,
            "Dialogue Validation Error": dialogue_validation_error,
        }
        directory = pathlib.Path(output_file).parent
        directory.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(benchmark, f, indent=4)

def validate_dialogue_output(seed, dialogue):
    if seed[-1] == '.':
        seed = seed[:-1]
    for question in dialogue:
        if seed.lower() in question.lower():
            return False
    return True

def validate_questions_output(seed, questions):
    if seed[-1] == '.':
        seed = seed[:-1]
    for question in questions["output"]:
        if seed.lower() not in question["question"].lower():
            return False
    return True

def validate_single_questions_output(seed, question):
    if seed[-1] == '.':
        seed = seed[:-1]
    if seed.lower() not in question.lower():
        return False
    return True

def validate_single_triples_output_v1(subgraph, triples, approach):
    if len(triples) > 0 and '(' not in triples[0] and ')' not in triples[0]:
        if len(triples) == 3:
            triples = [str(tuple(triples))]
        elif len(triples) == 2:
            triples = [str((triples[0], triples[1], ''))]

    for triple in triples:
        if not subgraph.contain_triple(triple, approach):
            return False
    return True

def validate_single_triples_output_v2(subgraph, triples, approach):
    triples_ = []
    for t in triples:
        if len(t) > 1:
            t_ = (t[0], t[1], '')
            triples_.append(t_)

    for t in triples_:
        if not subgraph.contain_triple(t, approach):
            return False
    return True

def validate_triples_output(subgraph, output, approach):
    if approach == "subgraph":
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

    else:
        for instance in output["output"]:
            triples = instance["triples"]
            if isinstance(triples, list) and isinstance(triples[0], str):
                triples = [triples]
            triples_ = []
            for t in triples:
                if len(t) > 1:
                    t_ = (t[0], t[1], '')
                    triples_.append(t_)
            instance["triples"] = triples_

            print("TRIPLES")
            for triple in triples_:
                print("t --> ", triple)
                if not subgraph.contain_triple(triple, approach):
                    return False
        return True



def execute_dialogue_generation_prompt(seed_entity, question_set):
    try:
        ch = pronoun_identification_and_substitution_chain.get("chain")
        post_processor = pronoun_identification_and_substitution_chain.get("post_processor")
        llm_result = ch.generate([{"entity": seed_entity, "questions": question_set[1:]}], None)
        output = post_processor(llm_result)
        transformed_questions = output["output"]
    except Exception as e:
        traceback.print_exc()
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
    json_parsing_error = 0
    question_validation_error = 0
    triple_validation_error = 0
    dialogue_validation_error = 0
    failed_stage = {
        "question_triple": 0,
        "sparql_generation": 0,
    }
    for idx, seed in enumerate(seed_nodes):
        if idx>500:
            break
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
                if config.pipeline_type == "original":
                    valid_question = False
                    valid_triples = False
                    retry = 0
                    while not (valid_question and valid_triples):
                        try:
                            output = execute_question_generation_prompt("summarized", prompt, subgraph, n, seed, config.pipeline_type)
                            valid_question = validate_questions_output(seed_entity, output)
                            valid_triples = validate_triples_output(subgraph, output, "optimized")
                        except ContextLengthError:
                            context_length_limit_error += 1
                            break
                        except JsonParsingError:
                            json_parsing_error += 1
                            break

                        retry += 1
                        if retry == 3:
                            break

                    if output is not None:
                        if not valid_question:
                            print("not valid question")
                            question_validation_error += 1
                        elif not valid_triples:
                            print("not valid triple")
                            triple_validation_error += 1
                        else:
                            question_set, answer_queries, triples_used, answer_status_dict = decouple_questions_and_answers(output, subgraph, "optimized", kg.sparql_endpoint, seed.uri)
                    else:
                        failed_stage["question_triple"] += 1
                
                elif config.pipeline_type == "simplified":
                    try:
                        output = execute_question_generation_prompt("summarized", prompt, subgraph, n, seed, config.pipeline_type)
                    except ContextLengthError:
                        context_length_limit_error += 1
                    except JsonParsingError:
                        json_parsing_error += 1

                    if output is not None:
                        valid_question = False if output.get("q_err",0) > 0 else True
                        valid_triples = False if output.get("t_err",0) > 0 else True
                        del output["q_err"]
                        del output["t_err"]
                        if not valid_question:
                            print("not valid question")
                            question_validation_error += 1
                        elif not valid_triples:
                            print("not valid triple")
                            triple_validation_error += 1
                        else:
                            question_set, answer_queries, triples_used, answer_status_dict = decouple_questions_and_answers(output, subgraph, "optimized", kg.sparql_endpoint, seed.uri)
                    else:
                        failed_stage["question_triple"] += 1
                        new_seed, new_subgraph = retrieve_one_node_with_subgraph(sampler, seed.nodetype, kg)
                        key = new_seed.label if new_seed.label else new_seed.uri
                        seed_nodes.append(new_seed)
                        seednode_to_subgraph_map[key] = new_subgraph
                        continue

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
                    if question_set and len(question_set) < 3:
                        failed_stage["sparql_generation"] += 1
                    # This is a trigger to sample the new node
                    question_set = None

        except Exception as e:
            traceback.print_exc()
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
            "seeds_used": idx,
            "data": benchmark_sample,
            "analysis" : benchmark_analysis,
            "total_time": total_time,
            "average_time": 0 if processed_seeds == 0 else (total_time / processed_seeds),
            "failed_stage": failed_stage,
            "Context Length Error": context_length_limit_error,
            "Json parsing Error": json_parsing_error,
            "Question Validation Error": question_validation_error,
            "Triples Validation Error": triple_validation_error,
            "Dialogue Validation Error": dialogue_validation_error,
        }
        directory = pathlib.Path(output_file).parent
        directory.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(benchmark, f, indent=4)


def execute_question_triple_binding_prompt(subgraph, subgraph_str, question_list, seed_entity):
    output_with_triple = []
    q_err_cnt = 0
    t_err_cnt = 0
    ch = get_triple_for_question_given_subgraph_chain_without_example.get("chain")
    post_processor = get_triple_for_question_given_subgraph_chain_without_example.get("post_processor")
    for q in question_list:
        try:
            prompt = get_triple_for_question_given_subgraph_chain_without_example.get("prompt").format(subgraph=subgraph_str, question=q)
            num_tokens = get_num_tokens(prompt)
            if num_tokens > 4097:
                return None
            llm_result = ch.generate([{"subgraph":subgraph_str, "question":q}], None)
            output = post_processor(llm_result) # output would be list of question

            valid_question = validate_single_questions_output(seed_entity, q)
            if not valid_question:
                q_err_cnt += 1
                continue
            valid_triples = validate_single_triples_output_v2(subgraph, output, "optimized")
            if not valid_triples:
                t_err_cnt += 1
                continue
            if valid_question and valid_triples:
                output_with_triple.append({"question":q, "triples":output})
        except Exception as e:
            print("Exception in triple binding, skipping question", e)
            continue
    return {"output": output_with_triple, "q_err": q_err_cnt, "t_err": t_err_cnt}

def execute_question_generation_prompt(subgraph_approach, prompt, subgraph, n, seed, pipeline_type="original"):
    output = None
    seed_entity = seed.label if seed.label else seed.uri
    try:
        if prompt == 1:
            if subgraph_approach == "subgraph":
                subgraph_str = subgraph.__str__(representation='uri')
                prompt = n_question_from_subgraph_chain_without_example.get("prompt").format(subgraph=subgraph_str, n=n)
                num_tokens = get_num_tokens(prompt)
                if num_tokens > 4097:
                    raise ContextLengthError()
                ch = n_question_from_subgraph_chain_without_example.get("chain")
                post_processor = n_question_from_subgraph_chain_without_example.get("post_processor")
                llm_result = ch.generate([{"subgraph": subgraph_str, "n": n}], None)
                output = post_processor(llm_result)
            elif subgraph_approach == "summarized":
                if pipeline_type == "original":
                    subgraph_str = subgraph.get_summarized_graph_str(approach='no_object')
                    # pdb.set_trace()
                    prompt = n_question_from_summarized_subgraph_chain_without_example.get("prompt").format(subgraph=subgraph_str, n=n)
                    num_tokens = get_num_tokens(prompt)
                    if num_tokens > 4097:
                        raise ContextLengthError()
                    ch = n_question_from_summarized_subgraph_chain_without_example.get("chain")
                    post_processor = n_question_from_summarized_subgraph_chain_without_example.get("post_processor")
                    llm_result = ch.generate([{"subgraph": subgraph_str, "n": n}], None)
                    output = post_processor(llm_result)
                elif pipeline_type == "simplified":
                    
                    subgraph_str = subgraph.get_summarized_graph_str(approach='no_object')
                    # pdb.set_trace()
                    prompt = n_question_from_summarized_subgraph_chain_without_example_without_triple.get("prompt").format(subgraph=subgraph_str, n=n)
                    num_tokens = get_num_tokens(prompt)
                    if num_tokens > 4097:
                        raise ContextLengthError()
                    ch = n_question_from_summarized_subgraph_chain_without_example_without_triple.get("chain")
                    post_processor = n_question_from_summarized_subgraph_chain_without_example_without_triple.get("post_processor")
                    llm_result = ch.generate([{"subgraph":subgraph_str, "n":n}], None)
                    output = post_processor(llm_result) # output would be list of question
                    question_list = output["output"]
                    output = execute_question_triple_binding_prompt(subgraph, subgraph_str, question_list,
                    seed_entity)
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
    except Exception as e:
        traceback.print_exc()
        response = str(e)
        if response.startswith("Failed to parse LLMInput from completion"):
            start_index = response.index('[')
            end_index = response.index('Got:')
            try:
                data = ast.literal_eval(response[start_index:end_index - 3])
                output = {"output": data}
            except Exception as e:
                raise JsonParsingError()
        else:
            raise JsonParsingError()
    return output
