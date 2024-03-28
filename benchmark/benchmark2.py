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

def decouple_questions_and_answers(que_trip_set, subgraph, approach, endpoint, seed_node_uri):
    questions = list()
    answer_queries = list()
    triples = list()
    query_results = dict()
    sparql_json_parse_err_cnt = 0
    sparql_validation_err_cnt = 0
    for element in que_trip_set:
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
            else:
                sparql_validation_err_cnt += 1
        except JsonParsingError:
            sparql_json_parse_err_cnt += 1
            continue
        except Exception as e:
            continue

    if len(answer_queries) < 3:
        # not all sparql binding were correct or there was json parsing error
        questions = None
        answer_queries = None
        triples = None
        query_results = None
        if sparql_json_parse_err_cnt >= sparql_validation_err_cnt:
            sparql_json_parse_err_cnt = 1
            sparql_validation_err_cnt = 0
        else:
            sparql_validation_err_cnt = 1
            sparql_json_parse_err_cnt = 0
    else:
        sparql_json_parse_err_cnt = 0
        sparql_validation_err_cnt = 0
    

    return questions, answer_queries, triples, query_results, sparql_json_parse_err_cnt, sparql_validation_err_cnt



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
        output_file = os.path.join(out_dir, f"{exp_name}_{config.pipeline_type}.json")
        tracer_instance = Tracer(os.path.join(out_dir, 'traces', f'{exp_name}.jsonl'))
        generate_dialogues_from_subgraph(seed_nodes, kg, tracer_instance, dialogue_size, output_file, prompt, sampler, seednode_to_subgraph_map)
    if "subgraph-summarized" in approach:
        exp_name = f"{kg_name}_e11_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}_{config.pipeline_type}.json")
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
    question_json_parsing_error = 0
    triple_validation_error = 0
    triple_json_parsing_error = 0
    sparql_json_parsing_error = 0
    sparql_validation_error = 0
    dialogue_json_parsing_error = 0
    dialogue_validation_error = 0
    for idx, seed in enumerate(seed_nodes):
        if idx>500:
            break
        start_time = time.time()
        key = seed.label if seed.label else seed.uri
        subgraph = seednode_to_subgraph_map[key]

        dialogue = None
        question_set_dialogue = None
        question_set = None
        cb_dict = None
        answer_queries = None
        triples_used = None
        answer_status_dict = None
        errors = {}
        skip_node = False
        que_trip_set = None
        sp_json_err = 0
        sp_val_err = 0
        try:
            with get_openai_callback() as cb:
                n = dialogue_size
                seed_label = seed.label if seed.label else defrag_uri(str(seed.uri))
                errors = {}
                que_trip_set, errors = execute_question_generation_prompt("subgraph", prompt, subgraph, n, seed, config.pipeline_type)

                if que_trip_set is None:
                    # question triple set was none so just count the errors and continue to next seed
                    context_length_limit_error += errors.get("context_length_error", 0)
                    question_json_parsing_error += errors.get("question_json_parsing_error", 0)
                    question_validation_error += errors.get("question_validation_error", 0)
                    triple_json_parsing_error += errors.get("triple_json_parsing_error", 0) 
                    triple_validation_error += errors.get("triple_validation_error", 0)
                    skip_node = True
                else:
                    # question triple stage was success move to answer query generation
                    question_set, answer_queries, triples_used, answer_status_dict, sp_json_err, sp_val_err = decouple_questions_and_answers(que_trip_set, subgraph, "subgraph", kg.sparql_endpoint, seed.uri)

                    if question_set is None:
                        sparql_json_parsing_error += sp_json_err
                        sparql_validation_error += sp_val_err
                        skip_node = True
                    else:
                        errors = {}
                        question_set_dialogue, errors = execute_dialogue_generation_prompt(seed, question_set)

                        if question_set_dialogue is None:
                            dialogue_json_parsing_error += errors.get("dialogue_json_parsing_error", 0)
                            dialogue_validation_error += errors.get("dialogue_validation_error", 0)
                            skip_node = True
                        else:
                            cb_dict = {
                                "total_tokens": cb.total_tokens,
                                "prompt_tokens": cb.prompt_tokens,
                                "completion_tokens": cb.completion_tokens,
                                # "successful_requests": cb.successful_requests,
                                # "total_cost": cb.total_cost
                            }
                            # save things here
                            dialogue = {
                                "seed_entity": str(seed.uri),
                                "seed_label": seed_label,
                                "dialogue": question_set_dialogue,
                                "original": question_set,
                                "queries": answer_queries,
                                "triples": triples_used,
                                "cost": cb_dict,
                                "query_status": answer_status_dict
                            }
                            print(dialogue)
                            benchmark_sample.append(dialogue)
                            end_time = time.time()
                            total_time += (end_time - start_time)
                            processed_seeds += 1

        except Exception as e:
            traceback.print_exc()
            logger.info(f"INDEX : {idx} -- ERROR: {idx} : {e} --")
            tracer_instance.save_to_file()

        # if question_set is None or question_set_dialogue is None or len(question_set) != len(question_set_dialogue):
        if skip_node == True:
            # Sample a new node and add it to seed nodes
            new_seed, subgraph = retrieve_one_node_with_subgraph(sampler, seed.nodetype, kg)
            key = new_seed.label if new_seed.label else new_seed.uri
            seed_nodes.append(new_seed)
            seednode_to_subgraph_map[key] = subgraph
        
        benchmark_analysis = analyze_benchmark_sample(benchmark_sample)
        benchmark = {
            "seeds_used": idx+1,
            "data": benchmark_sample,
            "analysis" : benchmark_analysis,
            "total_time": total_time,
            "average_time": 0 if processed_seeds == 0 else (total_time / processed_seeds),
            "Context Length Error": context_length_limit_error,
            "Question Validation Error": question_validation_error,
            "Question Json Error": question_json_parsing_error,
            "Triples Validation Error": triple_validation_error,
            "Triples Json Error": triple_json_parsing_error,
            "Sparql Validation Error": sparql_validation_error,
            "Sparql Json Error": sparql_json_parsing_error,
            "Dialogue Validation Error": dialogue_validation_error,
            "Dialogue Json Error": dialogue_json_parsing_error,
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
                if "(" in triples[0] and  ')' in triples[0]:
                    triples = triples[0].replace("(","").replace(")", "").replace("'","").split(', ')
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



def execute_dialogue_generation_prompt(seed, question_set):
    seed_entity = seed.label if seed.label else seed.uri
    seed_label = seed.label if seed.label else defrag_uri(str(seed.uri))
    transformed_questions = []
    errors = {
        "dialogue_json_parsing_error": 0,
        "dialogue_validation_error": 0
    }
    ch = pronoun_identification_and_substitution_chain.get("chain")
    post_processor = pronoun_identification_and_substitution_chain.get("post_processor")
    diag_json_parsing_err = False
    diag_validation_err = False
    valid = False
    retry = 0
    while not valid:
        if retry >= 3:
            break
        diag_json_parsing_err = False
        diag_validation_err = False
        try:
            llm_result = ch.generate([{"entity": seed_entity, "questions": question_set[1:]}], None)
            output = post_processor(llm_result)
            transformed_questions = output["output"]
            valid = validate_dialogue_output(seed_label, transformed_questions)
            if valid:
                diag_validation_err = False
                break
            else:
                diag_validation_err = True
        except Exception as e:
            response = str(e)
            start_index = response.index('[')
            end_index = response.index(']')
            if response.startswith("Failed to parse"):
                try:
                    transformed_questions = ast.literal_eval(response[start_index:end_index + 1])
                    valid = validate_dialogue_output(seed_label, transformed_questions)
                    if valid:
                        diag_validation_err = False
                        break
                    else:
                        diag_validation_err = True
                except Exception as e:
                    diag_json_parsing_err = True
            else:
                # json parsing error
                diag_json_parsing_err = True
        retry += 1
    
    if diag_json_parsing_err:
        question_set_dialogue = None
        errors["dialogue_json_parsing_error"] = 1
    elif diag_validation_err:
        question_set_dialogue = None
        errors["dialogue_validation_error"] = 1
    else:
        # no error
        question_0 = question_set[0]
        question_set_dialogue = [question_0, *transformed_questions]

    return question_set_dialogue, errors


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
    question_json_parsing_error = 0
    triple_validation_error = 0
    triple_json_parsing_error = 0
    sparql_json_parsing_error = 0
    sparql_validation_error = 0
    dialogue_json_parsing_error = 0
    dialogue_validation_error = 0
    for idx, seed in enumerate(seed_nodes):
        if idx>500:
            break
        start_time = time.time()
        key = seed.label if seed.label else seed.uri
        subgraph = seednode_to_subgraph_map[key]

        dialogue = None
        question_set_dialogue = None
        question_set = None
        cb_dict = None
        answer_queries = None
        triples_used = None
        answer_status_dict = None
        errors = {}
        skip_node = False
        que_trip_set = None
        sp_json_err = 0
        sp_val_err = 0
        try:
            with get_openai_callback() as cb:
                n = dialogue_size
                seed_label = seed.label if seed.label else defrag_uri(str(seed.uri))
                
                errors = {}
                que_trip_set, errors = execute_question_generation_prompt("summarized", prompt, subgraph, n, seed, config.pipeline_type)
                
                if que_trip_set is None:
                    # question triple set was none so just count the errors and continue to next seed
                    context_length_limit_error += errors.get("context_length_error", 0)
                    question_json_parsing_error += errors.get("question_json_parsing_error", 0)
                    question_validation_error += errors.get("question_validation_error", 0)
                    triple_json_parsing_error += errors.get("triple_json_parsing_error", 0) 
                    triple_validation_error += errors.get("triple_validation_error", 0)
                    skip_node = True
                else:
                    # question triple stage was success move to answer query generation
                    question_set, answer_queries, triples_used, answer_status_dict, sp_json_err, sp_val_err = decouple_questions_and_answers(que_trip_set, subgraph, "optimized", kg.sparql_endpoint, seed.uri)

                    if question_set is None:
                        sparql_json_parsing_error += sp_json_err
                        sparql_validation_error += sp_val_err
                        skip_node = True
                    else:
                        errors = {}
                        question_set_dialogue, errors = execute_dialogue_generation_prompt(seed, question_set)

                        if question_set_dialogue is None:
                            dialogue_json_parsing_error += errors.get("dialogue_json_parsing_error", 0)
                            dialogue_validation_error += errors.get("dialogue_validation_error", 0)
                            skip_node = True
                        else:
                            cb_dict = {
                                "total_tokens": cb.total_tokens,
                                "prompt_tokens": cb.prompt_tokens,
                                "completion_tokens": cb.completion_tokens,
                                # "successful_requests": cb.successful_requests,
                                # "total_cost": cb.total_cost
                            }
                            # save things here
                            dialogue = {
                                "seed_entity": str(seed.uri),
                                "seed_label": seed_label,
                                "dialogue": question_set_dialogue,
                                "original": question_set,
                                "queries": answer_queries,
                                "triples": triples_used,
                                "cost": cb_dict,
                                "query_status": answer_status_dict
                            }
                            print(dialogue)
                            benchmark_sample.append(dialogue)
                            end_time = time.time()
                            total_time += (end_time - start_time)
                            processed_seeds += 1
        except Exception as e:
            ## TODO: this error was not recovered add logic
            traceback.print_exc()
            logger.info(f"INDEX : {idx} -- ERROR: {idx} : {e} --")
            tracer_instance.save_to_file()
        
        if skip_node == True:
            # Sample a new node and add it to seed nodes
            new_seed, subgraph = retrieve_one_node_with_subgraph(sampler, seed.nodetype, kg)
            key = new_seed.label if new_seed.label else new_seed.uri
            seed_nodes.append(new_seed)
            seednode_to_subgraph_map[key] = subgraph

        benchmark_analysis = analyze_benchmark_sample(benchmark_sample)
        benchmark = {
            "seeds_used": idx+1,
            "data": benchmark_sample,
            "analysis" : benchmark_analysis,
            "total_time": total_time,
            "average_time": 0 if processed_seeds == 0 else (total_time / processed_seeds),
            "Context Length Error": context_length_limit_error,
            "Question Validation Error": question_validation_error,
            "Question Json Error": question_json_parsing_error,
            "Triples Validation Error": triple_validation_error,
            "Triples Json Error": triple_json_parsing_error,
            "Sparql Validation Error": sparql_validation_error,
            "Sparql Json Error": sparql_json_parsing_error,
            "Dialogue Validation Error": dialogue_validation_error,
            "Dialogue Json Error": dialogue_json_parsing_error,
        }
        directory = pathlib.Path(output_file).parent
        directory.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(benchmark, f, indent=4)


def execute_question_triple_binding_prompt(subgraph, subgraph_str, question_list, seed_entity):
    errors = {
        "triple_validation_error": 0,
        "triple_json_parsing_error": 0,
    }
    output_with_triple = []
    t_validation_err_cnt = 0
    t_json_parse_err_cnt = 0
    ch = get_triple_for_question_given_subgraph_chain_without_example.get("chain")
    post_processor = get_triple_for_question_given_subgraph_chain_without_example.get("post_processor")
    for q in question_list:
        try:
            llm_result = ch.generate([{"subgraph":subgraph_str, "question":q}], None)
            output = post_processor(llm_result) # output would be list of question

            valid_triples = validate_single_triples_output_v2(subgraph, output, "optimized")
            if not valid_triples:
                t_validation_err_cnt += 1
                continue
            else:
                output_with_triple.append({"question":q, "triples":output})
        except Exception as e:
            t_json_parse_err_cnt += 1
            print("Exception in triple binding, skipping question", e)
            continue
    if len(question_list) != len(output_with_triple):
        # not all triple binding were correct
        if t_json_parse_err_cnt >= t_validation_err_cnt:
            errors["triple_json_parsing_error"] = 1
            return None, errors
        else:
            errors["triple_validation_error"] = 1
            return None, errors
    else:
        return output_with_triple, errors

def execute_question_generation_prompt(subgraph_approach, prompt, subgraph, n, seed, pipeline_type="original"):
    output = None
    if subgraph_approach == "subgraph":
        # all error types
        errors = {
            "context_length_error": 0,
            "question_validation_error": 0,
            "question_json_parsing_error": 0,
            "triple_validation_error": 0,
            "triple_json_parsing_error": 0,
        }

        seed_label = seed.label if seed.label else defrag_uri(str(seed.uri))
        subgraph_str = subgraph.__str__(representation='uri')
        prompt = n_question_from_subgraph_chain_without_example.get("prompt").format(subgraph=subgraph_str, n=n)
        num_tokens = get_num_tokens(prompt)
        if num_tokens > 4097:
            errors["context_length_error"] = 1
            return None, errors

        ch = n_question_from_subgraph_chain_without_example.get("chain")
        post_processor = n_question_from_subgraph_chain_without_example.get("post_processor")

        que_trip_set = None
        valid_question = False
        valid_triples = False
        retry = 0
        question_json_parsing_error = False
        while not (valid_question and valid_triples):
            question_json_parsing_error = False
            try:
                llm_result = ch.generate([{"subgraph": subgraph_str, "n": n}], None)
                output = post_processor(llm_result)
                valid_question = validate_questions_output(seed_label, output)
                valid_triples = validate_triples_output(subgraph, output, "subgraph")
            except Exception as e:
                response = str(e)
                if response.startswith("Failed to parse"):
                    start_index = response.index('[')
                    end_index = response.index('Got:')
                    try:
                        data = ast.literal_eval(response[start_index:end_index - 3])
                        output = {"output": data}
                        valid_question = validate_questions_output(seed_label, output)
                        valid_triples = validate_triples_output(subgraph, output, "subgraph")
                    except Exception as e:
                        # json parsing error
                        question_json_parsing_error = True
                else:
                    # json parsing error
                    question_json_parsing_error = True

            retry += 1
            if retry == 3:
                break

        
        if question_json_parsing_error:
            # json parsing error in last attempt skip the node and count it as a q_json_parsing_error
            que_trip_set = None
            errors["question_json_parsing_error"] = 1
        else:
            if not valid_question:
                que_trip_set = None
                errors["question_validation_error"] = 1
            elif not valid_triples:
                que_trip_set = None
                errors["triple_validation_error"] = 1
            else:
                # no error
                que_trip_set = output["output"]

        return que_trip_set, errors


    elif subgraph_approach == "summarized":
        # all error types
        errors = {
            "context_length_error": 0,
            "question_validation_error": 0,
            "question_json_parsing_error": 0,
            "triple_validation_error": 0,
            "triple_json_parsing_error": 0,
        }
        if pipeline_type == "original":

            seed_label = seed.label if seed.label else defrag_uri(str(seed.uri))
            subgraph_str = subgraph.get_summarized_graph_str(approach='no_object')

            prompt = n_question_from_summarized_subgraph_chain_without_example.get("prompt").format(subgraph=subgraph_str, n=n)
            num_tokens = get_num_tokens(prompt)
            if num_tokens > 4097:
                errors["context_length_error"] = 1
                return None, errors

            ch = n_question_from_summarized_subgraph_chain_without_example.get("chain")
            post_processor = n_question_from_summarized_subgraph_chain_without_example.get("post_processor")

            que_trip_set = None
            valid_question = False
            valid_triples = False
            retry = 0
            question_json_parsing_error = False
            while not (valid_question and valid_triples):
                question_json_parsing_error = False
                try:
                    llm_result = ch.generate([{"subgraph": subgraph_str, "n": n}], None)
                    output = post_processor(llm_result)
                    valid_question = validate_questions_output(seed_label, output)
                    valid_triples = validate_triples_output(subgraph, output, "optimized")
                except Exception as e:
                    response = str(e)
                    if response.startswith("Failed to parse"):
                        start_index = response.index('[')
                        end_index = response.index('Got:')
                        try:
                            data = ast.literal_eval(response[start_index:end_index - 3])
                            output = {"output": data}
                            valid_question = validate_questions_output(seed_label, output)
                            valid_triples = validate_triples_output(subgraph, output, "optimized")
                        except Exception as e:
                            # json parsing error
                            question_json_parsing_error = True
                    else:
                        # json parsing error
                        question_json_parsing_error = True

                retry += 1
                if retry == 3:
                    break

            
            if question_json_parsing_error:
                # json parsing error in last attempt skip the node and count it as a q_json_parsing_error
                que_trip_set = None
                errors["question_json_parsing_error"] = 1
            else:
                if not valid_question:
                    que_trip_set = None
                    errors["question_validation_error"] = 1
                elif not valid_triples:
                    que_trip_set = None
                    errors["triple_validation_error"] = 1
                else:
                    # no error
                    que_trip_set = output["output"]

            return que_trip_set, errors

        elif pipeline_type == "simplified":
            seed_entity_representation = seed.label if seed.label else seed.uri
            subgraph_str = subgraph.get_summarized_graph_str(approach='no_object')
            prompt = n_question_from_summarized_subgraph_chain_without_example_without_triple.get("prompt").format(subgraph=subgraph_str, n=n)
            num_tokens = get_num_tokens(prompt)
            if num_tokens > 4097:
                errors["context_length_error"] = 1
                return None, errors
            

            ch = n_question_from_summarized_subgraph_chain_without_example_without_triple.get("chain")
            post_processor = n_question_from_summarized_subgraph_chain_without_example_without_triple.get("post_processor")

            que_trip_set = None
            retries = 0
            question_json_parsing_error = None
            question_validation_error = None
            question_list = None
            while True:
                if retries >= 3:
                    break

                question_json_parsing_error = False
                question_validation_error = False
                output = None
                try:
                    llm_result = ch.generate([{"subgraph":subgraph_str, "n":n}], None)
                    output = post_processor(llm_result) # output would be list of question
                    question_list = output["output"]
                    q_output = {"output": [{"question": q} for q in question_list]}
                    valid_question = validate_questions_output(seed_entity_representation, q_output)
                    if valid_question:
                        question_validation_error = False
                        break
                    else:
                        question_validation_error = True
                except Exception as e:
                    response = str(e)
                    if response.startswith("Failed to parse"):
                        start_index = response.index('[')
                        end_index = response.index('Got:')
                        try:
                            data = ast.literal_eval(response[start_index:end_index - 3])
                            output = {"output": data}
                            valid_question = validate_questions_output(seed_entity_representation, output)
                            if valid_question:
                                question_validation_error = False
                                break
                            else:
                                question_validation_error = True
                        except Exception as e:
                            # json parsing error
                            question_json_parsing_error = True
                    else:
                        # json parsing error
                        question_json_parsing_error = True

                retries += 1
            
            if question_json_parsing_error:
                # json parsing error in last attempt skip the node and count it as a q_json_parsing_error
                que_trip_set = None
                errors["question_json_parsing_error"] = 1
            elif question_validation_error:
                que_trip_set = None
                errors["question_validation_error"] = 1
            else:
                # no error
                # move to triple binding step
                que_trip_set, t_errors = execute_question_triple_binding_prompt(subgraph, subgraph_str, question_list, seed_entity_representation)
                errors["triple_json_parsing_error"] = t_errors["triple_json_parsing_error"]
                errors["triple_validation_error"] = t_errors["triple_validation_error"]
            
            return que_trip_set, errors
