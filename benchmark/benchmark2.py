import sys
import time

sys.path.append("../")
import os
import traceback
import random
import pathlib
import json
from llm.prompt_chains import get_prompt_chains
from llm.llms import get_num_tokens, llms_dict
import langchain
from logger import logger
from analysis import analyze_benchmark_sample
from kg.kg.kg import defrag_uri
from answer_creation import get_LLM_based_postprocessed
from answer_validation import validate_query, validate_query_v2
from benchmark.seed_node_extractor.seed_node_selector import SeedNodeSelector
from prepare_nodes_subgraph import (
    retrieve_one_node_with_subgraph,
    retrieve_seed_nodes_with_subgraphs_new,
)
import ast
from appconfig import config
from errors import JsonParsingError
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook
from wandb_utils import Trace, StatusCode


# wandb trigger sync
trigger_sync = TriggerWandbSyncHook()

# initialize wandb job run
run_name = f"{config.kgname}-{config.pipeline_type}-{config.used_llms()}-{config.dataset_size}"
wandb.init(
    name=run_name,
    project=config.wandb_project,
    mode=config.wandb_mode,
    config=config.config_for_wandb(),
)

langchain.debug = True

llm_callback = llms_dict.get("question_generation_model", {}).get("llm_callback", None)
if llm_callback is None:
    raise ValueError("Invalid llm callback")

questions_triple_llm = llms_dict.get("question_generation_model")
questions_answer_llm = llms_dict.get("sparql_generation_model")
questions_dialogue_llm = llms_dict.get("dialogue_generation_model")

prompt_chains = get_prompt_chains()
n_question_from_subgraph_chain_without_example = prompt_chains.get(
    "n_question_from_subgraph_chain_without_example"
)(questions_triple_llm)
n_question_from_summarized_subgraph_chain_without_example = prompt_chains.get(
    "n_question_from_summarized_subgraph_chain_without_example"
)(questions_triple_llm)
n_question_from_subgraph_chain_using_seed_entity = prompt_chains.get(
    "get_n_question_from_subgraph_chain_using_seed_entity"
)(questions_triple_llm)
n_question_from_subgraph_chain_using_seed_entity_and_type = prompt_chains.get(
    "get_n_question_from_subgraph_chain_using_seed_entity_and_type"
)(questions_triple_llm)
pronoun_identification_and_substitution_chain = prompt_chains.get(
    "get_pronoun_identification_and_substitution_chain_without_example"
)(questions_dialogue_llm)
get_triple_for_question_given_subgraph_chain_without_example = prompt_chains.get(
    "get_triple_for_question_given_subgraph_chain_without_example"
)(questions_triple_llm)
n_question_from_summarized_subgraph_chain_without_example_without_triple = (
    prompt_chains.get(
        "n_question_from_summarized_subgraph_chain_without_example_without_triple"
    )(questions_triple_llm)
)
singleshot_dialogue_chain = (
    prompt_chains.get(
        "singleshot_dialogue_chain"
    )(questions_triple_llm)
)

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
    for _, questions in categorized_questions.items():
        selected_question = random.choice(questions)
        selected_questions.append(selected_question)

    return selected_questions


def decouple_questions_and_answers(
    que_trip_set, subgraph, approach, endpoint, seed_node_uri, parent_trace
):
    questions = list()
    answer_queries = list()
    triples = list()
    query_results = dict()
    sparql_json_parse_err_cnt = 0
    sparql_validation_err_cnt = 0
    s_chain_input = ({"inputs": que_trip_set},)
    s_chain_trace = Trace(
        name="Answer Generation",
        kind="CHAIN",
        start_time_ms = time.time_ns() // 1000,
        model_dict={
            "_model": config.sparql_generation_model.model_name,
            "_kind": config.sparql_generation_model.model_type,
        },
    )
    for element in que_trip_set:
        print("element-", element)
        q_chain_trace = None
        try:
            q_chain_input = {
                "question": element["question"],
                "triples": element["triples"],
            }
            q_chain_trace = Trace(
                name="Sparql-Quey-Generation",
                kind="LLM",
                start_time_ms = time.time_ns() // 1000,
                model_dict={
                    "_model": config.sparql_generation_model.model_name,
                    "_kind": config.sparql_generation_model.model_type,
                },
            )

            triples_list = list()
            for triple in element["triples"]:
                if approach == "optimized":
                    triple = subgraph.get_triple_with_uris_no_object(triple) # optimized
                subject, predicate, object = triple
                triples_list.append((subject.__str__(), predicate.__str__(), object.__str__()))

            answer_query = get_LLM_based_postprocessed(
                element["question"],
                triples_list,
                subgraph,
                approach,
                q_chain_input,
                q_chain_trace,
            )
            q_chain_trace.outputs = {"llm_query": answer_query}
            status = validate_query(
                answer_query,
                triples_list,
                endpoint,
                subgraph,
                seed_node_uri,
                approach,
            )
            if status in query_results:
                query_results[status] += 1
            else:
                query_results[status] = 1

            if status == "Correct":
                questions.append(element["question"])
                answer_queries.append(answer_query)
                triples.append(triples_list)
                q_chain_trace.status_code = StatusCode.SUCCESS.name
            else:
                sparql_validation_err_cnt += 1
                q_chain_trace.status_code = StatusCode.SPARQL_VALIDATION_ERROR.name
                q_chain_trace.status_message = f"query validation status was: {status}"

            q_chain_trace._span.end_time_ms = time.time_ns() // 1000
            s_chain_trace.add_child(q_chain_trace)
        except JsonParsingError:
            traceback.print_exc()
            sparql_json_parse_err_cnt += 1
            q_chain_trace.status_code = StatusCode.SPARQL_JSON_ERROR
            q_chain_trace._span.end_time_ms = time.time_ns() // 1000
            s_chain_trace.add_child(q_chain_trace)
            continue
        except Exception as e:
            traceback.print_exc()
            q_chain_trace.status_code = StatusCode.ERROR
            q_chain_trace.status_message = "unknown error occurred"
            q_chain_trace._span.end_time_ms = time.time_ns() // 1000
            s_chain_trace.add_child(q_chain_trace)
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
            s_chain_trace.status_code = StatusCode.SPARQL_JSON_ERROR.name
            s_chain_trace._span.end_time_ms = time.time_ns() // 1000
        else:
            sparql_validation_err_cnt = 1
            sparql_json_parse_err_cnt = 0
            s_chain_trace.status_code = StatusCode.SPARQL_VALIDATION_ERROR.name
            s_chain_trace._span.end_time_ms = time.time_ns() // 1000
    else:
        sparql_json_parse_err_cnt = 0
        sparql_validation_err_cnt = 0
        s_chain_trace.status_code = StatusCode.SUCCESS.name
        s_chain_trace._span.end_time_ms = time.time_ns() // 1000

    s_chain_output = {"answer_queries": answer_queries}
    s_chain_trace.add_inputs_and_outputs(inputs=s_chain_input, outputs=s_chain_output)
    parent_trace.add_child(s_chain_trace)

    return (
        questions,
        answer_queries,
        triples,
        query_results,
        sparql_json_parse_err_cnt,
        sparql_validation_err_cnt,
    )


def generate_dialogues(
    kg_name,
    dataset_size=2,
    dialogue_size=2,
    approach=["subgraph"],
    out_dir="./results",
    prompt=1,
    use_label=True,
    seed_nodes_file=None,
):
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
    seed_nodes, seednode_to_subgraph_map, kg = retrieve_seed_nodes_with_subgraphs_new(
        kg_name, dataset_size, sampler, use_label
    )
    end = time.time()
    print(f"Seed node selection took {end - start} seconds")

    if "subgraph" in approach:
        exp_name = f"{kg_name}_subgraph_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}_{config.pipeline_type}.json")
        generate_dialogues_from_subgraph(
            seed_nodes,
            kg,
            dialogue_size,
            output_file,
            prompt,
            sampler,
            seednode_to_subgraph_map,
        )
    if "subgraph-summarized" in approach:
        exp_name = f"{kg_name}_subgraph_summarized_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}_{config.pipeline_type}.json")
        generate_dialogues_from_summarized_subgraph(
            seed_nodes,
            kg,
            dialogue_size,
            output_file,
            prompt,
            sampler,
            seednode_to_subgraph_map,
        )
    if "single-shot" in approach:
        exp_name = f"{kg_name}_single_shot_{dataset_size}_{dialogue_size}"
        output_file = os.path.join(out_dir, f"{exp_name}_{config.pipeline_type}.json")
        generate_dialogues_from_singleshot(
            seed_nodes,
            kg,
            dialogue_size,
            output_file,
            prompt,
            sampler,
            seednode_to_subgraph_map,
        )


def generate_dialogues_from_subgraph(
    initial_seed_nodes,
    kg,
    dialogue_size,
    output_file,
    prompt,
    sampler,
    seednode_to_subgraph_map,
):

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

    cost = { "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for idx, seed in enumerate(seed_nodes):
        if idx > 500:
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
        n = dialogue_size
        seed_label = seed.label if seed.label else defrag_uri(str(seed.uri))

        parent_trace_inputs = {
            "seed_uri": str(seed.uri),
            "subgraph": subgraph.__str__(representation="uri"),
            "seed_label": seed_label,
        }
        parent_trace_outputs = None
        parent_trace = Trace(
            name="Pipeline",
            kind="PIPELINE",
            start_time_ms = time.time_ns() // 1000,
        )
        with llm_callback() as cb:
            try:
                errors = {}
                que_trip_set, errors = execute_question_generation_prompt(
                    "subgraph",
                    prompt,
                    subgraph,
                    n,
                    seed,
                    config.pipeline_type,
                    parent_trace=parent_trace,
                )

                if que_trip_set is None:
                    # question triple set was none so just count the errors and continue to next seed
                    context_length_limit_error += errors.get("context_length_error", 0)
                    question_json_parsing_error += errors.get(
                        "question_json_parsing_error", 0
                    )
                    question_validation_error += errors.get(
                        "question_validation_error", 0
                    )
                    triple_json_parsing_error += errors.get(
                        "triple_json_parsing_error", 0
                    )
                    triple_validation_error += errors.get("triple_validation_error", 0)
                    skip_node = True
                else:
                    # question triple stage was success move to answer query generation
                    (
                        question_set,
                        answer_queries,
                        triples_used,
                        answer_status_dict,
                        sp_json_err,
                        sp_val_err,
                    ) = decouple_questions_and_answers(
                        que_trip_set,
                        subgraph,
                        "subgraph",
                        kg.sparql_endpoint,
                        seed.uri,
                        parent_trace=parent_trace,
                    )

                    if question_set is None:
                        sparql_json_parsing_error += sp_json_err
                        sparql_validation_error += sp_val_err
                        skip_node = True
                    else:
                        errors = {}
                        question_set_dialogue, errors = (
                            execute_dialogue_generation_prompt(
                                seed, question_set, parent_trace=parent_trace
                            )
                        )

                        if question_set_dialogue is None:
                            dialogue_json_parsing_error += errors.get(
                                "dialogue_json_parsing_error", 0
                            )
                            dialogue_validation_error += errors.get(
                                "dialogue_validation_error", 0
                            )
                            skip_node = True
                        else:
                            cb_dict = {
                                "total_tokens": cb.prompt_tokens + cb.completion_tokens,
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
                                "query_status": answer_status_dict,
                            }
                            parent_trace.status_code = StatusCode.SUCCESS.name
                            parent_trace_outputs = {
                                "dialogue": question_set_dialogue,
                                "original": question_set,
                                "queries": answer_queries,
                                "query_status": answer_status_dict,
                            }
                            print(dialogue)
                            benchmark_sample.append(dialogue)
                            end_time = time.time()
                            total_time += end_time - start_time
                            processed_seeds += 1

            except Exception as e:
                traceback.print_exc()
                logger.info(f"INDEX : {idx} -- ERROR: {idx} : {e} --")
                parent_trace.status_code = StatusCode.ERROR
                parent_trace.status_message = "unknown error occurred"
                parent_trace._span.end_time_ms = time.time_ns() // 1000

            cost["total_tokens"] += (cb.prompt_tokens + cb.completion_tokens)
            cost["prompt_tokens"] += cb.prompt_tokens
            cost["completion_tokens"] += cb.completion_tokens

        # if question_set is None or question_set_dialogue is None or len(question_set) != len(question_set_dialogue):
        if skip_node is True:
            node_added = False
            while not node_added:
                try:
                    # Sample a new node and add it to seed nodes
                    new_seed, subgraph = retrieve_one_node_with_subgraph(
                        sampler, seed.nodetype, kg
                    )
                    key = new_seed.label if new_seed.label else new_seed.uri
                    seed_nodes.append(new_seed)
                    seednode_to_subgraph_map[key] = subgraph
                    node_added = True
                except Exception as e:
                    print("Exception ", e)
                    continue



        if parent_trace is not None:
            print("####### trace saved ########")
            parent_trace.add_inputs_and_outputs(
                inputs=parent_trace_inputs, outputs=parent_trace_outputs
            )
            parent_trace._span.end_time_ms = time.time_ns() // 1000
            parent_trace.log("diag-gen-pipeline")

            # sync with wandb online
            trigger_sync()
            # break

        benchmark_analysis = analyze_benchmark_sample(benchmark_sample)
        benchmark = {
            "seeds_used": idx + 1,
            "data": benchmark_sample,
            "analysis": benchmark_analysis,
            "cost": cost,
            "total_time": total_time,
            "average_time": (
                0 if processed_seeds == 0 else (total_time / processed_seeds)
            ),
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
    if seed[-1] == ".":
        seed = seed[:-1]
    for question in dialogue:
        if seed.lower() in question.lower() or len(question.split(" ")) <= 1 or "?" not in question.lower():
            return False
    return True

def validate_singleshot_dialogue_output(seed, dialogue):
    if seed[-1] == ".":
        seed = seed[:-1]
    q0 = dialogue[0]
    valid = validate_singleshot_questions_output(seed, [q0])
    if not valid:
        return False
    for question in dialogue[1:]:
        if seed.lower() in question.lower() or len(question.split(" ")) <= 1 or "?" not in question.lower():
            return False
    return True

def validate_questions_output(seed, questions):
    if seed[-1] == ".":
        seed = seed[:-1]
    for question in questions["output"]:
        if seed.lower() not in question["question"].lower():
            return False
    return True

def validate_singleshot_questions_output(seed, questions):
    if seed[-1] == ".":
        seed = seed[:-1]
    for question in questions:
        if seed.lower() not in question.lower():
            return False
    return True

def validate_single_questions_output(seed, question):
    if seed[-1] == ".":
        seed = seed[:-1]
    if seed.lower() not in question.lower():
        return False
    return True


def validate_single_triples_output_v1(subgraph, triples, approach):
    if len(triples) > 0 and "(" not in triples[0] and ")" not in triples[0]:
        if len(triples) == 3:
            triples = [str(tuple(triples))]
        elif len(triples) == 2:
            triples = [str((triples[0], triples[1], ""))]

    for triple in triples:
        if not subgraph.contain_triple(triple, approach):
            return False
    return True


def validate_single_triples_output_v2(subgraph, triples, approach):
    triples_ = []
    for t in triples:
        if len(t) > 1:
            t_ = str(t)
            triples_.append(t_)

    valid_triples = []
    for triple in triples_:
        print("t --> ", triple)
        contains, original_triple = subgraph.contain_triple(triple, approach)
        if contains:
            valid_triples.append(original_triple)

    if len(valid_triples) == 0:
        return False, []
    else:
        triples = valid_triples
        return True, valid_triples


def validate_triples_output(subgraph, output, approach):
   
    for instance in output["output"]:
        triples = instance["triples"]
        triples_ = []
        if isinstance(triples, list) and isinstance(triples[0], list):
            for t in triples:
                if len(t) > 1:
                    t_ = str(t)
                    triples_.append(t_)
        elif isinstance(triples, list) and len(triples) >= 1:
            if isinstance(triples[0], str) and "," not in triples[0]:
                if (len(triples) % 3 == 0):
                    t_list = []
                    for i in range(0, len(triples), 3):
                        t_list.append(triples[i:i+3])
                    triples = t_list
                elif (len(triples) % 2 == 0):
                    t_list = []
                    for i in range(0, len(triples), 2):
                        t_list.append(triples[i:i+2])
                    triples = t_list

                triples = [triples]
            for triple in triples:
                if isinstance(triple, str):
                    triples_.append(triple)
                else:
                    triples_.append(str(triple))
        
        valid_triples = []
        for triple in triples_:
            print("t --> ", triple)
            contains, original_triple = subgraph.contain_triple(triple, approach)
            if contains:
                valid_triples.append(original_triple)

        if len(valid_triples) == 0:
            return False
        else:
            instance["triples"] = valid_triples
    return True

def validate_singleshot_triples_output(subgraph, output, approach):
   
    for instance in output["output"]:
        triples = instance["triple"]
        triples_ = []
        if isinstance(triples, list) and isinstance(triples[0], list):
            for t in triples:
                if len(t) > 1:
                    t_ = str(t)
                    triples_.append(t_)
        elif isinstance(triples, list) and len(triples) >= 1:
            if isinstance(triples[0], str) and "," not in triples[0]:
                if (len(triples) % 3 == 0):
                    t_list = []
                    for i in range(0, len(triples), 3):
                        t_list.append(triples[i:i+3])
                    triples = t_list
                elif (len(triples) % 2 == 0):
                    t_list = []
                    for i in range(0, len(triples), 2):
                        t_list.append(triples[i:i+2])
                    triples = t_list

                triples = [triples]
            for triple in triples:
                if isinstance(triple, str):
                    triples_.append(triple)
                else:
                    triples_.append(str(triple))
        
        valid_triples = []
        for triple in triples_:
            print("t --> ", triple)
            contains, original_triple = subgraph.contain_triple(triple, approach)
            if contains:
                valid_triples.append(original_triple)

        if len(valid_triples) == 0:
            return False
        else:
            instance["triple"] = valid_triples
    return True


def validate_singleshot_sparql_output(questions, transformed_questions, sparqls, endpoint):
    correct_questions = list()
    correct_queries = list()
    correct_transformed = list()
    query_results = dict()
    valid_sparql = True
    for question, transformed, answer_query in zip(questions, transformed_questions, sparqls):
        try:
            status = validate_query_v2(
                answer_query,
                endpoint,
            )
            if status in query_results:
                query_results[status] += 1
            else:
                query_results[status] = 1

            if status == "Correct":
                correct_questions.append(question)
                correct_queries.append(answer_query)
                correct_transformed.append(transformed)
        except Exception as e:
            traceback.print_exc()
            continue
    
    if len(correct_questions) < 3:
        print(correct_questions)
        print(query_results)
        valid_sparql = False
        query_results = dict()
        correct_questions = []
        correct_queries = []
        correct_transformed = []

    return (
        valid_sparql,
        query_results,
        correct_questions,
        correct_transformed,
        correct_queries
    )

def execute_dialogue_generation_prompt(seed, question_set, parent_trace):
    seed_entity = seed.label if seed.label else seed.uri
    seed_label = seed.label if seed.label else defrag_uri(str(seed.uri))
    transformed_questions = []
    errors = {"dialogue_json_parsing_error": 0, "dialogue_validation_error": 0}
    ch = pronoun_identification_and_substitution_chain.get("chain")
    post_processor = pronoun_identification_and_substitution_chain.get("post_processor")
    prompt = pronoun_identification_and_substitution_chain.get("prompt").format(
        entity=seed_entity, questions=question_set[1:]
    )

    chain_inputs = {"seed_label": seed_label, "prompt": prompt}
    d_chain_trace = Trace(
        name="D-Gen-Step",
        kind="CHAIN",
        start_time_ms = time.time_ns() // 1000,
        model_dict={
            "_model": config.dialogue_generation_model.model_name,
            "_kind": config.dialogue_generation_model.model_type,
        },
    )

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
            llm_result = ch.generate(
                [{"entity": seed_entity, "questions": question_set[1:]}], None
            )
            output = post_processor(llm_result, chain_inputs, d_chain_trace)
            transformed_questions = output["output"]
            valid = validate_dialogue_output(seed_label, transformed_questions)
            if valid:
                diag_validation_err = False
                break
            else:
                diag_validation_err = True
        except Exception as e:
            response = str(e)
            if response.startswith("Failed to parse"):
                try:
                    start_index = response.index("[")
                    end_index = response.rindex("]")
                    transformed_questions = ast.literal_eval(
                        response[start_index : end_index + 1]
                    )
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
        d_chain_trace.status_code = StatusCode.DIALOGUE_JSON_ERROR.name
    elif diag_validation_err:
        question_set_dialogue = None
        errors["dialogue_validation_error"] = 1
        d_chain_trace.status_code = StatusCode.DIALOGUE_VALIDATION_ERROR.name
    else:
        # no error
        question_0 = question_set[0]
        question_set_dialogue = [question_0, *transformed_questions]
        d_chain_trace.status_code = StatusCode.SUCCESS.name

    d_chain_trace._span.end_time_ms = time.time_ns() // 1000
    parent_trace.add_child(d_chain_trace)
    return question_set_dialogue, errors


def generate_dialogues_from_summarized_subgraph(
    initial_seed_nodes,
    kg,
    dialogue_size,
    output_file,
    prompt,
    sampler,
    seednode_to_subgraph_map,
):

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

    cost = { "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    for idx, seed in enumerate(seed_nodes):
        if idx > 500:
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
        n = dialogue_size
        seed_label = seed.label if seed.label else defrag_uri(str(seed.uri))

        parent_trace_inputs = {
            "seed_uri": str(seed.uri),
            "subgraph": subgraph.get_summarized_graph_str(approach="no_object"),
            "seed_label": seed_label,
        }
        parent_trace_outputs = None
        parent_trace = Trace(
            name="Pipeline",
            kind="PIPELINE",
            start_time_ms = time.time_ns() // 1000
        )
        with llm_callback() as cb:
            try:
                errors = {}

                que_trip_set, errors = execute_question_generation_prompt(
                    "summarized",
                    prompt,
                    subgraph,
                    n,
                    seed,
                    config.pipeline_type,
                    parent_trace=parent_trace,
                )

                if que_trip_set is None:
                    # question triple set was none so just count the errors and continue to next seed
                    context_length_limit_error += errors.get("context_length_error", 0)
                    question_json_parsing_error += errors.get(
                        "question_json_parsing_error", 0
                    )
                    question_validation_error += errors.get(
                        "question_validation_error", 0
                    )
                    triple_json_parsing_error += errors.get(
                        "triple_json_parsing_error", 0
                    )
                    triple_validation_error += errors.get("triple_validation_error", 0)
                    skip_node = True
                else:
                    # question triple stage was success move to answer query generation
                    (
                        question_set,
                        answer_queries,
                        triples_used,
                        answer_status_dict,
                        sp_json_err,
                        sp_val_err,
                    ) = decouple_questions_and_answers(
                        que_trip_set,
                        subgraph,
                        "optimized",
                        kg.sparql_endpoint,
                        seed.uri,
                        parent_trace=parent_trace,
                    )

                    if question_set is None:
                        sparql_json_parsing_error += sp_json_err
                        sparql_validation_error += sp_val_err
                        skip_node = True
                    else:
                        errors = {}
                        question_set_dialogue, errors = (
                            execute_dialogue_generation_prompt(
                                seed, question_set, parent_trace=parent_trace
                            )
                        )

                        if question_set_dialogue is None:
                            dialogue_json_parsing_error += errors.get(
                                "dialogue_json_parsing_error", 0
                            )
                            dialogue_validation_error += errors.get(
                                "dialogue_validation_error", 0
                            )
                            skip_node = True
                        else:
                            cb_dict = {
                                "total_tokens": cb.prompt_tokens + cb.completion_tokens,
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
                                "query_status": answer_status_dict,
                            }
                            parent_trace.status_code = StatusCode.SUCCESS.name
                            parent_trace_outputs = {
                                "dialogue": question_set_dialogue,
                                "original": question_set,
                                "queries": answer_queries,
                                "query_status": answer_status_dict,
                            }
                            print(dialogue)
                            benchmark_sample.append(dialogue)
                            end_time = time.time()
                            total_time += end_time - start_time
                            processed_seeds += 1
            except Exception as e:
                traceback.print_exc()
                logger.info(f"INDEX : {idx} -- ERROR: {idx} : {e} --")
                parent_trace.status_code = StatusCode.ERROR
                parent_trace.status_message = "unknown error occurred"
                parent_trace._span.end_time_ms = time.time_ns() // 1000

            cost["total_tokens"] += (cb.prompt_tokens + cb.completion_tokens)
            cost["prompt_tokens"] += cb.prompt_tokens
            cost["completion_tokens"] += cb.completion_tokens

        if skip_node is True:
            # Sample a new node and add it to seed nodes
            node_added = False
            while not node_added:
                try:
                    new_seed, subgraph = retrieve_one_node_with_subgraph(
                        sampler, seed.nodetype, kg
                    )
                    key = new_seed.label if new_seed.label else new_seed.uri
                    seed_nodes.append(new_seed)
                    seednode_to_subgraph_map[key] = subgraph
                    node_added = True
                except Exception as e:
                    print("Exception ", e)
                    continue

        if parent_trace is not None:
            print("####### trace saved ########")
            parent_trace.add_inputs_and_outputs(
                inputs=parent_trace_inputs, outputs=parent_trace_outputs
            )
            parent_trace._span.end_time_ms = time.time_ns() // 1000
            parent_trace.log("diag-gen-pipeline")

            # sync with wandb online
            trigger_sync()
            # break

        benchmark_analysis = analyze_benchmark_sample(benchmark_sample)
        benchmark = {
            "seeds_used": idx + 1,
            "data": benchmark_sample,
            "analysis": benchmark_analysis,
            "cost": cost,
            "total_time": total_time,
            "average_time": (
                0 if processed_seeds == 0 else (total_time / processed_seeds)
            ),
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


def execute_question_triple_binding_prompt(
    subgraph, subgraph_str, question_list, q_gen_trace
):  # noqa: E501
    errors = {
        "triple_validation_error": 0,
        "triple_json_parsing_error": 0,
    }
    output_with_triple = []
    t_validation_err_cnt = 0
    t_json_parse_err_cnt = 0
    ch = get_triple_for_question_given_subgraph_chain_without_example.get("chain")
    post_processor = get_triple_for_question_given_subgraph_chain_without_example.get(
        "post_processor"
    )
    for q in question_list:
        prompt = get_triple_for_question_given_subgraph_chain_without_example.get(
            "prompt"
        ).format(subgraph=subgraph_str, question=q)
        chain_input = ({"prompt": prompt},)
        t_chain_trace = Trace(
            name="Triple-Binding",
            kind="LLM",
            start_time_ms = time.time_ns() // 1000,
            model_dict={
                "_model": config.question_generation_model.model_name,
                "_kind": config.question_generation_model.model_type,
            },
        )
        try:
            llm_result = ch.generate([{"subgraph": subgraph_str, "question": q}], None)
            output = post_processor(llm_result, chain_input, t_chain_trace)

            is_valid, valid_triples = validate_single_triples_output_v2(
                subgraph, output, "optimized"
            )
            if not is_valid:
                t_validation_err_cnt += 1
                t_chain_trace._span.status_code = (
                    StatusCode.TRIPLES_VALIDATION_ERROR.name
                )
            else:
                output_with_triple.append({"question": q, "triples": valid_triples})
                t_chain_trace._span.status_code = StatusCode.SUCCESS
        except Exception as e:
            t_json_parse_err_cnt += 1
            print("Exception in triple binding, skipping question", e)
            t_chain_trace._span.status_code = StatusCode.TRIPLES_JSON_ERROR

        t_chain_trace._span.end_time_ms = time.time_ns() // 1000
        q_gen_trace.add_child(t_chain_trace)

    if len(question_list) != len(output_with_triple):
        # not all triple binding were correct
        if t_json_parse_err_cnt >= t_validation_err_cnt:
            errors["triple_json_parsing_error"] = 1
            q_gen_trace._span.status_code = StatusCode.TRIPLES_JSON_ERROR.name
            return None, errors
        else:
            errors["triple_validation_error"] = 1
            q_gen_trace._span.status_code = StatusCode.TRIPLES_VALIDATION_ERROR.name
            return None, errors
    else:
        q_gen_trace._span.status_code = StatusCode.SUCCESS.name
        return output_with_triple, errors


def execute_question_generation_prompt(
    subgraph_approach, prompt, subgraph, n, seed, pipeline_type, parent_trace
):
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
        subgraph_str = subgraph.__str__(representation="uri")
        prompt = n_question_from_subgraph_chain_without_example.get("prompt").format(
            subgraph=subgraph_str, n=n
        )

        chain_inputs = {"prompt": prompt}
        q_chain_trace = Trace(
            name="Q-Gen-Step",
            kind="CHAIN",
            start_time_ms = time.time_ns() // 1000,
            model_dict={
                "_model": config.question_generation_model.model_name,
                "_kind": config.question_generation_model.model_type,
            },
        )

        num_tokens = get_num_tokens(prompt)
        if num_tokens > 4097:
            errors["context_length_error"] = 1
            return None, errors

        ch = n_question_from_subgraph_chain_without_example.get("chain")
        post_processor = n_question_from_subgraph_chain_without_example.get(
            "post_processor"
        )

        que_trip_set = None
        valid_question = False
        valid_triples = False
        retry = 0
        question_json_parsing_error = False
        while not (valid_question and valid_triples):
            question_json_parsing_error = False
            try:
                llm_result = ch.generate([{"subgraph": subgraph_str, "n": n}], None)
                output = post_processor(llm_result, chain_inputs, q_chain_trace)
                valid_question = validate_questions_output(seed_label, output)
                valid_triples = validate_triples_output(subgraph, output, "subgraph")
            except Exception as e:
                response = str(e)
                if response.startswith("Failed to parse"):
                    try:
                        start_index = response.index("[")
                        # end_index = response.index("Got:")
                        end_index = response.rindex("]")
                        data = ast.literal_eval(response[start_index : end_index - 3])
                        output = {"output": data}
                        valid_question = validate_questions_output(seed_label, output)
                        valid_triples = validate_triples_output(
                            subgraph, output, "subgraph"
                        )
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
            q_chain_trace._span.status_code = StatusCode.QUESTION_JSON_ERROR.name
        else:
            if not valid_question:
                que_trip_set = None
                errors["question_validation_error"] = 1
                q_chain_trace._span.status_code = (
                    StatusCode.QUESTION_VALIDATION_ERROR.name
                )
            elif not valid_triples:
                que_trip_set = None
                errors["triple_validation_error"] = 1
                q_chain_trace._span.status_code = (
                    StatusCode.TRIPLES_VALIDATION_ERROR.name
                )
            else:
                # no error
                que_trip_set = output["output"]
                q_chain_trace._span.status_code = StatusCode.SUCCESS.name

        q_chain_trace._span.end_time_ms = time.time_ns() // 1000
        parent_trace.add_child(q_chain_trace)
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
            subgraph_str = subgraph.get_summarized_graph_str(approach="no_object")
            prompt = n_question_from_summarized_subgraph_chain_without_example.get(
                "prompt"
            ).format(subgraph=subgraph_str, entity=seed_label, n=n)

            chain_inputs = {"prompt": prompt}
            q_chain_trace = Trace(
                name="Q-Gen-Step",
                kind="CHAIN",
                start_time_ms = time.time_ns() // 1000,
                model_dict={
                    "_model": config.question_generation_model.model_name,
                    "_kind": config.question_generation_model.model_type,
                },
            )

            num_tokens = get_num_tokens(prompt)
            if num_tokens > 4097:
                errors["context_length_error"] = 1
                q_chain_trace._span.status_code = StatusCode.CONTEXT_LENGTH_ERROR.name
                q_chain_trace._span.end_time_ms = time.time_ns() // 1000
                parent_trace.add_child(q_chain_trace)
                return None, errors

            ch = n_question_from_summarized_subgraph_chain_without_example.get("chain")
            post_processor = (
                n_question_from_summarized_subgraph_chain_without_example.get(
                    "post_processor"
                )
            )

            que_trip_set = None
            valid_question = False
            valid_triples = False
            retry = 0
            question_json_parsing_error = False
            while not (valid_question and valid_triples):
                question_json_parsing_error = False
                try:
                    llm_result = ch.generate([{"subgraph": subgraph_str, "entity": seed_label, "n": n}], None)
                    output = post_processor(llm_result, chain_inputs, q_chain_trace)
                    valid_question = validate_questions_output(seed_label, output)
                    valid_triples = validate_triples_output(
                        subgraph, output, "optimized"
                    )
                except Exception as e:
                    response = str(e)
                    if response.startswith("Failed to parse"):
                        try:
                            start_index = response.index("[")
                            # end_index = response.index("Got:")
                            end_index = response.rindex("]")
                            data = ast.literal_eval(
                                response[start_index : end_index - 3]
                            )
                            output = {"output": data}
                            valid_question = validate_questions_output(
                                seed_label, output
                            )
                            valid_triples = validate_triples_output(
                                subgraph, output, "optimized"
                            )
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
                q_chain_trace._span.status_code = StatusCode.QUESTION_JSON_ERROR.name
            else:
                if not valid_question:
                    que_trip_set = None
                    errors["question_validation_error"] = 1
                    q_chain_trace._span.status_code = (
                        StatusCode.QUESTION_VALIDATION_ERROR.name
                    )
                elif not valid_triples:
                    que_trip_set = None
                    errors["triple_validation_error"] = 1
                    q_chain_trace._span.status_code = (
                        StatusCode.TRIPLES_VALIDATION_ERROR.name
                    )
                else:
                    # no error
                    que_trip_set = output["output"]
                    q_chain_trace._span.status_code = StatusCode.SUCCESS.name

            q_chain_trace._span.end_time_ms = time.time_ns() // 1000
            parent_trace.add_child(q_chain_trace)
            return que_trip_set, errors

        elif pipeline_type == "simplified":
            seed_entity_representation = seed.label if seed.label else seed.uri
            subgraph_str = subgraph.get_summarized_graph_str(approach="no_object")
            prompt = n_question_from_summarized_subgraph_chain_without_example_without_triple.get(
                "prompt"
            ).format(
                subgraph=subgraph_str, n=n, entity=seed_entity_representation
            )

            chain_inputs = {"prompt": prompt}
            q_chain_trace = Trace(
                name="Q-Gen-Step",
                kind="CHAIN",
                start_time_ms = time.time_ns() // 1000,
                model_dict={
                    "_model": config.question_generation_model.model_name,
                    "_kind": config.question_generation_model.model_type,
                },
            )

            num_tokens = get_num_tokens(prompt)
            if num_tokens > 4097:
                errors["context_length_error"] = 1
                q_chain_trace._span.status_code = StatusCode.CONTEXT_LENGTH_ERROR.name
                q_chain_trace._span.end_time_ms = time.time_ns() // 1000
                parent_trace.add_child(q_chain_trace)
                return None, errors

            ch = n_question_from_summarized_subgraph_chain_without_example_without_triple.get(
                "chain"
            )
            post_processor = n_question_from_summarized_subgraph_chain_without_example_without_triple.get(
                "post_processor"
            )

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
                    llm_result = ch.generate([{"subgraph": subgraph_str, "n": n, "entity": seed_entity_representation}], None)
                    output = post_processor(
                        llm_result, chain_inputs, q_chain_trace
                    )  # output would be list of question
                    question_list = output["output"]
                    q_output = {"output": [{"question": q} for q in question_list]}
                    valid_question = validate_questions_output(
                        seed_entity_representation, q_output
                    )
                    if valid_question:
                        question_validation_error = False
                        break
                    else:
                        question_validation_error = True
                except Exception as e:
                    traceback.print_exc()
                    response = str(e)
                    if response.startswith("Failed to parse"):
                        try:
                            start_index = response.index("[")
                            # end_index = response.index("Got:")
                            end_index = response.rindex("]")
                            data = ast.literal_eval(
                                response[start_index : end_index - 3]
                            )
                            # output = {"output": data}
                            question_list = data
                            q_output = {"output": [{"question": q} for q in question_list]}
                            valid_question = validate_questions_output(
                                seed_entity_representation, q_output
                            )
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
                q_chain_trace._span.status_code = StatusCode.QUESTION_JSON_ERROR
            elif question_validation_error:
                que_trip_set = None
                errors["question_validation_error"] = 1
                q_chain_trace._span.status_code = StatusCode.QUESTION_VALIDATION_ERROR
            else:
                # no error
                # move to triple binding step

                que_trip_set, t_errors = execute_question_triple_binding_prompt(
                    subgraph, subgraph_str, question_list, q_gen_trace=q_chain_trace
                )
                errors["triple_json_parsing_error"] = t_errors[
                    "triple_json_parsing_error"
                ]
                errors["triple_validation_error"] = t_errors["triple_validation_error"]

            q_chain_trace._span.end_time_ms = time.time_ns() // 1000
            parent_trace.add_child(q_chain_trace)
            return que_trip_set, errors


def generate_dialogues_from_singleshot(
    initial_seed_nodes,
    kg,
    dialogue_size,
    output_file,
    prompt,
    sampler,
    seednode_to_subgraph_map,
):
    benchmark_sample = []
    raw_benchmark_sample = []
    seed_nodes = initial_seed_nodes.copy()
    total_time = 0
    processed_seeds = 0
    context_length_limit_error = 0
    question_validation_error = 0
    sparql_validation_error = 0
    dialogue_validation_error = 0
    json_parsing_error = 0
    unequal_lists_error = 0

    cost = { "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    
    for idx, seed in enumerate(seed_nodes):
        if idx == 200:
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
        n = dialogue_size
        seed_uri = str(seed.uri)
        seed_label = seed.label if seed.label else defrag_uri(str(seed.uri))

        parent_trace_inputs = {
            "seed_uri": str(seed.uri),
            "subgraph": subgraph.get_summarized_graph_str(approach="no_object"),
            "seed_label": seed_label,
        }
        parent_trace_outputs = None
        parent_trace = Trace(
            name="Pipeline",
            kind="PIPELINE",
            start_time_ms = time.time_ns() // 1000
        )
        with llm_callback() as cb:
            try:
                errors = {}
                ch = singleshot_dialogue_chain.get("chain")
                post_processor = singleshot_dialogue_chain.get(
                    "post_processor"
                )

                query_subgraph_str = subgraph.get_summarized_graph_query_str(approach="no_object")
                prompt_v2 = singleshot_dialogue_chain.get("prompt").format(
                    n=n,
                    entity_uri=seed_uri,
                    entity_label=seed_label,
                    query_subgraph=query_subgraph_str,
                )

                chain_inputs = {"prompt": prompt_v2}
                q_chain_trace = Trace(
                    name="Q-Gen-Step",
                    kind="CHAIN",
                    start_time_ms = time.time_ns() // 1000,
                    model_dict={
                        "_model": config.question_generation_model.model_name,
                        "_kind": config.question_generation_model.model_type,
                    },
                )

                num_tokens = get_num_tokens(prompt_v2)
                if num_tokens > 4097:
                    q_chain_trace._span.status_code = StatusCode.CONTEXT_LENGTH_ERROR.name
                    q_chain_trace._span.end_time_ms = time.time_ns() // 1000
                    parent_trace.add_child(q_chain_trace)
                    context_length_limit_error += 1

                valid_question = False
                valid_triples = False
                parsing_error = False
                unequal_length = False
                try:
                    # Uncomment when running with Gemini
                    # print("TIME SLEEP of 30.0 seconds")
                    # time.sleep(30)

                    llm_result = ch.generate([{"query_subgraph": query_subgraph_str, "entity_label": seed_label, "n": n}], None)
                    raw_benchmark_sample.append({
                        "input": prompt,
                        "generations": llm_result.dict().get('generations'),
                        "token_usage": llm_result.dict().get('llm_output'),
                        "n": n
                    })
                    output = post_processor(llm_result, chain_inputs, q_chain_trace)
                    questions = output["questions"]
                    transformed_questions = output["dialogue"]
                    sparqls = output["sparql"]
                    if (len(questions) == len(transformed_questions) == len(sparqls)):
                        valid_question = validate_singleshot_questions_output(seed_label, questions)
                        print("QUESTION-validation", valid_question)
                        if valid_question:
                            valid_sparqls, answer_status_dict, correct_questions, correct_transformed, correct_queries = validate_singleshot_sparql_output(questions, transformed_questions, sparqls, kg.sparql_endpoint)
                            if valid_sparqls:
                                valid_dialogue = validate_singleshot_dialogue_output(seed_label, transformed_questions)
                    else:
                        unequal_length = True

                except Exception as e:
                    parsing_error = True
                    traceback.print_exc()

                if unequal_length == True:
                    errors["unequal_lists_error"] = 1
                    unequal_lists_error += 1
                    q_chain_trace._span.status_code = (
                        StatusCode.DIALOGUE_VALIDATION_ERROR
                    )
                    skip_node = True
                else:
                    if parsing_error:
                        errors["json_parsing_error"] = 1
                        json_parsing_error += 1
                        skip_node = True
                        q_chain_trace._span.status_code = StatusCode.JSON_ERROR
                    else:
                        if not valid_question:
                            errors["question_validation_error"] = 1
                            question_validation_error += 1
                            q_chain_trace._span.status_code = (
                                StatusCode.QUESTION_VALIDATION_ERROR
                            )
                            skip_node = True
                        elif not valid_sparqls:
                            errors["sparql_validation_error"] = 1
                            sparql_validation_error += 1
                            q_chain_trace._span.status_code = (
                                StatusCode.SPARQL_VALIDATION_ERROR
                            )
                            skip_node = True
                        elif not valid_dialogue:
                            errors["dialogue_validation_error"] = 1
                            dialogue_validation_error += 1
                            q_chain_trace._span.status_code = (
                                StatusCode.DIALOGUE_VALIDATION_ERROR
                            )
                            skip_node = True
                        else:
                            question_set_dialogue = correct_transformed
                            answer_queries = correct_queries
                            question_set = correct_questions

                            cb_dict = {
                                "total_tokens": cb.prompt_tokens + cb.completion_tokens,
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
                                "cost": cb_dict,
                                "query_status": answer_status_dict,
                            }
                            benchmark_sample.append(dialogue)
                            end_time = time.time()
                            total_time += end_time - start_time
                            parent_trace.status_code = StatusCode.SUCCESS.name
                            parent_trace_outputs = {
                                "dialogue": question_set_dialogue,
                                "original": question_set,
                                "queries": answer_queries,
                                "query_status": answer_status_dict,
                            }
                            print(dialogue)
                            processed_seeds += 1

                
            except Exception as e:
                traceback.print_exc()
                logger.info(f"INDEX : {idx} -- ERROR: {idx} : {e} --")
            
            cost["total_tokens"] += (cb.prompt_tokens + cb.completion_tokens)
            cost["prompt_tokens"] += cb.prompt_tokens
            cost["completion_tokens"] += cb.completion_tokens
        

        if parent_trace is not None:
            print("####### trace saved ########")
            parent_trace.add_inputs_and_outputs(
                inputs=parent_trace_inputs, outputs=parent_trace_outputs
            )
            parent_trace._span.end_time_ms = time.time_ns() // 1000
            parent_trace.log("singleshot-diag-gen-pipeline")

            # sync with wandb online
            trigger_sync()
            # break

        if skip_node is True:
            # Sample a new node and add it to seed nodes
            node_added = False
            while not node_added:
                try:
                    new_seed, subgraph = retrieve_one_node_with_subgraph(
                        sampler, seed.nodetype, kg
                    )
                    key = new_seed.label if new_seed.label else new_seed.uri
                    seed_nodes.append(new_seed)
                    seednode_to_subgraph_map[key] = subgraph
                    node_added = True
                except Exception as e:
                    print("Exception ", e)
                    continue

        benchmark_analysis = analyze_benchmark_sample(benchmark_sample)
        benchmark = {
            "seeds_used": idx + 1,
            "data": benchmark_sample,
            "analysis": benchmark_analysis,
            "cost": cost,
            "total_time": total_time,
            "average_time": (
                0 if processed_seeds == 0 else (total_time / processed_seeds)
            ),
            "Context Length Error": context_length_limit_error,
            "Question Validation Error": question_validation_error,
            "Sparql Validation Error": sparql_validation_error,
            "Dialogue Validation Error": dialogue_validation_error,
            "Json Error": json_parsing_error,
            "unequal Lists Error": unequal_lists_error
        }
        directory = pathlib.Path(output_file).parent
        directory.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(benchmark, f, indent=4)
        
        raw_benchmark = os.path.join(directory, "raw_benchmark.json")
        with open(raw_benchmark, "w") as f:
            json.dump(raw_benchmark_sample, f, indent=4)
