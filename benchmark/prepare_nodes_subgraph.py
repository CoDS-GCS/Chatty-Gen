import time
import os
import json
import traceback
import re
import pdb

from func_timeout import func_timeout, FunctionTimedOut
from kg.kg.kg import DblpKG, YagoKG, DbpediaKG
from seed_node_extractor import utils
from llm.prompt_chains import get_prompt_chains

prompt_chains = get_prompt_chains()
representative_label_for_type = prompt_chains.get("get_representative_label_for_type")
max_label_length = 10

def perform_operation(kg, seed):
    # Put your operation here
    start_time = time.time()
    estimate = kg.estimate_graph_size(seed)
    if estimate > 1000:
        return None
    subgraph = kg.subgraph_extractor(seed)
    end_time = time.time()
    # print("Subgraph Extracted After ", end_time - start_time, " seconds")
    subgraph = kg.filter_subgraph(subgraph, seed)
    if seed.label and len(seed.label.split(' ')) > max_label_length or len(subgraph.triples) > 400:
        return None
    # print("Subgraph filtered After ", time.time() - end_time, " seconds")
    return subgraph

def perform_operation_new(kg, seed):
    subgraph = kg.subgraph_extractor(seed)
    subgraph = kg.filter_subgraph(subgraph, seed)
    if len(subgraph.triples) > 400 or len(subgraph.triples) < 5:
        return None
    return subgraph

def get_kg_instance(kg_name):
    kgs = {"yago": YagoKG, "dblp": DblpKG, "dbpedia": DbpediaKG}
    kg = kgs.get(kg_name, None)
    if kg is None:
        raise ValueError(f"kg : {kg_name} not supported")
    return kg

def trim_after_first_occurrence(text, pattern):
    # Find the first occurrence of the pattern
    match = re.search(pattern, text)
    
    # If the pattern is found, return the text up to the first occurrence
    if match:
        return text[:match.end()]
    else:
        # If the pattern is not found, return the original text
        return text

def get_representative_label_per_node_type(endpoint, sampling_distribution, seed_nodes, exisiting_map, file_name):
    # pdb.set_trace()
    type_per_label = exisiting_map
    count = 0
    for key, value in sampling_distribution.items():
        if value == 0:
            continue
        key = key.strip()
        if key not in type_per_label:
            sample_node = seed_nodes[count]
            sample_node_str = str(sample_node)
            query = f"select distinct ?p, ?ent where {{ <{sample_node_str}> ?p ?ent}}"
            # query = ("select distinct ?p, ?ent where { {"
            #          f"<{sample_node_str}> ?p ?ent"
            #          "} UNION {?ent ?p "
            #          f"<{sample_node_str}>"
            #          "} } ")
            result = utils.send_sparql_query(endpoint, query)
            # if result is None:
                # continue
            predicates = list()
            for binding in result["results"]["bindings"]:
                entity_type = binding.get('ent', {}).get('type', None)
                predicate = binding.get('p', {}).get('value', None)
                if entity_type == 'literal' and predicate not in predicates:
                    predicates.append(predicate)

            try:
#                 output = representative_label_for_type.get("chain").run(
#                     {"node_type": key, "predicates": ', '.join(predicates)}
#                 )
                ch = representative_label_for_type.get("chain")
                llm_result = ch.generate([{"node_type": key, "predicates": ', '.join(predicates)}], None)
                print(llm_result)
                for generation in llm_result.generations:
                    generation[0].text = "```json" + trim_after_first_occurrence(generation[0].text, "```")
                    print("gen-text: ", generation[0].text)
                output = [
                    # Get the text of the top generated string.
                    {
                        ch.output_key: ch.output_parser.parse_result(generation),
                        "full_generation": generation,
                    }
                    for generation in llm_result.generations
                ]
                if ch.return_final_only:
                    output = [{ch.output_key: r[ch.output_key]} for r in output]
                output = output[0][ch.output_key].dict()
                print(output)
                type_per_label[key] = output["predicate"].strip()
            except Exception as e:
                response = str(e)
                print("response")
                print(response)
                if response.startswith("Got invalid return object. Expected key"):
                    start_index = response.index('got')
                    type_per_label[key] = response[start_index + 3: len(response)].strip()
            with open(file_name, 'w') as file:
                json.dump(type_per_label, file, indent=4)
        # To keep the mapping between seed entities and types
        count += value
    return type_per_label


def retrieve_seed_nodes_with_subgraphs_new(kg_name, dataset_size, sampler, use_label):
    # pdb.set_trace()
    seed_nodes, sample_distribution, type_to_predicate_map = sampler.retrieve_initial_list_top_k(kg_name, dataset_size)
    seednode_to_subgraph_map = dict()
    final_seed_nodes = list()
    KG = get_kg_instance(kg_name)
    kg = KG()
    kg.set_type_to_predicate_map(type_to_predicate_map)
    kg.set_use_label(use_label)
    for seed in seed_nodes:
        try:
            subgraph = func_timeout(300, perform_operation_new, args=(kg, seed))
            if subgraph is None:
                new_node = sampler.sample_node(seed.nodetype)
                seed_nodes.append(new_node)
                continue
            # print(f"{seed.uri}\t{seed.label}")
            key = seed.label if seed.label else seed.uri
            seednode_to_subgraph_map[key] = subgraph
            final_seed_nodes.append(seed)
        except FunctionTimedOut:
            print("Operation timed out for seed ", seed)
            new_node = sampler.sample_node_new(seed.nodetype)
            seed_nodes.append(new_node)
        except Exception as e:
            traceback.print_exc()
            print("Error while Handling seed ", seed)
            print(e)

    return final_seed_nodes, seednode_to_subgraph_map, kg


def retrieve_seed_nodes_with_subgraphs(kg_name, dataset_size, sampler, use_label):
    # pdb.set_trace()
    seed_nodes, sample_distribution = sampler.retrieve_initial_list_top_k(dataset_size)
    KG = get_kg_instance(kg_name)
    kg = KG()
    type_to_predicate_map = dict()
    if use_label:
        file_name = f"{kg_name}_types_representative.json"
        if os.path.exists(file_name):
            file = open(file_name, 'r')
            type_to_predicate_map = json.load(file)
        type_to_predicate_map = get_representative_label_per_node_type(kg.sparql_endpoint, sample_distribution,
                                                                       seed_nodes, type_to_predicate_map, file_name)
    kg.set_type_to_predicate_map(type_to_predicate_map)
    kg.set_use_label(use_label)
    seednode_to_subgraph_map = dict()

    final_seed_nodes = list()
    for seed in seed_nodes:
        try:
            # subgraph = func_timeout(3, perform_operation, args=(kg, seed))
            subgraph = perform_operation(kg, seed)
            if subgraph is None:
                new_node = sampler.sample_node(seed.nodetype)
                seed_nodes.append(new_node)
                continue
            # print(f"{seed.uri}\t{seed.label}")
            key = seed.label if seed.label else seed.uri
            seednode_to_subgraph_map[key] = subgraph
            final_seed_nodes.append(seed)
        except FunctionTimedOut:
            print("Operation timed out for seed ", seed)
            new_node = sampler.sample_node(seed.nodetype)
            seed_nodes.append(new_node)
        except Exception as e:
            traceback.print_exc()
            print("Error while Handling seed ", seed)
            print(e)

    return final_seed_nodes, seednode_to_subgraph_map, kg


def retrieve_one_node_with_subgraph(sampler, node_type, kg):
    valid = False
    while not valid:
        try:
            seed_node = sampler.sample_node(node_type)
            # subgraph = func_timeout(300, perform_operation, args=(kg, seed_node))
            subgraph = func_timeout(300, perform_operation_new, args=(kg, seed_node))
            valid = subgraph is not None
        except FunctionTimedOut:
            print("Operation timed out for seed ", seed_node)

    return seed_node, subgraph
