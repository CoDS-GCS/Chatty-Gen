from seed_node_extractor import utils
import traceback


def check_equality_of_lists(list1, list2):
    if len(list1) != len(list2):
        return False
    for el in list1:
        if el not in list2:
            return False
    return True

def get_answers_from_subgraph(subgraph, triples, seed_node_uri):
    predicates = set()
    for triple in triples:
        predicates.add(triple[1].__str__())
    predicates = list(predicates)
    seed_node_uri = seed_node_uri.__str__()
    answers = list()
    for triple in subgraph.triples:
        if triple[1].__str__() not in  predicates:
            continue
        if triple[0].__str__() == seed_node_uri:
            answers.append(triple[2].__str__())
        elif triple[2].__str__() == seed_node_uri:
            answers.append(triple[0].__str__())
    return answers

def get_namespace_prefix():
    prefix = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX yago: <http://yago-knowledge.org/resource/>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX schema: <http://schema.org/>
    """
    return prefix

def validate_query(query_string, triples_used, endpoint, subgraph, seed_node_uri, approach):
    """
    Returns 0: if the query is executable and correct, 1: query is executable but answer is wrong,
    2: query is not executable
    """
    if query_string.lower().startswith('ask'):
        query_string = get_namespace_prefix() + query_string
        return validate_ask_query(query_string, triples_used, endpoint, subgraph, seed_node_uri)
    elif 'count(' in query_string.lower():
        query_string = get_namespace_prefix() + query_string
        return validate_count_query(query_string, triples_used, endpoint, subgraph, seed_node_uri, approach)
    else:
        query_string = get_namespace_prefix() + query_string
        return validate_select_query(query_string, triples_used, endpoint, subgraph, seed_node_uri, approach)


def validate_ask_query(query_string, triples_used, endpoint, subgraph, seed_node_uri):
    """
    Currently The answer to all Ask queries generated is True
    """
    try :
        result = utils.send_sparql_query(endpoint, query_string)
        if "boolean" in result and result["boolean"] is True:
            return "Correct"
        print("Incorrect query ", query_string)
        return "In Correct"
    except Exception as e:
        traceback.print_exc()
        print("Error Query ", query_string)
        return "Syntax Error"


def validate_count_query(query_string, original_triples, endpoint, subgraph, seed_node_uri, approach):
    try:
        result = utils.send_sparql_query(endpoint, query_string)
        variable_name = result["head"]["vars"][0]
        endpoint_answers = list()
        for binding in result["results"]["bindings"]:
            value = binding.get(variable_name, {}).get('value', None)
            value = int(value)
            endpoint_answers.append(value)

        # original_triples = list()
        # for triple in triples_used:
        #     if approach == "optimized":
        #         triple_ = subgraph.get_triple_with_uris_no_object(triple)
        #         original_triples.append(triple_)
        #     else:
        #         original_triples.append(triple_)

        print(endpoint_answers)
        subgraph_answers = get_answers_from_subgraph(subgraph, original_triples, seed_node_uri)
        answer = ""
        if len(subgraph_answers) == 1 and subgraph_answers[0].isnumeric():
            answer = int(subgraph_answers[0])
        else:
            answer = len(subgraph_answers)
        if answer == endpoint_answers[0]:
            return "Correct"

        print("Incorrect query ", query_string)
        return "In Correct"
    except Exception as e:
        # Query cannot be executed so it is not a valid one
        traceback.print_exc()
        print("Error Query ", query_string)
        return "Syntax Error"

def validate_select_query(query_string, original_triples, endpoint, subgraph, seed_node_uri, approach):
    try:
        result = utils.send_sparql_query(endpoint, query_string)
        # Need to check if the query returned a result inside the subgraph
        variable_name = result["head"]["vars"][0]
        endpoint_answers = list()
        for binding in result["results"]["bindings"]:
            value = binding.get(variable_name, {}).get('value', None)
            endpoint_answers.append(value)
        # original_triples = list()
        # for triple in triples_used:
        #     if approach == "optimized":
        #         triple_ = subgraph.get_triple_with_uris_no_object(triple)
        #         original_triples.append(triple_)
        #     else:
        #         original_triples.append(triple_)

        subgraph_answers = list()
        subgraph_answers = get_answers_from_subgraph(subgraph, original_triples, seed_node_uri)
        if check_equality_of_lists(endpoint_answers, subgraph_answers):
            return "Correct"
        print("Incorrect query ", query_string)
        return "In Correct"
    except Exception as e:
        # Query cannot be executed so it is not a valid one
        traceback.print_exc()
        print("Error Query ", query_string)
        return "Syntax Error"

