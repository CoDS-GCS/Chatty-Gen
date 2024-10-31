from benchmark.seed_node_extractor import utils
import pandas as pd

def get_num_unique_predicates(knowledge_graph_uri):
    query = "select Count(distinct ?p) As ?count where {?e ?p ?o}"
    result = utils.send_sparql_query(knowledge_graph_uri, query)
    count = result["results"]["bindings"][0]["count"]["value"]
    print("Unique Predicates: ", count)
    return count

def get_num_entities(knowledge_graph_uri, knowledge_graph_prefix):
    type_distribution = utils.get_type_distrubution(knowledge_graph_uri, knowledge_graph_prefix)
    distribution = pd.DataFrame(type_distribution)
    distribution['Count'] = distribution['Count'].astype(int)
    total_count = distribution['Count'].sum()
    print("Num Entities: ", total_count)
    return total_count

def get_num_triples(knowledge_graph_uri):
    query = "select Count(*) As ?count where {?s ?p ?o}"
    result = utils.send_sparql_query(knowledge_graph_uri, query)
    count = result["results"]["bindings"][0]["count"]["value"]
    print("Num Triples: ", count)
    return count

if __name__ == '__main__':
    kg_name = 'makg'
    knowledge_graph_uri = utils.knowledge_graph_to_uri[kg_name][0]
    knowledge_graph_prefix = utils.knowledge_graph_to_uri[kg_name][1]
    preds = get_num_unique_predicates(knowledge_graph_uri)
    ents = get_num_entities(knowledge_graph_uri, knowledge_graph_prefix)
    trip = get_num_triples(knowledge_graph_uri)
    print("Unique Predicates: ", preds, ", Number of Entities: ", ents, ", Number of Triples: ", trip)

