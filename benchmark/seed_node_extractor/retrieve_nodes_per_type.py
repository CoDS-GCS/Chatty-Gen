import utils
import concurrent.futures
import os
import pandas as pd


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def write_entities_to_file(type, prefix, results):
    file_name = utils.get_file_name_from_type(type)
    file_name = f'data/{prefix}/{file_name}.txt'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = open(file_name, 'w')
    for result in results:
        file.write(result + '\n')
    file.close()


def get_all_nodes_for_type(type, sparql_endpoint, offset):
    query = ("select ?entity where {?entity rdf:type "
             f"<{type}>"
             "} limit 10000  "
             f"offset {offset}")
    query_result = utils.send_sparql_query(sparql_endpoint, query)
    data = []
    for binding in query_result['results']['bindings']:
        entity = binding.get('entity', {}).get('value', None)
        if entity.startswith('http://') or entity.startswith('https://'):
            data.append(entity)
    return data


def retrieve_node_parallel(type, sparql_endpoint, prefix, num_entities):
    num_threads = 10
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        offsets = range(0, num_entities, 10000)
        futures = [executor.submit(get_all_nodes_for_type, type, sparql_endpoint, offset) for offset in offsets]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.extend(result)

    write_entities_to_file(type, prefix, results)


if __name__ == '__main__':
    # inputs
    knowledge_graph_to_uri = {
        "dbpedia": ("http://206.12.95.86:8890/sparql", "dbpedia"),
        # "lc_quad": "http://206.12.95.86:8891/sparql",
        "microsoft_academic": ("http://206.12.97.159:8890/sparql", "makg"),
        "yago": ("http://206.12.95.86:8892/sparql", "yago"),
        "dblp": ("http://206.12.95.86:8894/sparql", "dblp"),
    }
    kg = "dblp"
    knowledge_graph_uri = knowledge_graph_to_uri[kg][0]
    knowledge_graph_prefix = knowledge_graph_to_uri[kg][1]
    distribution = utils.get_type_distrubution(knowledge_graph_uri, knowledge_graph_prefix)
    distribution = pd.DataFrame(distribution)
    types = distribution['Type'].values
    for type in types:
        retrieve_node_parallel(type, knowledge_graph_uri, knowledge_graph_prefix,
                               int(distribution[distribution['Type'] == type]['Count'].values[0]))