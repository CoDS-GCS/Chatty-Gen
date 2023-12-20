import utils
import time
import pandas as pd
import concurrent.futures
import threading
import os


predict_lock = threading.Lock()


def write_result_to_file(type, data, prefix, types_file):
    human_readable_type = utils.get_file_name_from_type(type)
    output_file = f'index_data/{prefix}/{human_readable_type}.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    file = open(output_file, 'w')
    df = pd.DataFrame(data)
    if len(df)==0:
        return 0.0
    groups = df.groupby('entity')['predicate']
    num_entities = 0
    unique_predicates = 0
    for group in groups:
        predicate_no_dup = set(group[1])
        unique_predicates += len(predicate_no_dup)
        line = group[0] + "\t" + str(len(predicate_no_dup)) + "\n"
        file.write(line)
        num_entities += 1
    average = unique_predicates / (num_entities * 1.0)
    file.close()
    line = f"{type} \t {num_entities} \t {unique_predicates} \t {average}\n"
    types_file.write(line)

def get_connected_predicates(entities, sparql_endpoint, prefix, predicate_file):
    serliazied_entities = ""
    for entity in entities:
        entity = entity.strip()
        serliazied_entities += f"<{entity}> "
    query_subject = ("select distinct ?entity, ?p where {VALUES ?entity {"
                     f"{serliazied_entities}"
                     "} ?entity ?p ?o}")
    query_object = ("select distinct ?entity, ?p where {VALUES ?entity {"
                    f"{serliazied_entities} "
                    "} ?s ?p ?entity}")
    subject_result = utils.send_sparql_query(sparql_endpoint, query_subject)
    object_result = utils.send_sparql_query(sparql_endpoint, query_object)
    data = []
    output_list = []
    if subject_result is not None:
        output_list = subject_result['results']['bindings']

    if object_result is not None:
        output_list = output_list + object_result['results']['bindings']

    if len(output_list) > 0:
        for binding in output_list:
            entity = binding.get('entity', {}).get('value', None)
            predicate = binding.get('p', {}).get('value', None)
            if predicate and predicate not in utils.excluded_predicates:
                data.append({'entity': entity, 'predicate': predicate})
                predict_lock.acquire()
                if prefix not in predicate:
                    predicate_file.write(f'{predicate}\n')
                predict_lock.release()

    return data

def prepre_entities_for_predicate_retrieval(type, sparql_endpoint, prefix, types_file, predicate_file):
    human_readable_type = utils.get_file_name_from_type(type)
    file_name = f'data/{prefix}/{human_readable_type}.txt'
    file = open(file_name, 'r')
    lines = file.readlines()
    chunk_size = 50
    num_threads = 10
    results = []
    line_chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    num_completed_threads = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(get_connected_predicates, chunk, sparql_endpoint, prefix, predicate_file)
                   for chunk in line_chunks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.extend(result)
            num_completed_threads += 1
            if num_completed_threads % 250 == 0:
                time.sleep(2)

    file.close()
    write_result_to_file(type, results, prefix, types_file)


if __name__ == '__main__':
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
    type_file_path = f'index_data/{knowledge_graph_prefix}/average_per_type.txt'
    os.makedirs(os.path.dirname(type_file_path), exist_ok=True)
    types_file = open(type_file_path, 'w')
    types_file.write('Type \t Num_Entities \t Num_Predicates \t Average\n')

    predicate_file_path = f'index_data/{knowledge_graph_prefix}/predicates.txt'
    os.makedirs(os.path.dirname(predicate_file_path), exist_ok=True)
    predicate_file = open(predicate_file_path, 'w')

    distribution = utils.get_type_distrubution(knowledge_graph_uri, knowledge_graph_prefix)
    distribution = pd.DataFrame(distribution)
    types = distribution['Type'].values
    count = 0
    for type in types:
        print('Type ', type, 'Started')
        prepre_entities_for_predicate_retrieval(type, knowledge_graph_uri, knowledge_graph_prefix, types_file,
                                                predicate_file)
        count += 1
        if count % 100 == 0:
            time.sleep(2)
        print('Type ', type, 'Finished')