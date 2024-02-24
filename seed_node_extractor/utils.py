import requests
import re
import time

excluded_predicates = ['http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://www.w3.org/2002/07/owl#sameAs',
                       'http://schema.org/image', 'http://schema.org/sameAs',
                       'http://www.w3.org/2000/01/rdf-schema#comment', 'http://schema.org/logo',
                       'http://schema.org/url', 'http://www.w3.org/2002/07/owl#differentFrom',
                       'http://www.w3.org/1999/02/22-rdf-syntax-ns#first',
                       'http://purl.org/spar/datacite/hasIdentifier', 'http://www.w3.org/2000/01/rdf-schema#seeAlso',
                       'http://xmlns.com/foaf/0.1/thumbnail', 'http://www.w3.org/2002/07/owl#differentFrom',
                       'http://xmlns.com/foaf/0.1/isPrimaryTopicOf', 'http://purl.org/dc/elements/1.1/type',
                       'http://xmlns.com/foaf/0.1/primaryTopic', 'http://xmlns.com/foaf/0.1/logo',
                       'http://purl.org/dc/elements/1.1/rights', 'http://www.w3.org/2000/01/rdf-schema#label',
                       'http://dbpedia.org/ontology/thumbnail']

knowledge_graph_to_uri = {
    "dbpedia": ("http://206.12.95.86:8890/sparql", "dbpedia"),
    # "lc_quad": "http://206.12.95.86:8891/sparql",
    "microsoft_academic": ("http://206.12.97.159:8890/sparql", "makg"),
    "yago": ("http://206.12.95.86:8892/sparql", "schema"),
    "dblp": ("http://206.12.95.86:8894/sparql", "dblp"),
}

# Returns only KG specific types
def sparql_results_to_dataframe(results, kg):
    data = []

    for binding in results['results']['bindings']:
        type = binding.get('type', {}).get('value', None)
        count = binding.get('count', {}).get('value', None)
        if kg in type and type not in ['http://dbpedia.org/ontology/Image', 'http://schema.org/GeoCoordinates', 'http://dbpedia.org/ontology/CareerStation']:
            data.append({'Type': type, 'Count': count})

    return data

def execute_sparql_query(endpoint_url, query):
    headers = {
        'Content-Type': 'application/sparql-query',
        'Accept': 'application/json'
    }

    params = {
        'query': query,
    }

    response = requests.get(endpoint_url, headers=headers, params=params)

    if response.status_code == 200:
        return response.status_code, response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return response.status_code, response.text

def send_sparql_query(endpoint_url, query):
    repeat = True
    count = 0
    while repeat:
        status_code, response = execute_sparql_query(endpoint_url, query)
        if status_code == 200:
            return response
        elif status_code == 404 and 'The requested URL was not found    URI  = \'/sparql\'' in response:
            count += 1
            print("Retrying ", count)
            time.sleep(2)

        else:
            return None


def get_type_distrubution(endpoint, prefix):
    query = ("SELECT ?type (COUNT(?entity) AS ?count) WHERE { ?entity rdf:type ?type. } GROUP BY ?type "
             "ORDER BY DESC(?count)")
    result = send_sparql_query(endpoint, query)
    result_df = sparql_results_to_dataframe(result, prefix)
    return result_df

def get_name(predicate):
    pattern = r'[#/]([^/#]+)$'
    match = re.search(pattern, predicate)
    if match:
        name = match.group(1)
        p2 = re.compile(r"([a-z0-9])([A-Z])")
        name = p2.sub(r"\1 \2", name)
        return name
    else:
        return ""


def get_file_name_from_type(type):
    type = type.strip()
    human_readable_type = get_name(type)
    human_readable_type = human_readable_type.replace(' ', '_')
    return human_readable_type