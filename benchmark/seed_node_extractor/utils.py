import requests
import re
import redis
import time
import json
from appconfig import config
from redis_util import RedisClient

try:
    redis_client = RedisClient(config.redis_url)
    if not redis_client.ping():
        redis_client = None
except Exception as e:
    print("could not create redis client", e)
    redis_client = None





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
                       'http://dbpedia.org/ontology/thumbnail', 'http://dbpedia.org/ontology/wikiPageID',
                       'http://purl.org/dc/terms/subject', 'http://purl.org/linguistics/gold/hypernym',
                       'http://xmlns.com/foaf/0.1/name', 'http://dbpedia.org/ontology/wikiPageRevisionID',
                       'http://dbpedia.org/ontology/wikiPageRedirects', 'http://www.w3.org/ns/prov#wasDerivedFrom',
                       'http://dbpedia.org/ontology/wikiPageExternalLink', 'http://dbpedia.org/ontology/abstract',
                       'http://xmlns.com/foaf/0.1/depiction']

host = config.kghost
knowledge_graph_to_uri = {
    "dbpedia": (f"http://{host}:8890/sparql", "dbpedia"),
    # "lc_quad": "http://206.12.95.86:8891/sparql",
    "microsoft_academic": (f"http://{host}:8890/sparql", "makg"),
    "yago": (f"http://{host}:8892/sparql", "schema"),
    "dblp": (f"http://{host}:8894/sparql", "dblp"),
}

# Returns only KG specific types
def sparql_results_to_dataframe(results, kg):
    data = []

    for binding in results['results']['bindings']:
        type = binding.get('type', {}).get('value', None)
        count = binding.get('count', {}).get('value', None)
        if (kg in type and type not in
                ['http://dbpedia.org/ontology/Image', 'http://schema.org/GeoCoordinates',
                 'http://dbpedia.org/ontology/CareerStation', 'http://dbpedia.org/ontology/TimePeriod',
                 'http://bioschemas.org/Taxon']):
            data.append({'Type': type, 'Count': count})

    return data


def execute_sparql_query(endpoint_url, query):
    global redis_client

    # Check if the response is cached
    cache_key = f"{endpoint_url}_{query}"
    if redis_client:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            # print("Result found in cache.")
            return 200, cached_result

    headers = {
        'Content-Type': 'application/sparql-query',
        'Accept': 'application/json'
    }
    params = {
        'query': query,
    }

    response = requests.get(endpoint_url, headers=headers, params=params)

    if response.status_code == 200:
        # Cache the response
        if redis_client:
            redis_client.set(cache_key, json.dumps(response.json()))
            print("Result cached.")
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
