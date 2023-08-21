import random
import asyncio
import re
import time
import jsonlines
from rdflib import Graph, URIRef, Literal, RDF
from rdflib.namespace import Namespace
from SPARQLWrapper import SPARQLWrapper, JSON

# from urllib.parse import quote
from collections import deque

# Define your RDF dump file or SPARQL endpoint URL
rdf_dump_file = None
sparql_endpoint_url = "http://206.12.95.86:8894/sparql/"

num_iterations = 100  # Set the number of iterations

# Number of nodes to include in the subgraph
subgraph_size = 10

# Define the number of random seed nodes you want
num_random_seed_nodes = 50


# Function to load RDF data from a file or SPARQL endpoint
async def load_rdf_data(rdf_graph):
    # Get a list of all available nodes from the RDF dump or SPARQL endpoint
    all_nodes = set()
    if rdf_dump_file:
        for subj, _, _ in rdf_graph:
            all_nodes.add(subj)
    else:
        sparql = SPARQLWrapper(sparql_endpoint_url)
        sparql.setQuery(
            """
            CONSTRUCT {
                ?person a <https://dblp.org/rdf/schema#Person> .
                ?paper <https://dblp.org/rdf/schema#authoredBy> ?person .
                ?person <https://dblp.org/rdf/schema#primaryAffiliation> ?affiliation .
                } WHERE {
                    ?person a <https://dblp.org/rdf/schema#Person> .
                    ?paper <https://dblp.org/rdf/schema#authoredBy> ?person .
                    ?person <https://dblp.org/rdf/schema#primaryAffiliation> ?affiliation .
                    } LIMIT 10000
        """
        )
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        for result in results["results"]["bindings"]:
            subject_value = result["s"]["value"]
            predicate_value = result["p"]["value"]
            obj_value = result["o"]["value"]

            # Check if the value is a URI or a literal
            if result["s"]["type"] == "uri":
                subject = URIRef(subject_value)
            else:
                subject = Literal(subject_value)

            if result["p"]["type"] == "uri":
                predicate = URIRef(predicate_value)
            else:
                predicate = Literal(predicate_value)

            if result["o"]["type"] == "uri":
                obj = URIRef(obj_value)
            else:
                obj = Literal(obj_value)

            rdf_graph.add((subject, predicate, obj))

    # print(len(rdf_graph))


# Define a function to retrieve triples for a given node
def bfs_coroutine(rdf_graph, max_triples=100, max_nodes=50):
    queue = deque()
    visited_nodes = set()

    while True:
        node_uri = yield
        queue.append(node_uri)
        while queue and len(visited_nodes) < max_nodes:
            current_node = queue.popleft()
            if current_node not in visited_nodes:
                visited_nodes.add(current_node)
                triples = list(rdf_graph.triples((current_node, None, None)))
                triples += list(rdf_graph.triples((None, None, current_node)))
                for triple in triples:
                    if len(visited_nodes) >= max_nodes:
                        break
                    queue.append(triple[2])
                    yield triple


def dfs_coroutine(rdf_graph, max_triples=100, max_nodes=50):
    stack = []
    visited_nodes = set()

    while True:
        node_uri = yield
        stack.append((node_uri, 0))
        while stack and len(visited_nodes) < max_nodes:
            current_node, depth = stack.pop()
            if current_node not in visited_nodes:
                visited_nodes.add(current_node)
                triples = list(rdf_graph.triples((current_node, None, None)))
                triples += list(rdf_graph.triples((None, None, current_node)))
                for triple in triples:
                    if len(visited_nodes) >= max_nodes:
                        break
                    if depth < 2:
                        stack.append((triple[2], depth + 1))
                    yield triple


def get_triples_for_node(rdf_graph, node_uri, traversal="bfs", max_triples=100, max_nodes=50):
    if traversal == "bfs":
        coroutine = bfs_coroutine(rdf_graph, max_triples, max_nodes)
    elif traversal == "dfs":
        coroutine = dfs_coroutine(rdf_graph, max_triples, max_nodes)
    else:
        raise ValueError("Invalid traversal option")

    next(coroutine)  # Start the coroutine

    triples = []
    try:
        while len(triples) < max_triples:
            triple = coroutine.send(node_uri)
            if triple:
                triples.append(triple)
            else:
                break
    except StopIteration:
        pass

    return triples


# Function to retrieve labels for a given URI
async def get_label(uri_or_literal):
    if not isinstance(uri_or_literal, URIRef):
        return uri_or_literal
    label_query = (
        """
        SELECT ?label WHERE {
            <%s> <http://www.w3.org/2000/01/rdf-schema#label> ?label .
        }
        LIMIT 1
    """
        % uri_or_literal
    )
    sparql = SPARQLWrapper(sparql_endpoint_url)
    sparql.setQuery(label_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if (
        "results" in results
        and "bindings" in results["results"]
        and len(results["results"]["bindings"]) > 0
    ):
        label = results["results"]["bindings"][0]["label"]["value"]
        return label
    return None


# Function to extract labels for a triple
def get_labels_for_triple(triple):
    subject, predicate, obj = triple
    subject_label = get_label(subject) or str(subject)
    predicate_label = get_label(predicate) or str(predicate)
    obj_label = get_label(obj) or str(obj)
    return subject_label, predicate_label, obj_label


def get_seed_subjects(rdf_graph):
    DBLP = Namespace("https://dblp.org/rdf/schema#")

    # Get a list of all subjects from the RDF graph
    seed_subjects = set()
    for sub, pred, obj in rdf_graph:
        if pred == RDF.type and obj == DBLP.Person:
            seed_subjects.add(sub)

    # Randomly select seed subjects
    seed_subjects = random.sample(seed_subjects, num_random_seed_nodes)
    # print(seed_subjects)
    return seed_subjects


# Function to extract labels for a triple
async def get_labels_for_triple_async(triple):
    subject, predicate, obj = triple
    subject_label = await get_label(subject) or str(subject)
    predicate_label = await get_label(predicate) or str(predicate)
    obj_label = await get_label(obj) or str(obj)
    return subject_label, predicate_label, obj_label


async def get_triples_and_labels(rdf_graph, seed_node):
    subgraph_triples = get_triples_for_node(rdf_graph, seed_node, traversal="bfs")
    # print("subgraph triples ----\n", subgraph_triples)
    labeled_subgraph = []

    for triple in subgraph_triples:
        labels = await get_labels_for_triple_async(triple)
        labeled_subgraph.append((labels[0], labels[1], labels[2]))

    return labeled_subgraph


# Generate subgraphs for seed nodes asynchronously
async def generate_subgraphs_async(rdf_graph, seed_subjects):
    print("Generating subgraphs asynchronously...")
    start_time = time.time()
    subgraphs = {}
    tasks = []

    for seed_node in seed_subjects:
        task = asyncio.create_task(get_triples_and_labels(rdf_graph, seed_node))
        tasks.append(task)

    # Gather all tasks
    subgraphs_results = await asyncio.gather(*tasks)

    for seed_node, triples in zip(seed_subjects, subgraphs_results):
        subgraphs[seed_node] = triples

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Subgraphs generated asynchronously in {elapsed_time:.2f} seconds.")
    return subgraphs


url_pattern = re.compile(r"^\w+://")


# Function to filter and write subgraphs to a file
async def write_subgraphs_to_file(subgraphs, file_name):
    with open(file_name, "a") as file:
        for seed_node, triples in subgraphs.items():
            filtered_triplets = []
            for subject, predicate, obj in triples[:subgraph_size]:
                # Check if the predicate is a URL pattern
                if not url_pattern.match(str(predicate)):
                    filtered_triplets.append((subject, predicate, obj))
            for triple in filtered_triplets:
                # print(f"{triple[0]} {triple[1]} {triple[2]}\n")
                file.write(f"{triple[0]} {triple[1]} {triple[2]}\n")
            file.write("\n---\n")


# Function to write subgraphs to a JSONL file using jsonlines library
def write_subgraphs_to_jsonl(subgraphs):
    with jsonlines.open("subgraphs.jsonl", mode="a") as writer:
        for seed_node, triples in subgraphs.items():
            filtered_triplets = []

            for subject, predicate, obj in triples[:subgraph_size]:
                # Check if the predicate is a URL pattern
                if not url_pattern.match(str(predicate)):
                    filtered_triplets.append((subject, predicate, obj))

            subgraph_data = {
                "seed_node": seed_node,
                "triples": [
                    {"subject": str(s), "predicate": str(p), "object": str(o)}
                    for s, p, o in filtered_triplets
                ]
            }

            writer.write(subgraph_data)


async def generate():
    rdf_graph = Graph()
    await load_rdf_data(rdf_graph)
    seed_subjects = get_seed_subjects(rdf_graph)
    subgraphs = await generate_subgraphs_async(rdf_graph, seed_subjects)

    # Write subgraphs to separate text files
    tasks = []
    for idx, (seed_node, triples) in enumerate(subgraphs.items()):
        file_name = f"subgraphs_{idx}.txt"
        task = asyncio.create_task(write_subgraphs_to_file(subgraphs, file_name))
        tasks.append(task)

    # Gather all file writing tasks
    await asyncio.gather(*tasks)

    # Write subgraphs to a JSONL file
    write_subgraphs_to_jsonl(subgraphs)


# Run the asyncio event loop for multiple iterations
for _ in range(num_iterations):
    asyncio.run(generate())
