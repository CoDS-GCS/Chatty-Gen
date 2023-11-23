import utils.graph_utils as graph_utils
import kg.dblp as dblp
import kg.yago as yago
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from rdflib import Graph, URIRef, Literal, RDF
import json
import yaml
import logging


def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_config(config_file_path):
    with open(config_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


# subgraph extractor logic
"""
implement a star pattern
- use utils - we got insert and access_graph
"""

# KG interface

# 10 star subgraphs using some schema

one_triple_right = """
            SELECT DISTINCT ?p ?e
            WHERE {
                <%(e)s> ?p ?e.
            }"""

one_triple_left = """
            SELECT DISTINCT ?e ?p ?type
            WHERE {
                ?e ?p <%(e)s>.
            }"""

out_predicate_list = """
            SELECT DISTINCT ?p
            WHERE {
                <%(e)s> ?p ?e.
            }"""

get_predicate = """
            SELECT DISTINCT ?o
            WHERE {
                <%(e)s> <%(p)s> ?o.
            }"""

# DBLP test done
# kg = dblp.DBLP()
# e = "https://dblp.org/pid/34/7659"
# answer = kg.shoot_custom_query(one_triple_right % {'e': e})
# print(answer)

# YAGO test done
# kg = yago.YAGO()
# e = "http://schema.org/Person"
# answer = kg.shoot_custom_query(out_predicate_list % {'e': e})
# print(answer)

"""
define the schema as dictonary with 3 things in dictionary
"""
dblp_schema_info = {
    "Creator": {
        "nodetype": "https://dblp.org/rdf/schema#Creator",
        "in_predicates": [
            "dblp:coCreatorWith",
            "dblp:coAuthorWith",
            "dblp:coEditorWith",
            "dblp:homonymousCreator",
            "dblp:possibleActualCreator",
            "dblp:signatureCreator",
            "dblp:createdBy",
            "dblp:authoredBy",
            "dblp:editedBy",
        ],
        "out_predicates": [
            "dblp:orcid",
            "dblp:creatorName",
            "dblp:primaryCreatorName",
            "dblp:creatorNote",
            "dblp:affiliation",
            "dblp:primaryAffiliation",
            "dblp:awardWebpage",
            "dblp:homepage",
            "dblp:primaryHomepage",
            "dblp:creatorOf",
            "dblp:authorOf",
            "dblp:editorOf",
            "dblp:coCreatorWith",
            "dblp:coAuthorWith",
            "dblp:coEditorWith",
            "dblp:homonymousCreator",
            "dblp:proxyAmbiguousCreator",
        ],
    },
    "Publication": {
        "nodetype": "https://dblp.org/rdf/schema#Publication",
        "in_predicates": [
            "dblp:creatorOf",
            "dblp:authorOf",
            "dblp:editorOf",
            "dblp:signaturePublication",
            "dblp:publishedAsPartOf",
        ],
        "out_predicates": [
            "dblp:doi",
            "dblp:isbn",
            "dblp:title",
            "dblp:bibtexType",
            "dblp:createdBy",
            "dblp:authoredBy",
            "dblp:editedBy",
            "dblp:numberOfCreators",
            "dblp:hasSignature",
            "dblp:documentPage",
            "dblp:primaryDocumentPage",
            "dblp:listedOnTocPage",
            "dblp:publishedIn",
            "dblp:publishedInSeries",
            "dblp:publishedInSeriesVolume",
            "dblp:publishedInJournal",
            "dblp:publishedInJournalVolume",
            "dblp:publishedInJournalVolumeIssue",
            "dblp:publishedInBook",
            "dblp:publishedInBookChapter",
            "dblp:pagination",
            "dblp:yearOfEvent",
            "dblp:yearOfPublication",
            "dblp:monthOfPublication",
            "dblp:publishedBy",
            "dblp:publishersAddress",
            "dblp:thesisAcceptedBySchool",
            "dblp:publicationNote",
            "dblp:publishedAsPartOf",
        ],
    },
}

yago_schema_info = {
    "Person": {
        "nodetype": "http://schema.org/Person",
        "in_predicates": [
            "yago:studentOf",
            "yago:doctoralAdvisor",
            "schema:leader",
            "schema:illustrator",
            "schema:editor",
            "schema:performer",
            "schema:actor",
            "schema:director",
            "schema:founder",
            "schema:leader",
            "yago:director",
            "schema:children",
            "schema:spouse",
            "schema:actor",
            "schema:director",
        ],
        "out_predicates": [
            "schema:affiliation",
            "schema:alumniOf",
            "schema:award",
            "schema:birthDate",
            "schema:birthPlace",
            "schema:children",
            "schema:deathDate",
            "schema:deathPlace",
            "schema:gender",
            "schema:homeLocation",
            "schema:knowsLanguage",
            "schema:memberOf",
            "schema:nationality",
            "schema:owns",
            "schema:spouse",
            "schema:worksFor",
            "yago:academicDegree",
            "yago:beliefSystem",
        ],
    },
    "Organization": {
        "nodetype": "http://schema.org/Organization",
        "in_predicates": [
            "schema:memberOf",
            "schema:productionCompany",
            "schema:recordLabel",
            "schema:memberOf",
            "schema:affiliation",
            "schema:worksFor",
            "schema:alumniOf",
            "schema:memberOf",
            "schema:productionCompany",
        ],
        "out_predicates": [
            "schema:address",
            "schema:award",
            "schema:dateCreated",
            "schema:duns",
            "schema:founder",
            "schema:leader",
            "schema:leiCode",
            "schema:location",
            "schema:locationCreated",
            "schema:logo",
            "schema:memberOf",
            "schema:motto",
            "schema:numberOfEmployees",
            "schema:ownedBy",
        ],
    },
    "actor": {"nodetype": "", "in_predicates": [], "out_predicates": []},
}


def format_template_with_dict(template, values_dict):
    try:
        formatted_string = template % values_dict
        return formatted_string
    except KeyError as e:
        return f"Error: Missing key '{e.args[0]}' in the dictionary."
    except ValueError:
        return "Error: Type mismatch in the template."
    except Exception as e:
        return f"An error occurred: {str(e)}"


# first get seed nodes of any one type first person

e = "http://schema.org/Person"
p = "http://schema.org/address"

sparql_templates = {
    "get_seed_nodes": """
        SELECT DISTINCT ?sub
        WHERE {
            ?sub ?type <%(e)s>.
        }
        LIMIT 10
        """,
    "get_seed_nodes_popular": """
        SELECT DISTINCT ?sub (COUNT(?predicate) AS ?predicateCount)
        WHERE {
          ?sub a <http://schema.org/Person> .
          ?sub ?predicate ?object .
        }
        GROUP BY ?sub
        ORDER BY DESC(?predicateCount)
        LIMIT 10
        """,
    "get_seed_nodes_popular_2": """
        SELECT DISTINCT ?node (COUNT(?outgoingPredicate) + COUNT(?incomingPredicate) AS ?predicateCount)
        WHERE {
              ?node a <%(e)s>.
              ?node ?outgoingPredicate ?object.
              ?subject ?incomingPredicate ?node.
        }
        GROUP BY ?node
        ORDER BY DESC(?predicateCount)
        LIMIT 100
        """,
}


spo = """<%(sub)s> <%(pred)s> ?obj"""
# spo % {}

# first complete the get_seed_nodes, next what
# we have seeds, collect the out_going edges and incoming_edges for all the predicates


@dataclass
class SparqlQueryResponse:
    head: Dict[str, List[str]]
    results: Dict[str, List[Dict[str, Optional[str]]]]


# def extract_values_by_key(
#     key: str, sparql_response: SparqlQueryResponse
# ) -> List[Optional[str]]:
#     values = []
#
#     # Check if the key exists in the response
#     if key in sparql_response.head.get("vars", []):
#         for binding in sparql_response.results.get("bindings", []):
#             value = binding.get(key, {}).get("value")
#             values.append(value)
#
#     return values


def extract_values_by_key(
    key: str, sparql_response: SparqlQueryResponse
) -> List[Optional[str]]:
    values = []

    # Check if the key exists in the response
    if key in sparql_response.head.get("vars", []):
        for binding in sparql_response.results.get("bindings", []):
            binding_value = binding.get(key, {})
            value_type = binding_value.get("type", None)
            value_ = binding_value.get("value", None)
            if value_type == "uri":
                value = URIRef(value_)
            else:
                value = Literal(value_)
                # print(type(value))
            values.append(value)

    return values


# answer = yago.shoot_custom_query(get_predicate % {'e': e, 'p': p})
# query_response = yago.shoot_custom_query(get_seed_nodes % {'e': e, 'p': p})
# sparql_response = SparqlQueryResponse(
#     head=query_response.get("head", {}), results=query_response.get("results", {})
# )
# subs = extract_values_by_key("sub", sparql_response)
# print(subs)


def generate_dynamic_sparql_query(subject_uri, predicates_list):
    # Construct the PREFIX part of the query
    prefixes = """
    PREFIX schema: <http://schema.org/>
    PREFIX yago: <http://yago-knowledge.org/resource/>
    """

    # Initialize an empty SPARQL query
    query = prefixes + "SELECT ?predicate ?object WHERE { VALUES ?predicate {"

    # Add each predicate from the list to the VALUES block
    for predicate in predicates_list:
        query += f"{predicate} "

    query += f"}} FILTER EXISTS {{ <{subject_uri}> ?predicate ?object }}}}"

    return query


def outgoing_star_pattern_sparql_query(subject_uri, predicates_list):
    # Construct the PREFIX part of the query
    prefixes = """
    PREFIX schema: <http://schema.org/>
    PREFIX yago: <http://yago-knowledge.org/resource/>
    """

    # Initialize the SPARQL query
    query = prefixes + "SELECT DISTINCT ?predicate ?object WHERE {\n"

    # Add the VALUES block for predicates
    query += "  VALUES ?predicate {\n"
    for predicate in predicates_list:
        query += f"{predicate} "
    query += "  }\n"

    # Add the subquery for the subject and objects
    query += "  {\n"
    query += f"    SELECT ?predicate ?object\n"
    query += "    WHERE {\n"
    query += f"      <{subject_uri}> ?predicate ?object .\n"
    query += "    }\n"
    query += "  }\n"

    # Close the main query
    query += "}\n"

    return query


person_examples = ["http://yago-knowledge.org/resource/Marcia_Furnilla"]
subject_uri = person_examples[0]

# sparql_query = outgoing_star_pattern_sparql_query(
#     subject_uri, person.get("out_predicates")
# )
# print(sparql_query)


def incoming_star_pattern_sparql_query(object_uri, predicates_list):
    # Construct the PREFIX part of the query
    prefixes = """
    PREFIX schema: <http://schema.org/>
    PREFIX yago: <http://yago-knowledge.org/resource/>
    """

    # Initialize the SPARQL query
    query = prefixes + "SELECT DISTINCT ?subject ?predicate WHERE {\n"

    # Add the VALUES block for predicates
    query += "  VALUES ?predicate {\n"
    for predicate in predicates_list:
        query += f"{predicate} "
    query += "  }\n"

    # Add the subquery for the subject and objects
    query += "  {\n"
    query += f"    SELECT ?subject ?predicate \n"
    query += "    WHERE {\n"
    query += f"      ?subject ?predicate <{object_uri}> .\n"
    query += "    }\n"
    query += "  }\n"

    # Close the main query
    query += "}\n"

    return query


person_examples = ["http://yago-knowledge.org/resource/Marcia_Furnilla"]
subject_uri = person_examples[0]

# sparql_query = incoming_star_pattern_sparql_query(
#     subject_uri, person.get("in_predicates")
# )
# print(sparql_query)


def get_label(kg, uri_or_literal):
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
    results = kg.shoot_custom_query(label_query)
    if (
        "results" in results
        and "bindings" in results["results"]
        and len(results["results"]["bindings"]) > 0
    ):
        label = results["results"]["bindings"][0]["label"]["value"]
        return label
    return None


# if __name__ == "__main__":
#     # generate subgraphs
#
#     # step 1 : select knowledge graph
#     kg = yago.YAGO()
#     # step 2 : get seed nodes (person)
#     seed_type = "http://schema.org/Person"
#     # query_template = sparql_templates["get_seed_nodes"]
#     query_template = sparql_templates["get_seed_nodes_popular_2"]
#     values = {"e": seed_type}
#     query = format_template_with_dict(query_template, values)
#     print(query)
#     seed_nodes = kg.shoot_custom_query(query)
#     response = SparqlQueryResponse(**seed_nodes)
#     seed_nodes = extract_values_by_key("node", response)
#
#     # step 3: create star patterns subgraphs.
#     # query_template = sparql_templates['incoming_']
#     # for seed_node in person_examples:
#     for seed_node in seed_nodes:
#         triples = []
#         seed_node_type_info = schema_info.get("Person")
#         query = incoming_star_pattern_sparql_query(
#             seed_node, seed_node_type_info.get("in_predicates")
#         )
#         # print(query)
#         # execute this query and parse the results as triples, missing label extraction
#         response = kg.shoot_custom_query(query)
#         response = SparqlQueryResponse(**response)
#         subs = extract_values_by_key("subject", response)
#         preds = extract_values_by_key("predicate", response)
#         for sub, pred in zip(subs, preds):
#             triples.append((sub, pred, seed_node))
#
#         query = outgoing_star_pattern_sparql_query(
#             seed_node, seed_node_type_info.get("out_predicates")
#         )
#         # print(query)
#         # execute this query and parse the results as triples, missing label extraction
#         response = kg.shoot_custom_query(query)
#         # print(response)
#         response = SparqlQueryResponse(**response)
#         preds = extract_values_by_key("predicate", response)
#         # print("extracting objects")
#         objs = extract_values_by_key("object", response)
#         for pred, obj in zip(preds, objs):
#             triples.append((seed_node, pred, obj))
#
#         subgraph = []
#         for triple in triples:
#             subgraph.append(
#                 {
#                     "subject": kg.get_label(triple[0]),
#                     "predicate": kg.get_label(triple[1]),
#                     "object": kg.get_label(triple[2]),
#                 }
#             )
#
#         print(subgraph)
#         with open("subgraphs.jsonl", "a") as f:
#             data = {"seed_node": seed_node, "triples": subgraph}
#             json.dump(data, f)
#             f.write("\n")
#     # label = get_label(kg, URIRef("http://yago-knowledge.org/resource/Aarti_Puri"))
#


def generate_subgraphs(kg, output_file, seed_node_types, schema_info):
    # generate subgraphs

    # step 1 : select knowledge graph
    kg = yago.YAGO()
    for seed_node_type in seed_node_types:
        # step 2 : get seed nodes (person)
        seed_node_info = schema_info.get(seed_node_type)
        seed_type_uri = seed_node_info['nodetype']
        # query_template = sparql_templates["get_seed_nodes"]
        query_template = sparql_templates["get_seed_nodes_popular_2"]
        values = {"e": seed_type_uri}
        query = format_template_with_dict(query_template, values)
        print(query)
        seed_nodes = kg.shoot_custom_query(query)
        response = SparqlQueryResponse(**seed_nodes)
        seed_nodes = extract_values_by_key("node", response)
        print(f"extracted seed nodes: {len(seed_nodes)}")

        # step 3: create star patterns subgraphs.
        # query_template = sparql_templates['incoming_']
        # for seed_node in person_examples:
        for seed_node in seed_nodes:
            triples = []
            query = incoming_star_pattern_sparql_query(
                seed_node, seed_node_info.get("in_predicates")
            )
            # print(query)
            # execute this query and parse the results as triples, missing label extraction
            response = kg.shoot_custom_query(query)
            response = SparqlQueryResponse(**response)
            subs = extract_values_by_key("subject", response)
            preds = extract_values_by_key("predicate", response)
            for sub, pred in zip(subs, preds):
                triples.append((sub, pred, seed_node))

            query = outgoing_star_pattern_sparql_query(
                seed_node, seed_node_info.get("out_predicates")
            )
            # print(query)
            # execute this query and parse the results as triples, missing label extraction
            response = kg.shoot_custom_query(query)
            # print(response)
            response = SparqlQueryResponse(**response)
            preds = extract_values_by_key("predicate", response)
            # print("extracting objects")
            objs = extract_values_by_key("object", response)
            for pred, obj in zip(preds, objs):
                triples.append((seed_node, pred, obj))

            subgraph = []
            for triple in triples:
                subgraph.append(
                    {
                        "subject": kg.get_label(triple[0]),
                        "predicate": kg.get_label(triple[1]),
                        "object": kg.get_label(triple[2]),
                    }
                )

            print(subgraph)
            with open(output_file, "a") as f:
                data = {"seed_node": seed_node, "triples": subgraph}
                json.dump(data, f)
                f.write("\n")
    # label = get_label(kg, URIRef("http://yago-knowledge.org/resource/Aarti_Puri"))


def get_kg_instance(kg):
    if kg == "yago":
        return yago.YAGO()
    elif kg == "dblp":
        return dblp.DBLP()
    else:
        raise ValueError(f"Unsupported kg: {kg}")


def get_kg_schema_info(kg):
    if kg == "yago":
        return yago_schema_info
    elif kg == "dblp":
        return dblp_schema_info
    else:
        raise ValueError(f"Unsupported kg: {kg}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    # Load configuration from YAML file
    config = load_config(config_file)

    # Setup logging
    setup_logging(config["log_file"])

    generate_subgraphs(
        kg=get_kg_instance(config["kg"]),
        output_file=config["output_subgraph_file"],
        seed_node_types=config["seed_node_types"],
        schema_info=get_kg_schema_info(config["kg"]),
    )

###
# - finish the dialogue generation by using subgraph generated here
# - get the query for important seed nodes
# - problem will be in dblp where we don't have many predicates

# subjects = extract_values_by_key("pred", response)

# # Example usage:
# json_response = {
#     "head": {"vars": ["sub", "pred", "obj"]},
#     "results": {
#         "bindings": [
#             {"sub": {"type": "uri", "value": "http://example.org/subject"}, "pred": {"type": "uri", "value": "http://example.org/predicate"}, "obj": {"type": "uri", "value": "http://example.org/object"}},
#             # Additional bindings...
#         ]
#     }
# }
#
# response = SparqlQueryResponse(
#     head=json_response.get("head", {}),
#     results=json_response.get("results", {})
# )
#
# subjects = extract_values_by_key("sub", response)
# print(subjects)
# predicates = extract_values_by_key("pred", response)
# print(predicates)
#


# - first we will have both in and out predicates on node
# - extrct star pattern

# class Node(BaseModel):
#     nodetype: str
#     in_predicates: List[Tuple[str, 'Node']]
#     out_predicates: List[Tuple[str, 'Node']]
#
# organization = Node(
#         nodetype="Organization",
#         in_predicates=[()],
#         out_predicates=[()])
#
# actor = Node(
#         nodetype="Actor",
#         in_predicates=[()],
#         out_predicates=[("movie", person)])
#
# person = Node(
#         nodetype="Person",
#         in_predicates=[("movie", actor)],
#         out_predicates=[()])
