import sys
import os
from SPARQLWrapper import SPARQLWrapper, JSON
from operator import itemgetter
from pprint import pprint
import numpy as np
import traceback
import warnings
import pickle
import random
import redis
import json
from rdflib import Graph, URIRef, Literal, RDF
from kg.utils import (
    KgSchema,
    format_sparql_template_with_dict,
    SparqlQueryResponse,
    extract_values_by_key,
    seed_node_sparql_query,
    incoming_star_pattern_sparql_query,
    outgoing_star_pattern_sparql_query,
)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# GLOBAL MACROS
DBPEDIA_ENDPOINTS = ["http://dbpedia.org/sparql/", "http://live.dbpedia.org/sparql/"]
MAX_WAIT_TIME = 1.0


class DBPedia:
    def __init__(self, _method="round-robin", _verbose=False, _db_name=0):
        # Explanation: selection_method is used to select from the DBPEDIA_ENDPOINTS, hoping that we're not blocked too soon
        if _method in ["round-robin", "random", "select-one"]:
            self.selection_method = _method
        else:
            warnings.warn(
                "Selection method not understood, proceeding with 'select-one'"
            )
            self.selection_method = "select-one"

        self.verbose = _verbose
        self.sparql_endpoint = DBPEDIA_ENDPOINTS[0]
        # self.r  = redis.StrictRedis(host='localhost', port=6379, db=_db_name)
        self.r = redis.Redis(host="localhost", port=6379, db=_db_name)
        print(os.path.join(CURR_DIR, "dbpedia_rdf_schema.nt"))
        self.schema = KgSchema(
            os.path.join(CURR_DIR, "dbpedia_rdf_schema.nt"),
            rdf_format="nt",
            class_namespace="OWL",
        )

    def select_sparql_endpoint(self):
        """
        This function is to be called whenever we're making a call to DBPedia. Based on the selection mechanism selected at __init__,
        this function tells which endpoint to use at every point.
        """
        if self.selection_method == "round-robin":
            index = DBPEDIA_ENDPOINTS.index(self.sparql_endpoint)
            return (
                DBPEDIA_ENDPOINTS[index + 1]
                if index >= len(DBPEDIA_ENDPOINTS)
                else DBPEDIA_ENDPOINTS[0]
            )

        if self.selection_method == "select-one":
            return self.sparql_endpoint

    def shoot_custom_query(self, _custom_query):
        """
        Shoot any custom query and get the SPARQL results as a dictionary
        """
        caching_answer = self.r.get(_custom_query)
        if caching_answer:
            # print "@caching layer"
            return json.loads(caching_answer)
        sparql = SPARQLWrapper(self.select_sparql_endpoint())
        sparql.setQuery(_custom_query)
        sparql.setReturnFormat(JSON)
        caching_answer = sparql.query().convert()
        self.r.set(_custom_query, json.dumps(caching_answer))
        return caching_answer

    def get_label(self, uri_or_literal):
        if not isinstance(uri_or_literal, URIRef):
            return uri_or_literal
        label_query = (
            """
            SELECT ?label WHERE {
                <%s> <http://www.w3.org/2000/01/rdf-schema#label> ?label .
            }
        """
            % uri_or_literal
        )
        results = self.shoot_custom_query(label_query)
        label = None
        if (
            "results" in results
            and "bindings" in results["results"]
            and len(results["results"]["bindings"]) > 0
        ):
            all_bindings = results["results"]["bindings"]
            for r in all_bindings:
                if label is None:
                    if r["label"].get("xml:lang") is None:
                        label = r["label"]["value"]
                    elif r["label"].get("xml:lang") == "en":
                        label = r["label"]["value"]
                else:
                    break
            return label
        return None

    def get_triple_list(self):
        return self.schema.nodetype_triple_list

    def select_seed_nodes(self, n=10):
        """
        based on the classes from rdf schema and after filteration (whitelist)
        get seed nodes for these classes based on diversity criteria
        # TODO: fix the diversity as sparql query
        # TODO: do we need n or not ??
        """
        parsed_schema_map = self.schema.parsed_schema
        print(parsed_schema_map)
        self.seed_nodes = {}
        counter = 0
        for node_type, node_info in parsed_schema_map.items():
            if counter > 2:
                break
            counter += 1
            node_type_uri = node_info.get("nodeuri")
            incoming_predicates = [p[0] for p in node_info.get("incoming_predicates")]
            outgoing_predicates = [p[0] for p in node_info.get("outgoing_predicates")]

            query = seed_node_sparql_query(
                node_type_uri, incoming_predicates, outgoing_predicates
            )
            print(query)
            result = self.shoot_custom_query(query)
            response = SparqlQueryResponse(**result)
            seeds = extract_values_by_key("node", response)
            self.seed_nodes[node_type] = seeds
            print(f"extracted seed nodes: {len(seeds)}")
            print(f"extracted seed nodes: {seeds}")

        return self.seed_nodes

    def extract_subgraphs(self, seed_nodes):
        """
        using star pattern for now
        TODO: fix the other pattern later
        """
        # TODO: get prefixes for sparql query prepend
        prefixes = ""
        # for seed, seed_info in seeds.items():
        subgraphs_map = {}
        parsed_schema_map = self.schema.parsed_schema
        for node_type, node_info in parsed_schema_map.items():
            subgraphs_map[node_type] = []
            seeds = seed_nodes.get(node_type, [])

            for seed_node in seeds:
                # subject
                incoming_predicates = node_info.get("incoming_predicates")
                outgoing_predicates = node_info.get("outgoing_predicates")

                incoming_triples = []
                # seed as object
                predicate_subject_types = []
                for p in incoming_predicates:
                    print(p)
                    predicate, subject_type = p
                    predicate_subject_types.append((predicate, subject_type))
                # create query and execute and get back result and fill subgraph
                # incoming subgraph query
                object_uri = seed_node
                query = incoming_star_pattern_sparql_query(
                    prefixes, object_uri, predicate_subject_types
                )
                print(f"Query: {query}")
                response = self.shoot_custom_query(query)
                print(response)
                response = SparqlQueryResponse(**response)
                subs = extract_values_by_key("subject", response)
                preds = extract_values_by_key("predicate", response)
                for pred, sub in zip(preds, subs):
                    for predicate, subject_type in predicate_subject_types:
                        if predicate == pred:
                            type_ = subject_type
                            break
                    print(pred, type_, sub)
                    # pred = self.get_label(pred)
                    # type_ = self.get_label(type_)
                    # sub = self.get_label(sub)
                    incoming_triples.append((pred, type_, sub))

                print(f"seed : {seed_node}")
                print(f"incoming : {incoming_triples}")

                outgoing_triples = []
                # seed as subject
                predicate_object_types = []
                for p in outgoing_predicates:
                    predicate, object_type = p
                    predicate_object_types.append((predicate, object_type))
                # create query and execute and get back result and fill subgraph
                # outgoing subgraph query
                subject_uri = seed_node
                query = outgoing_star_pattern_sparql_query(
                    prefixes, subject_uri, predicate_object_types
                )
                print(f"Query: {query}")
                response = self.shoot_custom_query(query)
                response = SparqlQueryResponse(**response)
                objs = extract_values_by_key("object", response)
                preds = extract_values_by_key("predicate", response)
                for pred, obj in zip(preds, objs):
                    for predicate, object_type in predicate_object_types:
                        if predicate == pred:
                            type_ = object_type
                            break
                    print(pred, type_, obj)
                    # pred = self.get_label(pred)
                    # type_ = self.get_label(type_)
                    # obj = self.get_label(obj)
                    outgoing_triples.append((pred, type_, obj))

                print(f"seed : {seed_node}")
                print(f"outgoing: {outgoing_triples}")

                seed_type = node_type
                seed_label = self.get_label(seed_node)
                subgraph = {
                    "seed_node": (seed_node, seed_label, seed_type),
                    "incoming_predicates": incoming_triples,
                    "outgoing_predicates": outgoing_triples,
                }
                subgraphs_map[node_type].append(subgraph)
        return subgraphs_map


if __name__ == "__main__":
    dbpedia = DBPedia()
    query = """
    SELECT * WHERE {
      ?subject ?predicate ?object .
    }
    LIMIT 10
    """
    # answer = dbpedia.shoot_custom_query(query)
    # print(answer)
