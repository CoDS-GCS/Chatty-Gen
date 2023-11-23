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

# GLOBAL MACROS
YAGO_ENDPOINTS = ["http://206.12.95.86:8892/sparql/"]
MAX_WAIT_TIME = 1.0

class YAGO:
    def __init__(self, _method="round-robin", _verbose=False, _db_name=0):
        # Explanation: selection_method is used to select from the YAGO_ENDPOINTS, hoping that we're not blocked too soon
        if _method in ["round-robin", "random", "select-one"]:
            self.selection_method = _method
        else:
            warnings.warn(
                "Selection method not understood, proceeding with 'select-one'"
            )
            self.selection_method = "select-one"

        self.verbose = _verbose
        self.sparql_endpoint = YAGO_ENDPOINTS[0]
        # self.r  = redis.StrictRedis(host='localhost', port=6379, db=_db_name)
        self.r = redis.Redis(host="localhost", port=6379, db=_db_name)
        # try:
        #     self.labels = pickle.load(open("resources/labels.pickle"))
        # except:
        #     print("Label Cache not found. Creating a new one")
        #     traceback.print_exc()
        #     # with open("resources/labels.pickle") as f:
        #     # self.labels = pickle.load(f)
        #     # lmultiform.merge_multiple_forms()			#This should populate the dictionary with multiple form info and already pickle it
        # self.fresh_labels = 0

    # initilizing the redis server.

    def select_sparql_endpoint(self):
        """
        This function is to be called whenever we're making a call to DBLP. Based on the selection mechanism selected at __init__,
        this function tells which endpoint to use at every point.
        """
        if self.selection_method == "round-robin":
            index = YAGO_ENDPOINTS.index(self.sparql_endpoint)
            return (
                YAGO_ENDPOINTS[index + 1]
                if index >= len(YAGO_ENDPOINTS)
                else YAGO_ENDPOINTS[0]
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
            LIMIT 1
        """
            % uri_or_literal
        )
        results = self.shoot_custom_query(label_query)
        if (
            "results" in results
            and "bindings" in results["results"]
            and len(results["results"]["bindings"]) > 0
        ):
            label = results["results"]["bindings"][0]["label"]["value"]
            return label
        return None

if __name__ == "__main__":
    yago = YAGO()
    query = """
    SELECT * WHERE {
      ?subject ?predicate ?object .
    }
    LIMIT 10
    """
    answer = yago.shoot_custom_query(query)
    print(answer)
