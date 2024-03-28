import os
import warnings
import random
import re
import time
from collections import defaultdict
import redis
import json
from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from rdflib.namespace import Namespace
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from SPARQLWrapper import SPARQLWrapper, JSON
from logger import Logger
from appconfig import config
from seed_node_extractor import utils

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

host = config.kghost
redis_host = config.redishost

# Custom URI type with validation
class URI:
    def __init__(self, uri: str):
        if not self.validate_uri(uri):
            raise ValueError("Invalid URI format")
        self.uri = uri

    @staticmethod
    def validate_uri(uri: str) -> bool:
        # Example: Basic URI validation (modify as needed)
        # This is a simple example; actual URI validation might be more complex
        uri_regex = re.compile(r"^[a-zA-Z0-9:/._-]+$")
        return bool(uri_regex.match(uri))


# @dataclass
# class Triple:
#     sub: Node
#     pred: Predicate
#     obj: Node

#     def __str__(self, format: str = 'uri'):

@dataclass
class Node:
    uri: Optional[URIRef] = None
    label: Optional[Literal] = None
    nodetype: Optional[URIRef] = None

    def __init__(self, uri=None, nodetype=None, label=None):
        self.uri = uri
        self.nodetype = nodetype
        self.label = label

    def __str__(self):
        return str(self.uri) if self.uri else str(self.label)

@dataclass
class Predicate:
    uri: Optional[URIRef] = None
    label: Optional[Literal] = None

    def __str__(self):
        return str(self.uri) if self.uri else str(self.label)

def defrag_uri(uri):
    pattern = r'[#/]([^/#]+)$'
    match = re.search(pattern, uri)
    if match:
        name = match.group(1)
        p2 = re.compile(r"([a-z0-9])([A-Z])")
        name = p2.sub(r"\1 \2", name)
        return name
    else:
        return ""


@dataclass
class NodeSchema:
    node: Node
    in_preds: List[Tuple[Predicate, Node]] = field(default_factory=list)
    out_preds: List[Tuple[Predicate, Node]] = field(default_factory=list)

    def __str__(self, representation: str = 'uri'):
        triple_list = []
        for pair in self.in_preds:
            p, s = pair
            triple = (s, p, self.node)
            triple = self._get_triple_representation(triple, representation)
            triple_list.append(triple)
        for pair in self.out_preds:
            p, o = pair
            triple = (self.node, p, o)
            triple = self._get_triple_representation(triple, representation)
            triple_list.append(triple)
        return f"{triple_list}"

    def _get_triple_representation(self, triple: Tuple[Node, Predicate, Node], representation: str):
        if representation == 'label':
            sub_, pred_, obj_ = triple
            return (sub_.label, pred_.label, obj_.label)
        elif representation == 'uri':
            sub_, pred_, obj_ = triple
            return (defrag_uri(str(sub_.uri)),
                    defrag_uri(str(pred_.uri)),
                    defrag_uri(str(obj_.uri)))
        else:
            raise ValueError('invalid representation, allowed ["uri", "label"]')


@dataclass
class SubGraph:
    seed_node: Node
    quadruples: List[Tuple[Node, Predicate, Node, int]] = field(default_factory=list)
    triples: List[Tuple[Node, Predicate, Node]] = field(default_factory=list)

    def __str__(self, representation: str = 'uri'):
        triple_list = []
        for triple in self.triples:
            triple = self.get_triple_representation(triple, representation)
            triple_list.append(triple)
        return f"{triple_list}"

    def get_triple_representation_for_optimized(self, triple: Tuple[Node, Predicate, Node]):
        sub_, pred_, obj_ = triple
        predicate = pred_.label if pred_.label else defrag_uri(str(pred_.uri))
        if self.seed_node == sub_:
            subject = sub_.label if sub_.label else defrag_uri(str(sub_.uri))
            object = obj_.label if obj_.label else ""
        elif self.seed_node == obj_:
            subject = sub_.label if sub_.label else ""
            object = obj_.label if obj_.label else defrag_uri(str(obj_.uri))
        return (subject, predicate, object)

    def get_triple_representation_no_object(self, triple: Tuple[Node, Predicate, Node]):
        sub_, pred_, obj_ = triple
        predicate = pred_.label if pred_.label else defrag_uri(str(pred_.uri))
        if self.seed_node == sub_:
            subject = sub_.label if sub_.label else defrag_uri(str(sub_.uri))
            return (subject, predicate, '')
        elif self.seed_node == obj_:
            object = obj_.label if obj_.label else defrag_uri(str(obj_.uri))
            return('', predicate, object)

    def get_triple_with_uris_no_object(self, triple: Tuple[Node, Predicate, Node]):
        sub_, pred_, obj_ = triple
        if self.seed_node == sub_:
            return (sub_.uri, pred_.uri, '')
        elif self.seed_node == obj_:
            return ('', pred_.uri, obj_.uri)
            
    def get_quadruple_representation(self, quadruple: Tuple[Node, Predicate, Node, int]):
        triple_representation = self.get_triple_representation_for_optimized(quadruple[:3])
        return tuple(triple_representation) + (quadruple[3],)

    def get_triple_representation(self, triple: Tuple[Node, Predicate, Node], representation: str):
        output = list()
        for el in triple:
            # if el.uri:
            #     output.append(defrag_uri(str(el.uri)))
            # elif el.label:
            #     output.append(str(el.label))
            if el.label:
                output.append(str(el.label))
            elif el.uri:
                output.append(defrag_uri(str(el.uri)))
            else:
                raise ValueError('Invalid Element in Tuple, No URI or label for ', el)
        return tuple(output)
        # if representation == 'label':
        #     sub_, pred_, obj_ = triple
        #     return (sub_.label, pred_.label, obj_.label)
        # elif representation == 'uri':
        #     sub_, pred_, obj_ = triple
        #     return (defrag_uri(str(sub_.uri)),
        #     defrag_uri(str(pred_.uri)),
        #     defrag_uri(str(obj_.uri)))
        # else:
        #     raise ValueError('invalid representation, allowed ["uri", "label"]')

    def get_quadruple_summary(self, representation: str) -> str:
        "create quadruple summary"
        predicate_summary = defaultdict(int)
        first_occurrence = {}

        # Finding first occurrences of predicates
        for triple_index, triple in enumerate(self.triples):
            _, predicate, _ = triple
            predicate_str = str(predicate)
            if predicate_str not in first_occurrence:
                first_occurrence[predicate_str] = triple_index

            predicate_summary[predicate_str] += 1

        # Forming quadruples with counts from predicate_summary
        quadruples = []
        for predicate_str, triple_index in first_occurrence.items():
            triple = self.triples[triple_index]
            count = predicate_summary[predicate_str]
            quadruple = (*triple, count)
            quadruples.append(quadruple)

        self.quadruples = quadruples
        summary_str = ""
        for quadruple in quadruples:
            label_representation = self.get_triple_representation_for_optimized(quadruple[:3])
            summary_str += f"{tuple(label_representation) + (quadruple[3],)}\n"

        return summary_str

    def contain_triple(self, triple, approach):
        if approach == "subgraph":
            triple = triple.replace('"', '').replace("'", "")
            for el in self.triples:
                seralized_triple = ""
                seralized_triple = self.get_triple_representation(el, 'uri')
                seralized_triple = str(seralized_triple)
                seralized_triple = seralized_triple.replace('"', '').replace("'", "")
                if triple == seralized_triple:
                    return True
            return False
        elif approach == "optimized":
            for el in self.triples:
                seralized_triple = ""
                seralized_triple = self.get_triple_representation_no_object(el)
                if triple == seralized_triple:
                    return True
            return False

    def get_summarized_graph(self):
        seen_predicates = set()
        summarized_triples = list()
        for triple in self.triples:
            _, predicate, _ = triple
            if str(predicate) not in seen_predicates:
                seen_predicates.add(str(predicate))
                summarized_triples.append(triple)
        return summarized_triples

    def get_summarized_graph_str(self, approach):
        summarized_graph = self.get_summarized_graph()
        triple_list = []
        for triple in summarized_graph:
            if approach == "default":
                triple = self.get_triple_representation_for_optimized(triple)
            elif approach == "no_object":
                triple = self.get_triple_representation_no_object(triple)
            triple_list.append(triple)
        return f"{triple_list}"


@dataclass
class SparqlQueryResponse:
    head: Dict[str, List[str]]
    results: Dict[str, List[Dict[str, Optional[str]]]]

    def get_keys(self) -> List:
        return self.head.get("vars", [])

    def values_by_key(self, key: str) -> List:
        if not key in self.get_keys():
            raise ValueError(f"invalid key, allowd keys {self.get_keys()}")

        values = []
        for binding in self.results.get("bindings", []):
            binding_value = binding.get(key, {})
            value_type = binding_value.get("type", None)
            value_ = binding_value.get("value", None)
            if value_type == "uri":
                value = URIRef(value_)
            else:
                value = Literal(value_)
            values.append(value)

        return values


class KG:
    """
    Knowledge Graph Interface - extend it for various usecases
    - provides sparql query wrapper for knowledge graph's sparql endpoint
    - provides simple subgraph extractor for any seed node (1-hop star pattern)
    - uses redis cache for saving sparql query results for faster response
    """

    def __init__(self, type_to_predicate_map=None, endpoints=[], redis_host=redis_host, _redis_db_name=0, _verbose=False,
                 _selection_method="select-one"):
        self.verbose = _verbose
        self.endpoints = endpoints
        if len(endpoints) == 0:
            raise ValueError('Atleast one sparql endpoint required. use endpoints=["some endpoint uri"]')
        self.sparql_endpoint = endpoints[0]
        self.selection_method = _selection_method
        self.r = redis.Redis(host=redis_host, port=6379, db=_redis_db_name)
        self.logger = Logger().get_logger()
        # self.label_predicate_url = self.get_label_predicate_uri(label_predicate)
        # tried with just label suffix as input and finding full url from the kg, but the sparql-endpoint timeout
        self.type_to_predicate_map = type_to_predicate_map
        self.use_label = True

    def set_type_to_predicate_map(self, type_to_predicate_map):
        self.type_to_predicate_map = type_to_predicate_map

    def set_use_label(self, use_label):
        self.use_label = use_label

    def get_label_predicate_uri(self, label_predicate):
        query_template = '''
            SELECT DISTINCT ?predicate
            WHERE {
                ?s ?predicate ?o .
                FILTER(STRENDS(STR(?predicate), "%s"))
            }
            LIMIT 1
        '''
        query = query_template % label_predicate
        print(query)

        result = self.shoot_custom_query(query)
        if result and 'results' in result and 'bindings' in result['results']:
            bindings = result['results']['bindings']
            if bindings:
                return bindings[0]['predicate']['value']

        return None

    def get_predicate_label_for_type(self, node):
        if hasattr(node, 'nodetype') and node.nodetype is not None:
            if str(node.nodetype) in self.type_to_predicate_map:
                return self.type_to_predicate_map[str(node.nodetype)]
        elif 'default' in self.type_to_predicate_map:
            return self.type_to_predicate_map['default']
        return None

    def get_label(self, node):
        "use predicate label url and get label"
        label_query_template = """
            SELECT ?label WHERE {
                <%s> <%s> ?label .
            }
        """
        if not self.use_label:
            return None

        uri = str(node.uri)
        predicate = self.get_predicate_label_for_type(node)
        query = label_query_template % (uri, predicate)
        results = self.shoot_custom_query(query)
        label = None
        if results is None:
            return None
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

    def select_sparql_endpoint(self):
        """
        Based on the selection mechanism selected at __init__,
        this function tells which endpoint to use at every point.
        """
        if self.selection_method == "round-robin":
            index = self.endpoints.index(self.sparql_endpoint)
            return (
                self.endpoints[index + 1]
                if index >= len(self.endpoints)
                else self.endpoints[0]
            )

        if self.selection_method == "select-one":
            return self.sparql_endpoint

    def shoot_custom_query(self, _custom_query):
        """
        Shoot any custom query and get the SPARQL results as a dictionary
        """
        try:
            caching_answer = self.r.get(_custom_query)
            if caching_answer:
                print("@caching layer")
                return json.loads(caching_answer)
            print("kg cache miss")
#             return None
            sparql = SPARQLWrapper(self.select_sparql_endpoint())
            sparql.setQuery(_custom_query)
            sparql.setReturnFormat(JSON)
            try:
                caching_answer = sparql.query().convert()
            except Exception as e:
                print(e)
                print("Retrying With queries")
                time.sleep(5)
                caching_answer = sparql.query().convert()
            self.r.set(_custom_query, json.dumps(caching_answer))
            return caching_answer

        except Exception as e:
            self.logger.exception("Error occurred while executing custom query: %s", str(e))
            return None

    def estimate_graph_size(self, seed_node: Node):
        seed_uri = seed_node.uri
        sparql = f"""SELECT count(*) as ?count WHERE {{ {{?s ?p <{seed_uri}>}} Union {{<{seed_uri}> ?p ?o}} }}"""
        sub_result = utils.send_sparql_query(self.sparql_endpoint, sparql)
        sub_count = sub_result["results"]["bindings"][0].get('count', {}).get('value', None)
        return int(sub_count)


    def subgraph_extractor(self, seed_node: Node) -> SubGraph:
        """
        subgraph in two steps - two SPARQL
        1. seed will be a subject
        2. seed will be an object
        """
        try:
            triples = []
            seed_label = self.get_label(seed_node)
            if seed_label:
                seed_node.label = seed_label
            subject_ = seed_node.uri
            sparql_sub = f"""
            SELECT ?predicate ?object WHERE {{
                <{subject_}> ?predicate ?object
            }}
            """
            result = self.shoot_custom_query(sparql_sub)
            q_response = SparqlQueryResponse(**result)
            # keys = q_response.get_keys()
            preds = q_response.values_by_key('predicate')
            objs = q_response.values_by_key('object')
            i = 0
            for p, o in zip(preds, objs):
                # print("In loop ", i)
                # i += 1
                if isinstance(p, URIRef):
                    pred = Predicate(uri=p)
                    # p_label = self.get_label(pred)
                    # pred.label = p_label
                else:
                    pred = Predicate(label=p)

                if isinstance(o, URIRef):
                    obj = Node(uri=o)
                    # o_label = self.get_label(obj)
                    # obj.label = o_label
                else:
                    obj = Node(label=o)

                triples.append((seed_node, pred, obj))

            # print("out loop subject")
            object_ = seed_node.uri
            sparql_obj = f"""
            SELECT ?predicate ?subject WHERE {{
                ?subject ?predicate <{object_}>
            }}
            """
            result = self.shoot_custom_query(sparql_obj)
            q_response = SparqlQueryResponse(**result)
            # keys = q_response.get_keys()
            preds = q_response.values_by_key('predicate')
            subs = q_response.values_by_key('subject')
            i = 0
            for p, s in zip(preds, subs):
                # print("in loop object ", i)
                # i+=1
                if isinstance(p, URIRef):
                    pred = Predicate(uri=p)
                    # p_label = self.get_label(pred)
                    # pred.label = p_label
                else:
                    pred = Predicate(label=p)

                if isinstance(s, URIRef):
                    subj = Node(uri=s)
                    # s_label = self.get_label(subj)
                    # subj.label = s_label
                else:
                    subj = Node(label=s)
                triples.append((subj, pred, seed_node))

            subgraph = SubGraph(seed_node=seed_node, triples=triples)
            return subgraph
        except Exception as e:
            self.logger.exception("Error occurred while extracting subgraph: %s", str(e))
            return None


    def filter_subgraph(self, subgraph: SubGraph, seed_node) -> SubGraph:
        """add rules or condition to remove unnecessary triples"""
        filtered_triples = list()
        description_predicate = self.get_predicate_label_for_type(seed_node)
        for triple in subgraph.triples:
            # Skip Triples whose Subject or object is a blank node
            subject, predicate, object = triple
            if "nodeID://" in subject.__str__() or "nodeID://" in object.__str__():
                continue

            # Only include triples whose predicates not in pre-defined excluded list
            if predicate.__str__() not in utils.excluded_predicates:
                # check if the label predicate used to describe the type is in the triple
                if seed_node == subject and not description_predicate == predicate.__str__():
                    filtered_triples.append(triple)


        subgraph.triples = filtered_triples
        return subgraph


class DblpKG(KG):
    def __init__(self, label_predicate=None, rdf_schema_file=os.path.join(CURR_DIR, "dblp_rdf_schema.nt"),
                 # rdf_format="nt", endpoints=["http://206.12.95.86:8894/sparql/", "https://sparql.dblp.org/sparql"]):
                 rdf_format="nt", endpoints=[f"http://{host}:8894/sparql/"]):
        super().__init__(label_predicate, endpoints)
        # Additional attributes or initialization specific to Dblp

        allowed_formats = ["nt", "xml", "n3", "trix"]
        if not rdf_format in allowed_formats:
            raise ValueError("invalid rdf_format, please provide any of these ")
        self.schema = Graph()
        self.schema.parse(rdf_schema_file, format=rdf_format)
        self.class_namespace = RDFS
        self._classes = None
        self._parsed_schema = None

    @property
    def parsed_schema(self):
        if self._parsed_schema is None:
            self.parse_rdf_schema_to_dict()
        return self._parsed_schema

    @property
    def classes(self):
        if self._classes is None:
            self.parse_schema_classes()
        return self._classes

    def parse_rdf_schema_to_dict(self):
        if not self._classes:
            self.parse_schema_classes()
        self._parsed_schema = {}
        for class_, class_node in self._classes.items():
            class_out_preds = []
            for out_p in list(self.schema.subjects(self.class_namespace.domain, class_)):
                class_out_preds.extend(
                    [(out_p, x) for x in list(self.schema.objects(out_p, self.class_namespace.range))]
                )
            class_in_preds = []
            for in_p in list(self.schema.subjects(self.class_namespace.range, class_)):
                class_in_preds.extend(
                    [(in_p, x) for x in list(self.schema.objects(in_p, self.class_namespace.domain))]
                )
            class_in_preds = [(Predicate(uri=p), Node(uri=x, nodetype=x)) for p, x in class_in_preds]
            class_out_preds = [(Predicate(uri=p), Node(uri=x, nodetype=x)) for p, x in class_out_preds]
            # if (len(class_out_preds) > 0) and (len(class_in_preds) > 0):
            self._parsed_schema[class_] = NodeSchema(node=class_node, in_preds=class_in_preds,
                                                     out_preds=class_out_preds)

    def parse_schema_classes(self) -> Dict[str, Node]:
        # Iterate through the classes in the RDF graph
        self._classes = {}
        for class_uri in self.schema.subjects(RDF.type, self.class_namespace.Class):
            class_label = self.schema.value(class_uri, self.class_namespace.label)
            # class_comment = self.schema.value(class_uri, self.class_namespace.comment)
            self.classes[class_uri] = Node(uri=URIRef(class_uri), label=Literal(class_label))

    def schema_extractor(self, seed_node: Node) -> NodeSchema:
        """based on the nodetype of seed_node and already parsed schma"""
        return self.parsed_schema.get(seed_node.nodetype)


class YagoKG(KG):
    def __init__(self, label_predicate=None, rdf_schema_file=os.path.join(CURR_DIR, "yago_rdf_schema.nt"),
                 # rdf_format="nt", endpoints=["http://206.12.95.86:8892/sparql/"]):
                 rdf_format="nt", endpoints=[f"http://{host}:8892/sparql/"]):
        super().__init__(label_predicate, endpoints)
        # Additional attributes or initialization specific to YAGO 
        allowed_formats = ["nt", "xml", "n3", "trix"]
        if not rdf_format in allowed_formats:
            raise ValueError("invalid rdf_format, please provide any of these ")
        self.schema = Graph()
        self.schema.parse(rdf_schema_file, format=rdf_format)
        self.class_namespace = RDFS
        self.schema_namespace = Namespace(
            "http://schema.org/")  # yago does have domainIncludes and rangeIncludes from schema.org datamodel
        self._classes = None
        self._parsed_schema = None

    @property
    def parsed_schema(self):
        if self._parsed_schema is None:
            self.parse_rdf_schema_to_dict()
        return self._parsed_schema

    @property
    def classes(self):
        if self._classes is None:
            self.parse_schema_classes()
        return self._classes

    def parse_rdf_schema_to_dict(self):
        if not self._classes:
            self.parse_schema_classes()
        self._parsed_schema = {}
        for class_, class_node in self._classes.items():
            class_out_preds = []
            for out_p in list(self.schema.subjects(self.schema_namespace.domainIncludes, class_)):
                class_out_preds.extend(
                    [(out_p, x) for x in list(self.schema.objects(out_p, self.schema_namespace.rangeIncludes))]
                )
            class_in_preds = []
            for in_p in list(self.schema.subjects(self.schema_namespace.rangeIncludes, class_)):
                class_in_preds.extend(
                    [(in_p, x) for x in list(self.schema.objects(in_p, self.schema_namespace.domainIncludes))]
                )

            class_in_preds = [(Predicate(uri=p), Node(uri=x, nodetype=x)) for p, x in class_in_preds]
            class_out_preds = [(Predicate(uri=p), Node(uri=x, nodetype=x)) for p, x in class_out_preds]
            # if (len(class_out_preds) > 0) and (len(class_in_preds) > 0):
            self._parsed_schema[class_] = NodeSchema(node=class_node, in_preds=class_in_preds,
                                                     out_preds=class_out_preds)

    def parse_schema_classes(self) -> Dict[str, Node]:
        # Iterate through the classes in the RDF graph
        self._classes = {}
        for class_uri in self.schema.subjects(RDF.type, self.class_namespace.Class):
            class_label = self.schema.value(class_uri, self.class_namespace.label)
            # class_comment = self.schema.value(class_uri, self.class_namespace.comment)
            self.classes[class_uri] = Node(uri=URIRef(class_uri), label=Literal(class_label))

    def schema_extractor(self, seed_node: Node) -> NodeSchema:
        """based on the nodetype of seed_node and already parsed schma"""
        return self.parsed_schema.get(seed_node.nodetype)


class DbpediaKG(KG):
    def __init__(self, label_predicate=None, rdf_schema_file=os.path.join(CURR_DIR, "dbpedia_rdf_schema.nt"),
                 # rdf_format="nt", endpoints=["http://206.12.95.86:8890/sparql/", "http://dbpedia.org/sparql/", "http://live.dbpedia.org/sparql/"]):
                 rdf_format="nt", endpoints=[f"http://{host}:8890/sparql/"]):
        super().__init__(label_predicate, endpoints)
        # Additional attributes or initialization specific to DBPedia
        allowed_formats = ["nt", "xml", "n3", "trix"]
        if not rdf_format in allowed_formats:
            raise ValueError("invalid rdf_format, please provide any of these ")
        self.schema = Graph()
        self.schema.parse(rdf_schema_file, format=rdf_format)
        self.rdfs_namespace = RDFS
        self.class_namespace = OWL
        self._classes = None
        self._parsed_schema = None

    @property
    def parsed_schema(self):
        if self._parsed_schema is None:
            self.parse_rdf_schema_to_dict()
        return self._parsed_schema

    @property
    def classes(self):
        if self._classes is None:
            self.parse_schema_classes()
        return self._classes

    def parse_rdf_schema_to_dict(self):
        if not self._classes:
            self.parse_schema_classes()
        self._parsed_schema = {}
        for class_, class_node in self._classes.items():
            class_out_preds = []
            for out_p in list(self.schema.subjects(self.rdfs_namespace.domain, class_)):
                class_out_preds.extend(
                    [(out_p, x) for x in list(self.schema.objects(out_p, self.rdfs_namespace.range))]
                )
            class_in_preds = []
            for in_p in list(self.schema.subjects(self.rdfs_namespace.range, class_)):
                class_in_preds.extend(
                    [(in_p, x) for x in list(self.schema.objects(in_p, self.rdfs_namespace.domain))]
                )
            class_in_preds = [(Predicate(uri=p), Node(uri=x, nodetype=x)) for p, x in class_in_preds]
            class_out_preds = [(Predicate(uri=p), Node(uri=x, nodetype=x)) for p, x in class_out_preds]
            # if (len(class_out_preds) > 0) and (len(class_in_preds) > 0):
            self._parsed_schema[class_] = NodeSchema(node=class_node, in_preds=class_in_preds,
                                                     out_preds=class_out_preds)

    def parse_schema_classes(self):
        # Iterate through the classes in the RDF graph
        self._classes = {}
        for class_uri in self.schema.subjects(RDF.type, self.class_namespace.Class):
            class_label = self.schema.value(class_uri, self.rdfs_namespace.label)
            # class_comment = self.schema.value(class_uri, self.rdfs_namespace.comment)
            self.classes[class_uri] = Node(uri=URIRef(class_uri), label=Literal(class_label))

    def schema_extractor(self, seed_node: Node) -> NodeSchema:
        """based on the nodetype of seed_node and already parsed schma"""
        return self.parsed_schema.get(seed_node.nodetype)
