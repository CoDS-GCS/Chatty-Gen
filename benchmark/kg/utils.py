import json
from rdflib import Graph, RDF, RDFS, OWL
from dataclasses import dataclass
from rdflib import Graph, URIRef, Literal, RDF
from typing import List, Tuple, Optional, Dict


def save_dict_to_json_file(dictionary, filename):
    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)


class KgSchema:
    def __init__(self, rdf_file_or_uri, rdf_format, class_namespace="RDFS"):
        allowed_formats = ["nt", "xml", "n3", "trix"]
        if not rdf_format in allowed_formats:
            raise ValueError("invalid rdf_format, please provide any of these ")
        self.schema = Graph()
        self.schema.parse(rdf_file_or_uri, format=rdf_format)
        self.class_namespace = class_namespace
        self._classes = None
        self._parsed_schema = None
        self._nodetype_triple_list = None

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

    @property
    def nodetype_triple_list(self):
        if self._nodetype_triple_list is None:
            self.create_nodetype_triple_list()
        return self._nodetype_triple_list

    def parse_rdf_schema_to_dict(self):
        if not self._classes:
            self.parse_schema_classes()
        self._parsed_schema = {}
        for class_ in self._classes.keys():
            class_out_preds = []
            for out_p in list(self.schema.subjects(RDFS.domain, class_)):
                class_out_preds.extend(
                    [(out_p, x) for x in list(self.schema.objects(out_p, RDFS.range))]
                )
            class_in_preds = []
            for in_p in list(self.schema.subjects(RDFS.range, class_)):
                class_in_preds.extend(
                    [(in_p, x) for x in list(self.schema.objects(in_p, RDFS.domain))]
                )
            if (len(class_out_preds) > 0) and (len(class_in_preds) > 0):
                self._parsed_schema[class_] = {
                    "nodeuri": class_,
                    "incoming_predicates": class_in_preds,
                    "outgoing_predicates": class_out_preds,
                }
        # save_dict_to_json_file(self._parsed_schema, "schema.json")

    def parse_schema_classes(self):
        # Iterate through the classes in the RDF graph
        self._classes = {}
        if self.class_namespace == "RDFS":
            NAMESPACE = RDFS
        elif self.class_namespace == "OWL":
            NAMESPACE = OWL

        for class_uri in self.schema.subjects(RDF.type, NAMESPACE.Class):
            class_label = self.schema.value(class_uri, RDFS.label)
            class_comment = self.schema.value(class_uri, RDFS.comment)
            self.classes[class_uri] = {
                "uri": class_uri,
                "label": class_label,
                "comment": class_comment,
            }

    def create_nodetype_triple_list(self):
        parsed_schema = self.parsed_schema

        # TODO: fix the label instead of direct uri in triple
        triple_list = []
        for node_type, node_info in parsed_schema.items():
            incoming_predicates = node_info.get("incoming_predicates")
            outgoing_predicates = node_info.get("outgoing_predicates")

            node_triples = []
            object_type = node_type
            for p in incoming_predicates:
                predicate, subject_type = p
                node_triples.append((subject_type, predicate, object_type))

            subject_type = node_type
            for p in outgoing_predicates:
                predicate, object_type = p
                node_triples.append((subject_type, predicate, object_type))

            triple_list.extend(node_triples)
            # break

        self._nodetype_triple_list = triple_list


SPARQL_TEMPLATES = {
    "get_seed_nodes": """
        SELECT DISTINCT ?node (COUNT(?outgoingPredicate) + COUNT(?incomingPredicate) AS ?predicateCount)
        WHERE {
              ?node a <%(e)s>.
              ?node ?outgoingPredicate ?object.
              ?subject ?incomingPredicate ?node.
        }
        GROUP BY ?node
        ORDER BY DESC(?predicateCount)
        LIMIT 1
        """,
}


def seed_node_sparql_query(node_type_uri, incoming_predicates, outgoing_predicates):
    incoming_predicates_str = ", ".join(
        f"<{p}> " if isinstance(p, str) else f"<{p.toPython()}> "
        for p in incoming_predicates
    )
    outgoing_predicates_str = ", ".join(
        f"<{p}> " if isinstance(p, str) else f"<{p.toPython()}> "
        for p in outgoing_predicates
    )

    node_type_uri_str = (
        node_type_uri if isinstance(node_type_uri, str) else node_type_uri.toPython()
    )

    sparql_query_template = f"""
    SELECT DISTINCT ?node (COUNT(?outgoingPredicate) + COUNT(?incomingPredicate) AS ?predicateCount)
    WHERE {{
        ?node a <{node_type_uri}>.
        ?node ?outgoingPredicate ?object.
        ?subject ?incomingPredicate ?node.
        FILTER (?outgoingPredicate IN ({outgoing_predicates_str}))
        FILTER (?incomingPredicate IN ({incoming_predicates_str}))
    }}
    GROUP BY ?node
    HAVING (COUNT(?outgoingPredicate) > 0 && COUNT(?incomingPredicate) > 0)
    ORDER BY DESC(?predicateCount)
    LIMIT 1
    """

    return sparql_query_template


def format_sparql_template_with_dict(template, values_dict):
    try:
        formatted_string = template % values_dict
        return formatted_string
    except KeyError as e:
        return f"Error: Missing key '{e.args[0]}' in the dictionary."
    except ValueError:
        return "Error: Type mismatch in the template."
    except Exception as e:
        return f"An error occurred: {str(e)}"


@dataclass
class SparqlQueryResponse:
    head: Dict[str, List[str]]
    results: Dict[str, List[Dict[str, Optional[str]]]]


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


# def incoming_star_pattern_sparql_query(prefixes, object_uri, predicate_and_type_list):
#     # Construct the PREFIX part of the query
#     # Initialize the SPARQL query
#     query = prefixes + "SELECT DISTINCT ?sampleSubject ?predicate ?type WHERE {\n"
#
#     # Add the VALUES block for predicates
#     query += "  VALUES (?predicate ?type) {\n"
#     for pred, pred_type in predicate_and_type_list:
#         query += f"(<{pred}> <{pred_type}>) "
#     query += "  }\n"
#
#     # Add the subquery for the subject and objects
#     query += "  {\n"
#     query += f"SELECT ?predicate (SAMPLE(?subject) AS ?sampleSubject) ?type\n"
#     query += "    WHERE {\n"
#     query += f"  ?subject a ?type ;\n"
#     query += f"     ?predicate <{object_uri}>.\n"
#     query += "    }\n"
#     query += "GROUP BY ?predicate ?type"
#     query += "  }\n"
#
#     # Close the main query
#     query += "}\n"
#
#     return query

# def outgoing_star_pattern_sparql_query(prefixes, subject_uri, predicate_and_type_list):
#     # Construct the PREFIX part of the query
#
#     # Initialize the SPARQL query
#     query = prefixes + "SELECT DISTINCT ?sampleObject ?predicate ?type WHERE {\n"
#
#     # Add the VALUES block for predicates
#     query += "  VALUES (?predicate ?type) {\n"
#     for pred, pred_type in predicate_and_type_list:
#         print(pred_type)
#         query += f"(<{pred}> <{pred_type}>) "
#     query += "  }\n"
#
#     # Add the subquery for the subject and objects
#     query += "  {\n"
#     query += f"SELECT ?predicate (SAMPLE(?object) AS ?sampleObject) ?type\n"
#     query += "    WHERE {\n"
#     query += f"  <{subject_uri}> a ?type ;\n"
#     query += f"     ?predicate ?object.\n"
#     query += "    }\n"
#     query += "GROUP BY ?predicate ?type"
#     query += "  }\n"
#
#     # Close the main query
#     query += "}\n"
#
#     return query


def incoming_predicate_sparql_util(object_uri, predicate, subject_type):
    query_template = """
    SELECT DISTINCT ?subject ?predicate WHERE {{
        ?subject rdf:type <{subject_type}>;
                 <{predicate}> <{object_uri}>.
        BIND(<{predicate}> AS ?predicate)
    }}
    LIMIT 1
    """

    sparql_query = query_template.format(
        subject_type=subject_type, predicate=predicate, object_uri=object_uri
    )

    return sparql_query


def outgoing_predicate_sparql_util(subject_uri, predicate, object_type):
    query_template = """
    SELECT DISTINCT ?object ?predicate WHERE {{
        <{subject_uri}> rdf:type <{object_type}>;
                      <{predicate}> ?object.
        BIND(<{predicate}> AS ?predicate)
    }}
    LIMIT 1
    """

    sparql_query = query_template.format(
        subject_uri=subject_uri, object_type=object_type, predicate=predicate
    )
    return sparql_query


def incoming_star_pattern_sparql_query(prefixes, object_uri, predicate_subject_types):
    # Construct the PREFIX part of the query
    # Initialize the SPARQL query
    query = prefixes

    subqueries = []

    for predicate, subject_type in predicate_subject_types:
        subquery = incoming_predicate_sparql_util(object_uri, predicate, subject_type)
        subqueries.append(subquery)

    combined_query = " UNION ".join(["{" + subq + "}" for subq in subqueries])
    final_query = prefixes + "\nSELECT * WHERE {" + combined_query + "}"

    return final_query


def outgoing_star_pattern_sparql_query(prefixes, subject_uri, predicate_object_types):
    # Construct the PREFIX part of the query
    # Initialize the SPARQL query
    query = prefixes

    subqueries = []

    for predicate, object_type in predicate_object_types:
        subquery = outgoing_predicate_sparql_util(subject_uri, predicate, object_type)
        subqueries.append(subquery)

    combined_query = " UNION ".join(["{" + subq + "}" for subq in subqueries])
    final_query = prefixes + "\nSELECT * WHERE {" + combined_query + "}"

    return final_query


if __name__ == "__main__":
    test_kg = KgSchema("rdf_schema.nt", "nt")
    print(test_kg.parsed_schema)
