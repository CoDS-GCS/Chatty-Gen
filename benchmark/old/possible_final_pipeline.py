import sys
sys.path.append('/omij/Chatbot-Resources')

import openai
import langchain
import subgraph_extractor.kg.yago as yago
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
)
from langchain.chains import LLMChain
from langchain.pydantic_v1 import BaseModel, Field, validator
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from rdflib import Graph, URIRef, Literal, RDF
import requests
import json
import os

langchain.debug = True


response_schemas = [
    ResponseSchema(
        name="output",
        description="list of question templates",
        type="List[string]",
    )
]

json_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
json_format_instructions = json_output_parser.get_format_instructions()

llm = OpenAI(model_name="text-davinci-003", temperature=0.5, streaming=False)
PROMPT = PromptTemplate(
    input_variables=["example_input", "example_output", "input"],
    partial_variables={"format_instructions": json_format_instructions},
    template="""Generate a list of 6 questions as template strings, given the 'triple'. A triple consisting of 3 elements: a subject node type, a predicate, an object node type. The output should be a list containing these different question variations. The first 3 questions should be with subject node as an answer, where as next 3 questions should be with object node as an answer.
{format_instructions}

example,
input: {example_input}
output: {example_output}

input: {input}
output: """,
)

question_template_chain = LLMChain(
    llm=llm, prompt=PROMPT, verbose=True, output_parser=json_output_parser
)
example_output = [
    "Which company employs {subject}?",
    "At which company does {subject} work?",
    "What is the workplace of {subject}?",
    "Who works at {object}?",
    "Which person is employed at {object}?",
    "At {object}, who is an employee?",
]

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


def format_template_with_dict(template, values_dict):
    try:
        formatted_string = template.format(**values_dict)
        return formatted_string
    except KeyError as e:
        return f"Error: Missing key '{e.args[0]}' in the dictionary."
    except ValueError:
        return "Error: Type mismatch in the template."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def question_template_from_triple(triple):
    example_input = ("Person", "worksAt", "Company")
    output = question_template_chain.run(
        {
            "example_input": example_input,
            "example_output": example_output,
            "input": triple,
        }
    )
    qs = output.get("output", [])
    sub_q = qs[:3]
    obj_q = qs[3:]
    return {"subject": sub_q, "object": obj_q}


# step 1
"""
given the rdf schema we parsed it into some mapping format
where we have node type as key and value as information about it in schema
the info is in dict format, with incoming, outgoing predicates
both will be list of tuple of format (predicate, subject/object node type)
"""
parsed_schema_map = {
    "Person": {
        "nodetype": "http://schema.org/Person",
        "incoming_predicates": [("actor", "Movie"), ("spouse", "Person")],
        "outgoing_predicates": [("worksFor", "Organization"), ("spouse", "Person"), ("nationality", "Country")],
    }
}

# question template generation step
# for now considering its already there
"""
create a map of triple to list of questions from gpt using prompt
"""
triple_question_templates_map = {}
triple_list = []
for node_type, node_info in parsed_schema_map.items():
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

# print(triple_list)


# how to get this triple_list
# - simple we have map of node type along with incoming and outgoing predicate and respected node types
for triple in triple_list:
    triple_question_templates_map[triple] = question_template_from_triple(triple)

# print(triple_question_templates_map)

# step 2
"""
given the parsed schema map we will get the seed nodes for the different node types
the result will be list of tuples of format (url, label-literal) for the node type
TODO: use subgraph_extractor's popular seeds extraction
"""
seed_nodes = {
    "Person": [
        ("http://yago-knowledge.org/resource/Abhishek_Kapoor", "Abhishek Kapoor")
    ],
}

sparql_query_templates={
    "get_seed_nodes_popular_2": """
        SELECT DISTINCT ?node (COUNT(?outgoingPredicate) + COUNT(?incomingPredicate) AS ?predicateCount)
        WHERE {
              ?node a <%(e)s>.
              ?node ?outgoingPredicate ?object.
              ?subject ?incomingPredicate ?node.
        }
        GROUP BY ?node
        ORDER BY DESC(?predicateCount)
        LIMIT 5
        """,
        }

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

for node_type, node_info in parsed_schema_map.items():
    node_type_uri = node_info.get('nodetype')
    incoming_predicates = node_info.get("incoming_predicates")
    outgoing_predicates = node_info.get("outgoing_predicates")

    query_template = sparql_query_templates.get("get_seed_nodes_popular_2")
    values = {"e": node_type_uri}
    query = format_sparql_template_with_dict(query_template, values)
    print(query)
    kg = yago.YAGO()
    seed_nodes = kg.shoot_custom_query(query)
    response = SparqlQueryResponse(**seed_nodes)
    seed_nodes = extract_values_by_key("node", response)
    print(f"extracted seed nodes: {len(seed_nodes)}")
    # TODO: label extraction
    print(f"extracted seed nodes: {seed_nodes}")

# sys.exit(1)

# step 3
"""
given this seed_nodes list we used to get subgraph for star pattern arround it.
this step contains two sub task
1) get one incoming triple per incoming predicate
2) get one outgoing triple per outgoing predicate
kind of complex to do all in one sparql query
we need the type of subject/object along with the instance of it
why we need type - from the type we can get the question template from generated templates
using this template we can get question for predicate. (for now not considering multiple subject/object with single predicate)
can we do it using single sparql - yes
we get the optional type along with triple
the type we get is not that obvious so instead we will give type along with query from the parsed_schema_map - works
the output - dictionary:
    seed_node: (uri, label)
    incoming_predicates: [("predicate", "subject type", "subject")]
    outgoing_predicates: [("predicate", "object type", "object")]

TODO: use the query for star patterns

"""
seed_node_subgraph = {
    "seed_node": (
        "http://yago-knowledge.org/resource/Abhishek_Kapoor",
        "Abhishek Kapoor",
        "Person",
    ),
    "incoming_predicates": [("spouse", "Person", "Pragya Yadav")],
    "outgoing_predicates": [("nationality", "Country", "India")],
}




# step 4
"""
generating set of questions from the subgraph (using template questions)
now we have some question template per triplet, triplet contains the (subject-type, predicate, object-type)
using this map and seed_node_subgraph we can generate set of questions.
two steps:
    incoming predicate triple
    outgoing predicate triple
"""
question_set = []

seed_uri, seed_label, seed_type = seed_node_subgraph["seed_node"]
## incoming type
# in this types of question, subject will be an answer and object will be our seed node
# so when filling the template, we will fill object and will have unknown for subject
for triple in seed_node_subgraph.get("incoming_predicates"):
    predicate, subject_type, subject_label = triple
    query_triple = (subject_type, predicate, seed_type)
    # print(query_triple)
    # ideally we should use one question per predicate, which will make sure to have small subgraphs
    # for selection we can do randomaly for now as we don't have knowledge of other question type in subgraph
    # or just select all and then implement selction strategy at the end.
    # for now we will get just one random template in list
    question_templates = triple_question_templates_map.get(query_triple).get("object")
    for q_t in question_templates:
        values = {"object": seed_label}
        qst = format_template_with_dict(q_t, values)
        ans = subject_label
        question_set.append((qst, ans))

## outgoing type
# in this types of question, object will be an answer and subject will be our seed node
# so when filling the template, we will fill subject and will have unknown for object
for triple in seed_node_subgraph.get("outgoing_predicates"):
    predicate, object_type, object_label = triple
    query_triple = (seed_type, predicate, object_type)
    # print(query_triple)
    # ideally we should use one question per predicate, which will make sure to have small subgraphs
    # for selection we can do randomaly for now as we don't have knowledge of other question type in subgraph
    # or just select all and then implement selction strategy at the end.
    # for now we will get just one random template in list
    question_templates = triple_question_templates_map.get(query_triple).get("subject")
    for q_t in question_templates:
        values = {"subject": seed_label}
        qst = format_template_with_dict(q_t, values)
        ans = object_label
        question_set.append((qst, ans))

# we got set of questions
print(question_set)


# step 5
"""
converting the question set of subgraph to dialogue
steps:
    pronoun indentification and substitution
    filteration
"""

