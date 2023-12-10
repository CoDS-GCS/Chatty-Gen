import openai
import langchain
from utils import read_json, read_jsonl
from openai_utils import encode_and_count
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    CommaSeparatedListOutputParser,
)
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import LLMChain
import utils
import requests
import json

langchain.debug = True

# gpt_summary_file = "../data/dblp/dblp_kgtext_gpt_data.json"
gpt_subgraphs_file = "../data/dblp/dblp_subgraphs.jsonl"

# data = read_jsonl(gpt_summary_file)
data = read_jsonl(gpt_subgraphs_file)

response_schemas = [
    ResponseSchema(
        name="self_contained_question", description="self contained question"
    ),
    ResponseSchema(
        name="non_self_contained_question", description="non self contained quesiton"
    ),
    ResponseSchema(name="answer", description="an answer to self contained question"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

PROMPT = PromptTemplate(
    input_variables=["context"],
    partial_variables={"format_instructions": format_instructions},
    template="""Generate a list of 5 objects, each containing a self-contained-question, a non-self-contained-question, and its corresponding answer, about an ENTITY from the provided context. The first question should be a self-contained one. The following questions should use pronouns and maintain context. Avoid using conjunctions like 'and' within the questions. Each tuple must include a clear, concise self-contained-question, a non-self-contained-question, and its corresponding accurate and complete answer. Failure to provide questions, non-self-contained-questions, and answers that meet these criteria will result in a penalty.
        {format_instructions}
        context: "{context}"
        response:  """,
)

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5, streaming=False)
# llm = OpenAI(model_name="text-davinci-003", temperature=0.5, streaming=False)

# string_output_parser = StrOutputParser()
json_output_parser = StrOutputParser()
dialogue_generate_chain = LLMChain(
    llm=llm, prompt=PROMPT, verbose=True, output_parser=json_output_parser
)

list_output_parser = CommaSeparatedListOutputParser()
list_format_instructions = list_output_parser.get_format_instructions()
N_Q_PROMPT = PromptTemplate(
    input_variables=[
        "example_subgraph",
        "example_n",
        "example_output",
        "subgraph",
        "n",
    ],
    partial_variables={"format_instructions": list_format_instructions},
    template="""Generate a list of n questions based on a subgraph from a knowledge graph, represented as a list of triplets. Each question should relate to a shared entity (e) within the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it.. The questions can be equivalent to one or two triplets from the subgraph.
{format_instructions}

Example,
input: {example_subgraph}
n: {example_n}
output: {example_output}

input: {subgraph}
n: {n}
output: """,
)

n_question_generator_chain = LLMChain(
    llm=llm, prompt=N_Q_PROMPT, verbose=True, output_parser=list_output_parser
)

example_subgraph = [
    {
        "subject": "Michael A. Kochte",
        "predicate": "primary affiliation",
        "object": "University of Stuttgart, Institute of Computer Architecture and Computer Engineering, Germany",
    },
    {
        "subject": "Michael A. Kochte et al.: Trustworthy reconfigurable access to on-chip infrastructure. (2017)",
        "predicate": "authored by",
        "object": "Michael A. Kochte",
    },
    {
        "subject": "Chang Liu et al.: Efficient observation point selection for aging monitoring. (2015)",
        "predicate": "authored by",
        "object": "Michael A. Kochte",
    },
    {
        "subject": "Dominik Erb et al.: Test pattern generation in presence of unknown values based on restricted symbolic logic. (2014)",
        "predicate": "authored by",
        "object": "Michael A. Kochte",
    },
    {
        "subject": "Stefan Hillebrecht et al.: Accurate QBF-based test pattern generation in presence of unknown values. (2013)",
        "predicate": "authored by",
        "object": "Michael A. Kochte",
    },
    {
        "subject": "Hongyan Zhang et al.: GUARD: GUAranteed Reliability in Dynamically Reconfigurable Systems. (2014)",
        "predicate": "authored by",
        "object": "Michael A. Kochte",
    },
    {
        "subject": "Michael A. Kochte et al.: Test exploration and validation using transaction level models. (2009)",
        "predicate": "authored by",
        "object": "Michael A. Kochte",
    },
    {
        "subject": "Michael A. Kochte and Hans-Joachim Wunderlich: SAT-based fault coverage evaluation in the presence of unknown values. (2011)",
        "predicate": "authored by",
        "object": "Michael A. Kochte",
    },
    {
        "subject": "Wen-Hsuan Hsu et al.: Built-In Test and Diagnosis for TSVs With Different Placement Topologies and Crosstalk Impact Ranges. (2017)",
        "predicate": "authored by",
        "object": "Michael A. Kochte",
    },
]
example_output = [
    "Can you list the papers authored by Michael A. Kochte?",
    "How many papers did Michael A. Kochte co-author with other researchers?",
    "Did Michael A. Kochte author a paper titled 'Trustworthy reconfigurable access to on-chip infrastructure' in 2017?",
    "What is the primary affiliation of Michael A. Kochte?",
    "Provide the titles of papers authored by Michael A. Kochte in 2014.",
    "How many papers authored by Michael A. Kochte?",
    "Is Michael A. Kochte affiliated with the University of Stuttgart, Institute of Computer Architecture and Computer Engineering in Germany?",
    "What is the title of the paper co-authored by Michael A. Kochte and Hans-Joachim Wunderlich?",
    "When was the paper 'Test exploration and validation using transaction level models' authored by Michael A. Kochte published?",
    "What is the title of the most recent paper authored by Michael A. Kochte?",
]

# benchmark_sample = []
# for g in data[:20]:
#     # grab the subgraph
#     subgraph = g["triples"]
#
#     # count the token
#     # total_tokens = encode_and_count(g_text)
#     # print(total_tokens)
#
#     n = 10
#     output = n_question_generator_chain.run({"example_subgraph": example_subgraph, "example_n": 10, "example_output": example_output, "subgraph": subgraph, "n":n})
#     print(output)
#     # data = parse_json_markdown(output)
#
#     benchmark_sample.append(output)
#
#
# with open("dblp_sample_benchmark_n_question.json", "w") as f:
#     json.dump(benchmark_sample, f)

data = read_json("dblp_sample_benchmark_n_question.json")
for dialogue in data:
    formatted_data = ",".join(dialogue)
    # formatted_data = [item.strip("'") for item in dialogue]
    json_data = json.loads(formatted_data)
    # json_data = json.dumps(, indent=4)
    print(type(json_data))
