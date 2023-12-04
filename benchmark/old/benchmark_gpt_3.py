"""

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
[
{
question": string  // question
},
]
```

Example,
context: ""
n: 5
response: 
context: "[{"subject": "Jasper van de Ven", "predicate": "primary affiliation", "object": "University of Bremen, Germany"}, {"subject": "Yun-Ming Shih et al.: Translation of String-and-Pin-based Shortest Path Construction into Data-Scalable Agent-based Computational Models. (2018)", "predicate": "authored by", "object": "Jasper van de Ven"}, {"subject": "Jasper van de Ven and Frank Dylla: The Spatial Interaction Laboratory - A Distributed Middleware and Qualitative Representation for Ambient Intelligence. (2013)", "predicate": "authored by", "object": "Jasper van de Ven"}, {"subject": "Jasper van de Ven and Frank Dylla: Qualitative Privacy Description Language - Integrating Privacy Concepts, Languages, and Technologies. (2016)", "predicate": "authored by", "object": "Jasper van de Ven"}, {"subject": "Jasper van de Ven: Supporting communication in spatially distributed groups: privacy as a service for ambient intelligence. (2016)", "predicate": "authored by", "object": "Jasper van de Ven"}, {"subject": "Jasper van de Ven et al.: Multi-PN-learning for tracking applications. (2014)", "predicate": "authored by", "object": "Jasper van de Ven"}, {"subject": "Jasper van de Ven and Frank Dylla: The Spatial Interaction Laboratory. (2017)", "predicate": "authored by", "object": "Jasper van de Ven"}, {"subject": "Christian Freksa et al.: Geometric problem solving with strings and pins. (2019)", "predicate": "authored by", "object": "Jasper van de Ven"}, {"subject": "Ahmed Loai Ali et al.: Experience with the Mobile4D Disaster Reporting and Alerting System in Lao PDR. (2017)", "predicate": "authored by", "object": "Jasper van de Ven"}]"
n: 10
response:
"""

import openai
import langchain
from utils import read_json, read_jsonl
from openai_utils import encode_and_count
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, CommaSeparatedListOutputParser
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
    ResponseSchema(name="self_contained_question", description="self contained question"),
    ResponseSchema(name="non_self_contained_question", description="non self contained quesiton"),
    ResponseSchema(name="answer", description="an answer to self contained question")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()


llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5, streaming=False)
# llm = OpenAI(model_name="text-davinci-003", temperature=0.5, streaming=False)

# string_output_parser = StrOutputParser()
json_output_parser = StrOutputParser()

list_output_parser = CommaSeparatedListOutputParser()
list_format_instructions = list_output_parser.get_format_instructions()
D_N_Q_PROMPT = PromptTemplate(
    input_variables=["example_subgraph", "example_n", "example_output", "subgraph", "n"],
    partial_variables={"format_instructions": list_format_instructions},
    template="""Create a dialogue in the form of a list with n objects. Each object should contain a question, all related to a specific ENTITY from the provided context. The first question initiates the conversation, while the subsequent questions should refer to the ENTITY using pronouns and maintain context. Avoid using conjunctions like 'and' within the questions. Please ensure that the questions form a coherent dialogue. The questions can be equivalent to one or two triples and should be of one of the following categories: list, count, boolean, wh or date-related. Failure to follow these guidelines may result in a penalty.
{format_instructions}

Example,
input: {example_subgraph}
n: {example_n}
output: {example_output}

input: {subgraph}
n: {n}
output: """
)

n_question_generator_chain = LLMChain(
    llm = llm,
    prompt = D_N_Q_PROMPT,
    verbose = True,
    output_parser = list_output_parser 
)

# example_subgraph =  [{"subject": "Michael A. Kochte", "predicate": "primary affiliation", "object": "University of Stuttgart, Institute of Computer Architecture and Computer Engineering, Germany"}, {"subject": "Michael A. Kochte et al.: Trustworthy reconfigurable access to on-chip infrastructure. (2017)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Chang Liu et al.: Efficient observation point selection for aging monitoring. (2015)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Dominik Erb et al.: Test pattern generation in presence of unknown values based on restricted symbolic logic. (2014)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Stefan Hillebrecht et al.: Accurate QBF-based test pattern generation in presence of unknown values. (2013)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Hongyan Zhang et al.: GUARD: GUAranteed Reliability in Dynamically Reconfigurable Systems. (2014)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Michael A. Kochte et al.: Test exploration and validation using transaction level models. (2009)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Michael A. Kochte and Hans-Joachim Wunderlich: SAT-based fault coverage evaluation in the presence of unknown values. (2011)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Wen-Hsuan Hsu et al.: Built-In Test and Diagnosis for TSVs With Different Placement Topologies and Crosstalk Impact Ranges. (2017)", "predicate": "authored by", "object": "Michael A. Kochte"}]
# example_output = ["Can you list the papers authored by Michael A. Kochte?", "How many papers did Michael A. Kochte co-author with other researchers?","Did Michael A. Kochte author a paper titled 'Trustworthy reconfigurable access to on-chip infrastructure' in 2017?", "What is the primary affiliation of Michael A. Kochte?", "Provide the titles of papers authored by Michael A. Kochte in 2014.", "How many papers authored by Michael A. Kochte?", "Is Michael A. Kochte affiliated with the University of Stuttgart, Institute of Computer Architecture and Computer Engineering in Germany?", "What is the title of the paper co-authored by Michael A. Kochte and Hans-Joachim Wunderlich?", "When was the paper 'Test exploration and validation using transaction level models' authored by Michael A. Kochte published?", "What is the title of the most recent paper authored by Michael A. Kochte?"]

example_subgraph = [{"subject": "Parisa Memarmoshrefi", "predicate": "primary affiliation", "object": "University of Göttingen, Institute for Computer Science, Germany"}, {"subject": "Hang Zhang et al.: Investigating the Learning Phase of an Autonomous Authentication in Mobile Ad-hoc Networks. (2016)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "William Casey et al.: Identity Deception and Game Deterrence via Signaling Games. (2016)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "William Casey et al.: Identity Deception and Game Deterrence via Signaling Games. (2015)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "Parisa Memarmoshrefi et al.: Investigation of a bio-inspired security mechanism in Mobile Ad hoc Networks. (2013)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "Parisa Memarmoshrefi et al.: Autonomous Ant-based Public Key Authentication Mechanism for Mobile Ad-hoc Networks. (2016)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "Parisa Memarmoshrefi: A Bio-Inspired Autonomous Authentication Mechanism in Mobile Ad Hoc Networks. (2012)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "William Casey et al.: Deception, identity, and security: the game theory of sybil attacks. (2019)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}, {"subject": "Emmanuel Charleson Dapaah et al.: An AI-Based Transmission Power-Control Certificate Omission in Vehicular Ad-Hoc Networks. (2021)", "predicate": "authored by", "object": "Parisa Memarmoshrefi"}]

example_n = 5

example_output = """```json
[
{
    "question": "What is Parisa Memarmoshrefi's primary affiliation?"
},
{
    "question": "Did she author a paper titled 'Investigating the Learning Phase of an Autonomous Authentication in Mobile Ad-hoc Networks. (2016)'?"
},
{
    "question": "What other papers did Parisa Memarmoshrefi author?"
},
{
    "question": "How many papers she authored"
},
{
    "question": "Did she collaborate with William Casey on any other papers?"
},
]
```
"""


benchmark_sample = []
for g in data[:1]:
    # grab the subgraph
    subgraph = g["triples"]

    # count the token 
    # total_tokens = encode_and_count(g_text)
    # print(total_tokens)
    try:
        n = 10
        output = n_question_generator_chain.run({"example_subgraph": example_subgraph, "example_n": example_n, "example_output": example_output, "subgraph": subgraph, "n":n})
        output = json.loads(output[0])
        # data = parse_json_markdown(output)

        benchmark_sample.append(output)
    except:
        pass


# with open("dblp_sample_benchmark_n_question_dialogue.json", "w") as f:
#     json.dump(benchmark_sample, f)
