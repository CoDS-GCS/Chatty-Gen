import openai
import langchain
from utils import read_json, read_jsonl
from openai_utils import encode_and_count
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema.output_parser import StrOutputParser 
from langchain.chains import LLMChain
import utils
import requests
import json

langchain.debug = False

# gpt_summary_file = "../data/dblp/dblp_kgtext_gpt_data.json"
gpt_subgraphs_file = "../data/dblp/dblp_subgraphs.jsonl"

# data = read_jsonl(gpt_summary_file)
data = read_jsonl(gpt_subgraphs_file)

response_schemas = [
    ResponseSchema(name="stand_alone_question", description="standalone question"),
    ResponseSchema(name="non_standalone_question", description="non-standalone quesiton"),
    # ResponseSchema(name="answer", description="an answer to standalone question")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

# PROMPT = PromptTemplate(
#     input_variables=["context", "example", "example_response"], 
#     partial_variables={"format_instructions": format_instructions},
PROMPT = PromptTemplate(
    input_variables=["context"], 
    partial_variables={"format_instructions": format_instructions},
    # template="""Generate a list of 5 objects, each containing a standalone question, a non-standalone question, and its corresponding answer, about an ENTITY from the provided context. The first question should be a standalone one. The following questions should be non-standalone and should use pronouns and maintain context. Avoid using conjunctions like 'and' within the questions. Each tuple must include a clear, concise standalone question, a non-standalone question, and its corresponding accurate and complete answer. Failure to provide questions, non-standalone questions, and answers that meet these criteria will result in a penalty.
    #     {format_instructions}
    #     context: "{context}"
    #     response:  """
    # template="""Generate a list of 5 objects, each containing a standalone question, a non-standalone question, about an ENTITY from the provided context. The first question should be a standalone one. The following questions should be non-standalone and should use pronouns and maintain context. Avoid using conjunctions like 'and' within the questions. Each tuple must include a clear, concise standalone question, a non-standalone question. Failure to provide questions, non-standalone questions, that meet these criteria will result in a penalty.
    #     {format_instructions}
    #     context: "{context}"
    #     response:  """
    template="""Generate a list of 5 objects, each containing a standalone question, a non-standalone question, about an ENTITY from the provided context. The first question should be a standalone one. The following questions should be non-standalone and should use pronouns and maintain context. Avoid using conjunctions like 'and' within the questions. Each tuple must include a clear, concise standalone question, a non-standalone question. The order of the questions should form a dialogue. Failure to provide questions, non-standalone questions, that meet these criteria will result in a penalty.
        {format_instructions}
        context: "{context}"
        response:  """
    # template="""Generate a list of 5 objects, each containing a standalone question, a non-standalone question, about an ENTITY from the provided context. The first question should be a standalone one. The following questions should be non-standalone and should use pronouns and maintain context. Avoid using conjunctions like 'and' within the questions. Each tuple must include a clear, concise standalone question, a non-standalone question. Failure to provide questions, non-standalone questions, that meet these criteria will result in a penalty.
    #     {format_instructions}
    #     example: "{example}"
    #     response: "{example_response}"
    #     context: "{context}"
    #     response:  """
)

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5, streaming=True)

# string_output_parser = StrOutputParser()
json_output_parser = StrOutputParser()
dialogue_generate_chain = LLMChain(
    llm = llm,
    prompt = PROMPT,
    verbose = True,
    output_parser = json_output_parser
)

example = ""
example_response = """```json
{
"stand_alone_question": "What is Edmund Nickel 0002 primarily affiliated with?",
"non_standalone_question": "Who is primarily affiliated with RWTH Aachen University in Germany?"
},
{
"stand_alone_question": "What are Edmund Nickel 0002's research interests?",
"non_standalone_question": "Whose research interests include computer simulations, particularly in the context of 18-Krone-6?"
},
{
"stand_alone_question": "When did Edmund Nickel author a paper titled 'Computersimulation von 18-Krone-6'?",
"non_standalone_question": "Whose paper titled 'Computersimulation von 18-Krone-6' delves into the topic of computer simulations related to 18-Krone-6?"
},
{
"stand_alone_question": "What does Edmund Nickel's work likely explore?",
"non_standalone_question": "Whose work likely explores the application of computational techniques to study and understand the properties and behavior of 18-Krone-6?"
},
{
"stand_alone_question": "What does Edmund Nickel's affiliation with RWTH Aachen University highlight?",
"non_standalone_question": "Whose affiliation with RWTH Aachen University highlights their involvement in academic and research activities, particularly in the domain of computer simulations?"
}
```"""

benchmark_sample = []
for g in data[:5]:
    # g_text = g["input"]
    subgraph = g["triples"]
    # count the token 
    # total_tokens = encode_and_count(g_text)
    # print(total_tokens)
    # 5 independent questions
    # making them conversational
    output = dialogue_generate_chain.run({"context": subgraph})
    print(output)
    # data = parse_json_markdown(output)

    benchmark_sample.append(output)

# with open("dblp_sample_benchmark.json", "w") as f:
#     json.dump(benchmark_sample, f)
