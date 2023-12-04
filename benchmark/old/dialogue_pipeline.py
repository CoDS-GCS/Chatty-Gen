import random
import openai
import langchain
from utils import read_json, read_jsonl
from openai_utils import encode_and_count
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    CommaSeparatedListOutputParser,
)
from langchain.output_parsers.json import parse_json_markdown, SimpleJsonOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import LLMChain
from langchain.pydantic_v1 import BaseModel, Field, validator
import utils
import requests
import json
import os

langchain.debug = True


# gpt_summary_file = "../data/dblp/dblp_kgtext_gpt_data.json"
# gpt_subgraphs_file = "../data/dblp/dblp_subgraphs_authors.jsonl"
# gpt_subgraphs_file = "../data/dblp/dblp_subgraphs_publication.jsonl"
# gpt_subgraphs_file = "../data/yago/yago_subgraphs_person.jsonl"
gpt_subgraphs_file = "../data/yago/yago_subgraphs_movie.jsonl"

# data = read_jsonl(gpt_summary_file)
data = read_jsonl(gpt_subgraphs_file)


llm = OpenAI(model_name="text-davinci-003", temperature=0.5, streaming=False)
# llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.8, streaming=False)


# Define your desired data structure.
class QuestionSet(BaseModel):
    question: str = Field(description="question")


n_q_response_schemas = [
    ResponseSchema(
        name="output", description="a list of questions", type="List[string]"
    )
]

n_q_json_output_parser = StructuredOutputParser.from_response_schemas(
    n_q_response_schemas
)
n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()


N_Q_PROMPT = PromptTemplate(
    input_variables=[
        "example_subgraph",
        "example_n",
        "example_output",
        "subgraph",
        "n",
    ],
    partial_variables={"format_instructions": n_q_json_format_instructions},
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
    llm=llm, prompt=N_Q_PROMPT, verbose=True, output_parser=n_q_json_output_parser
)

# example_subgraph =  [{"subject": "Michael A. Kochte", "predicate": "primary affiliation", "object": "University of Stuttgart, Institute of Computer Architecture and Computer Engineering, Germany"}, {"subject": "Michael A. Kochte et al.: Trustworthy reconfigurable access to on-chip infrastructure. (2017)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Chang Liu et al.: Efficient observation point selection for aging monitoring. (2015)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Dominik Erb et al.: Test pattern generation in presence of unknown values based on restricted symbolic logic. (2014)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Stefan Hillebrecht et al.: Accurate QBF-based test pattern generation in presence of unknown values. (2013)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Hongyan Zhang et al.: GUARD: GUAranteed Reliability in Dynamically Reconfigurable Systems. (2014)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Michael A. Kochte et al.: Test exploration and validation using transaction level models. (2009)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Michael A. Kochte and Hans-Joachim Wunderlich: SAT-based fault coverage evaluation in the presence of unknown values. (2011)", "predicate": "authored by", "object": "Michael A. Kochte"}, {"subject": "Wen-Hsuan Hsu et al.: Built-In Test and Diagnosis for TSVs With Different Placement Topologies and Crosstalk Impact Ranges. (2017)", "predicate": "authored by", "object": "Michael A. Kochte"}]
# example_output = ["Can you list the papers authored by Michael A. Kochte?", "How many papers did Michael A. Kochte co-author with other researchers?","Did Michael A. Kochte author a paper titled 'Trustworthy reconfigurable access to on-chip infrastructure' in 2017?", "What is the primary affiliation of Michael A. Kochte?", "Provide the titles of papers authored by Michael A. Kochte in 2014.", "How many papers authored by Michael A. Kochte?", "Is Michael A. Kochte affiliated with the University of Stuttgart, Institute of Computer Architecture and Computer Engineering in Germany?", "What is the title of the paper co-authored by Michael A. Kochte and Hans-Joachim Wunderlich?", "When was the paper 'Test exploration and validation using transaction level models' authored by Michael A. Kochte published?", "What is the title of the most recent paper authored by Michael A. Kochte?"]
# example_n = 10
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
]
example_n = 3


"""
in bench
"""
pronoun_examples_data = {
    "he/him": [
        (
            "Albert Einstein developed the theory of relativity.",
            ("Albert Einstein", "He/Him"),
        ),
        ("Leonardo da Vinci created the Mona Lisa.", ("Leonardo da Vinci", "He/Him")),
        ("Isaac Newton formulated the laws of motion.", ("Isaac Newton", "He/Him")),
    ],
    "she/her": [
        ("Helen Keller overcame deafness and blindness.", ("Helen Keller", "She/Her")),
        (
            "Marie Curie conducted groundbreaking research in the field of radiology.",
            ("Marie Curie", "She/Her"),
        ),
        (
            "Amelia Earhart was a pioneering aviator who disappeared during a flight.",
            ("Amelia Earhart", "She/Her"),
        ),
    ],
    "it": [
        (
            "The Eiffel Tower was constructed in the late 19th century.",
            ("The Eiffel Tower", "It"),
        ),
        (
            "The ancient city of Rome was a hub of culture and power.",
            ("The ancient city of Rome", "It"),
        ),
        (
            "The Statue of Liberty was a gift from France to the United States.",
            ("The Statue of Liberty", "It"),
        ),
    ],
    "they/them": [
        (
            "The pyramids of Giza were built by ancient Egyptians.",
            ("The pyramids of Giza", "They/Them"),
        ),
        (
            "The Wright brothers invented the first successful powered aircraft.",
            ("The Wright brothers", "They/Them"),
        ),
        (
            "The Beatles were a legendary British rock band.",
            ("The Beatles", "They/Them"),
        ),
    ],
}

p_idf_response_schemas = [
    ResponseSchema(
        name="output",
        description="a tuple of an Entity and its pronouns",
        type="List[string]",
    )
]

p_idf_json_output_parser = StructuredOutputParser.from_response_schemas(
    p_idf_response_schemas
)
p_idf_json_format_instructions = p_idf_json_output_parser.get_format_instructions()

P_IDF_PROMPT = PromptTemplate(
    input_variables=[
        "query",
        "e_1_inp",
        "e_1_out",
        "e_2_inp",
        "e_2_out",
        "e_3_inp",
        "e_3_out",
        "e_4_inp",
        "e_4_out",
    ],
    # input_variables=["query", "e_1_inp", "e_1_out"],
    partial_variables={"format_instructions": p_idf_json_format_instructions},
    template="""Given a question or sentence with a single focused entity, identify that entity and provide the appropriate pronoun that refers to it.
{format_instructions}

Example,
input: "{e_1_inp}"
output: ```json
{{
    "output": ["{e_1_out[0]}", "{e_1_out[1]}"]
}}```
input: "{e_2_inp}"
output: ```json
{{
    "output": ["{e_2_out[0]}", "{e_2_out[1]}"]
}}```
input: "{e_3_inp}"
output: ```json
{{
    "output": ["{e_3_out[0]}", "{e_3_out[1]}"]
}}```
input: "{e_4_inp}"
output: ```json
{{
    "output": ["{e_4_out[0]}", "{e_4_out[1]}"]
}}```

input: "{query}"
output: """,
)

pronoun_identification_chain = LLMChain(
    llm=llm, prompt=P_IDF_PROMPT, verbose=True, output_parser=p_idf_json_output_parser
)


def select_random_pronoun_examples():
    out = []
    for k, v in pronoun_examples_data.items():
        out.append(random.choice(v))
    return out


p_sub_response_schemas = [
    ResponseSchema(
        name="output",
        description="a list of transformed questions",
        type="List[string]",
    )
]

p_sub_json_output_parser = StructuredOutputParser.from_response_schemas(
    p_sub_response_schemas
)
p_sub_json_format_instructions = p_sub_json_output_parser.get_format_instructions()

P_SUB_PROMPT = PromptTemplate(
    input_variables=[
        "query_inp",
        "query_entity",
        "query_pronouns",
        "example_entity",
        "example_pronouns",
        "example_inp",
        "example_out",
    ],
    partial_variables={"format_instructions": p_sub_json_format_instructions},
    template="""Given an entity, its pronounce and a list of questions related to a specific entity, rewrite the questions by replacing the entity's name with appropriate pronouns. The output should be a list of rewritten questions with pronouns.
{format_instructions}

Example,
entity: {example_entity}
pronouns: {example_pronouns}
input: {example_inp}
output: {example_out}

entity: {query_entity}
pronouns: {query_pronouns}
input: "{query_inp}"
output: """,
)
# output: ```json
# {{
#     "output":
# }}```

pronoun_substitution_chain = LLMChain(
    llm=llm, prompt=P_SUB_PROMPT, verbose=True, output_parser=p_sub_json_output_parser
)

example_entity = "Michael A. Kochte"
example_pronouns = "he/him"
example_inp = [
    "How many papers did Michael A. Kochte co-author with other researchers?",
    "Did Michael A. Kochte author a paper titled 'Trustworthy reconfigurable access to on-chip infrastructure' in 2017?",
    "Provide the titles of papers authored by Michael A. Kochte in 2014.",
    "How many papers authored by Michael A. Kochte?",
]
example_out = [
    "How many papers did he co-author with other researchers?",
    "Did he author a paper titled 'Trustworthy reconfigurable access to on-chip infrastructure' in 2017?",
    "Provide the titles of papers authored by him in 2014.",
    "How many papers authored by him?",
]


benchmark_sample = []
for g in data[:50]:
    # grab the subgraph
    subgraph = g["triples"]
    if len(subgraph) > 5:
        print(len(subgraph))

        n = 5
        output = None
        try:
            with get_openai_callback() as cb:
                output = n_question_generator_chain.run(
                    {
                        "example_subgraph": example_subgraph,
                        "example_n": example_n,
                        "example_output": example_output,
                        "subgraph": subgraph,
                        "n": n,
                    }
                )
                question_set = output["output"]
                print(question_set)

                question_0 = question_set[0]
                pronoun_examples = select_random_pronoun_examples()
                print(pronoun_examples)
                examples_dict = {
                    f"e_{idx+1}_inp": x[0] for idx, x in enumerate(pronoun_examples)
                }
                examples_dict.update(
                    {f"e_{idx+1}_out": list(x[1]) for idx, x in enumerate(pronoun_examples)}
                )
                print(examples_dict)
                output = pronoun_identification_chain.run(
                    {"query": question_0, **examples_dict}
                )
                question_0_ent_pron = output["output"]
                print(question_0_ent_pron)

                examples_dict = {
                    "example_inp": example_inp,
                    "example_out": example_out,
                    "example_entity": example_entity,
                    "example_pronouns": example_pronouns,
                }
                query_dict = {
                    "query_inp": question_set[1:],
                    "query_entity": question_0_ent_pron[0],
                    "query_pronouns": question_0_ent_pron[1],
                }
                output = pronoun_substitution_chain.run({**query_dict, **examples_dict})
                transformed_questions = output["output"]
                question_set_dialogue = [question_0, *transformed_questions]
                print(question_set_dialogue)
                cb_dict = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "successful_requests": cb.successful_requests,
                    "total_cost": cb.total_cost
                }

                dialogue = {
                        "dialogue": question_set_dialogue,
                        "cost": cb_dict
                }

                print(cb_dict)

                benchmark_sample.append(dialogue)
        except Exception as e:
            print(f"ERROR: {e}")
            continue



# with open("dblp_dialogues_publication.json", "w") as f:
#     json.dump(benchmark_sample, f, indent=4)

# with open("dblp_dialogues_authors.json", "w") as f:
    # json.dump(benchmark_sample, f, indent=4)

# with open("yago_dialogues_persons.json", "w") as f:
#     json.dump(benchmark_sample, f, indent=4)

with open("yago_dialogues_movies.json", "w") as f:
    json.dump(benchmark_sample, f, indent=4)

# with open("dblp_dialogues.json", "w") as f:
#     json.dump(benchmark_sample, f)
#
# data = read_json("dblp_sample_benchmark_n_question.json")
# for dialogue in data:
#     formatted_data = ",".join(dialogue)
#     # formatted_data = [item.strip("'") for item in dialogue]
#     json_data = json.loads(formatted_data)
#     # json_data = json.dumps(, indent=4)
#     print(type(json_data))
