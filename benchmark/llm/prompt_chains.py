import random
import re
import langchain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    PydanticOutputParser
)
from pydantic import BaseModel
from langchain.chains import LLMChain
from typing import List, Tuple

langchain.debug = True

llm_config = {
    'max_new_tokens': 512,
    'early_stopping': "```",
    'do_sample': True
}


class QuestionItem(BaseModel):
    question: str
    triples: List[str]

class LLMInput(BaseModel):
    output: List[QuestionItem]

class QuestionSchema(BaseModel):
    question: str
    triples: List[Tuple[str,str,str]]

class QuestionSet(BaseModel):
    output: List[QuestionSchema]


class Triples_1(BaseModel):
    triples: List[Tuple[str, str, str]]

class Triples_2(BaseModel):
    triples: List[Tuple[str, str]]

class Triples_3(BaseModel):
    triples: List[str]

class SchemaInput(BaseModel):
    output: List[str]

class Item(BaseModel):
    output: str

def trim_after_first_occurrence(text, pattern):
    # Find the first occurrence of the pattern
    match = re.search(pattern, text)
    
    # If the pattern is found, return the text up to the first occurrence
    if match:
        return text[:match.end()]
    else:
        # If the pattern is not found, return the original text
        return text + "```"

def replace_single_quotes(text):
    return re.sub(r"(?<![a-z])'|'(?![a-z])", '"', text)

def format_template_with_dict(template, values_dict):
    try:
        formatted_string = template.format(**values_dict)
        return formatted_string
    except KeyError as e:
        return f"Error: Missing key '{e.args[0]}' in the dictionary."
    except ValueError:
        return "Error: Type mismatch in the template."
    except Exception as e:
        return f"Error: error occurred: {str(e)}"


def get_question_template_chain(placeholder, llm):
    response_schemas = [
        ResponseSchema(
            name="output",
            description="a list of question templates",
            type="List[string]",
        )
    ]

    json_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # json_output_parser_with_retry = RetryWithErrorOutputParser(parser=json_output_parser)
    # json_output_parser_with_retry = RetryWithErrorOutputParser.from_llm(llm, json_output_parser)
    json_format_instructions = json_output_parser.get_format_instructions()

    SUBJECT_QUESTION_TEMPLATE_PROMPT = PromptTemplate(
        # input_variables=["example_input", "example_output_0", "example_output_1", "example_output_2", "input"],
        input_variables=[
            "example_input",
            "example_output_0",
            "example_output_1",
            "placeholder",
            "input",
        ],
        partial_variables={"format_instructions": json_format_instructions},
        # template="""Generate a list of two questions as template strings, given the 'triple'. A triple consisting of 3 elements: a subject node type, a predicate, an object node type. The output should be a list containing these different question variations.
        template="""Generate a list of 2 template strings based on a given 'triple' consisting of a subject node type, a predicate, and an object node type. Use the following steps:
        Identify entities and relationship in the triple.
        Determine a condition based on the relationship.
        Create template strings using placeholders representing the identified entity ({placeholder}) based on the condition. Output the list containing these question variations.
        {format_instructions}

        example,
        input: {example_input}
        output: [
        "{example_output_0}",
        "{example_output_1}",
        ]

        input: {input}
        """,
    )
    # "{example_output_2}",

    example_input = ("person", "worksAt", "company")
    if placeholder == "object":
        example_output = [
            # "Which company employs {subject}?",
            # "At which company does {subject} work?",
            # "What is the workplace of {subject}?",
            "Who works at {object}?",
            "Which person is employed at {object}?",
            # "At {object}, who is an employee?",
        ]
    elif placeholder == "subject":
        example_output = [
            "Which company employs {subject}?",
            "At which company does {subject} work?",
            # "What is the workplace of {subject}?",
            # "Who works at {object}?",
            # "Which person is employed at {object}?",
            # "At {object}, who is an employee?",
        ]
    else:
        raise ValueError(
            'Error: invalid value for placeholder, allowed values ["object", "subject"]'
        )

    examples_dict = {f"example_output_{idx}": x for idx, x in enumerate(example_output)}

    question_template_chain = LLMChain(
        llm=llm,
        prompt=SUBJECT_QUESTION_TEMPLATE_PROMPT,
        verbose=False,
        output_parser=json_output_parser,
    )
    payload = {
        "example_input": example_input,
        "placeholder": placeholder,
        **examples_dict,
    }

    def valid_output(output, placeholder):
        dummy_val = {placeholder: "dummy"}

        for op in output["output"]:
            if format_template_with_dict(op, dummy_val).startswith("Error"):
                return False

        return True

    def run_chain(input):
        max_retry = 3
        i = 0
        retry = False
        while not retry and i < max_retry:
            try:
                i += 1
                # in case of not valid response just retry
                output = question_template_chain.run(**input)
                if valid_output(output, placeholder):
                    break
            except Exception as e:
                print("ERROR: retry...{i}")
                continue

        if output is None:
            return None
        return output

    return {"chain": run_chain, "payload": payload}


def get_pronoun_identification_chain(llm):
    # p_idf_response_schemas = [
    #     ResponseSchema(
    #         name="output",
    #         description="a tuple of Entity and its pronouns",
    #         type="List[string]",
    #     )
    # ]
    #
    # p_idf_json_output_parser = StructuredOutputParser.from_response_schemas(
    #     p_idf_response_schemas
    # )
    # p_idf_json_format_instructions = p_idf_json_output_parser.get_format_instructions()
    p_idf_json_output_parser = PydanticOutputParser(pydantic_object=SchemaInput)
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
        llm=llm,
        prompt=P_IDF_PROMPT,
        verbose=False,
        output_parser=p_idf_json_output_parser,
    )
    pronoun_examples_data = {
        "he/him": [
            (
                "Albert Einstein developed the theory of relativity.",
                ("Albert Einstein", "He/Him"),
            ),
            (
                "Leonardo da Vinci created the Mona Lisa.",
                ("Leonardo da Vinci", "He/Him"),
            ),
            ("Isaac Newton formulated the laws of motion.", ("Isaac Newton", "He/Him")),
        ],
        "she/her": [
            (
                "Helen Keller overcame deafness and blindness.",
                ("Helen Keller", "She/Her"),
            ),
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

    def select_random_pronoun_examples():
        out = []
        for k, v in pronoun_examples_data.items():
            out.append(random.choice(v))
        return out

    pronoun_examples = select_random_pronoun_examples()
    examples_dict = {f"e_{idx+1}_inp": x[0] for idx, x in enumerate(pronoun_examples)}
    examples_dict.update(
        {f"e_{idx+1}_out": list(x[1]) for idx, x in enumerate(pronoun_examples)}
    )
    return {
        "chain": pronoun_identification_chain,
        "payload": examples_dict,
    }


def get_pronoun_substitution_chain(llm):
    # p_sub_response_schemas = [
    #     ResponseSchema(
    #         name="output",
    #         description="a list of transformed questions",
    #         type="List[string]",
    #     )
    # ]

    # p_sub_json_output_parser = StructuredOutputParser.from_response_schemas(
    #     p_sub_response_schemas
    # )
    # p_sub_json_format_instructions = p_sub_json_output_parser.get_format_instructions()
    p_sub_json_output_parser = PydanticOutputParser(pydantic_object=SchemaInput)
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

    pronoun_substitution_chain = LLMChain(
        llm=llm,
        prompt=P_SUB_PROMPT,
        verbose=False,
        output_parser=p_sub_json_output_parser,
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
    examples_dict = {
        "example_inp": example_inp,
        "example_out": example_out,
        "example_entity": example_entity,
        "example_pronouns": example_pronouns,
    }
    return {
        "chain": pronoun_substitution_chain,
        "payload": examples_dict,
    }


def get_n_question_from_subgraph_chain_without_example(llm):
    # n_q_response_schemas = [
    #     ResponseSchema(
    #         name="output", description="a list of questions", type="List[string]"
    #     )
    # ]
    #
    # n_q_json_output_parser = StructuredOutputParser.from_response_schemas(
    #     n_q_response_schemas
    # )
    # n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()

    n_q_json_output_parser = PydanticOutputParser(pydantic_object=LLMInput)
    n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()

    N_Q_PROMPT = PromptTemplate(
        input_variables=[
            "subgraph",
            "n",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGenerate a list of n questions based on a subgraph from a knowledge graph, represented as a list of triples. Each question should relate to a shared entity (e) within the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triples from the subgraph. Return each question with the triple or triples used to generate the question. Maximum number of returned triples per questions is 5\n\n{format_instructions}.\n\ninput: {subgraph}\nn: {n}\n\n### Response:```json""",
    )
    if llm["config"] is not None:
        n_question_generator_chain = LLMChain(
            llm=llm["llm"], prompt=N_Q_PROMPT,
            verbose=False,
            output_parser=n_q_json_output_parser,
            llm_kwargs=llm["config"]
        )
    else:
        n_question_generator_chain = LLMChain(
            llm=llm["llm"], prompt=N_Q_PROMPT,
            verbose=False,
            output_parser=n_q_json_output_parser)

    payload = {"stop": "```\n\n"}
    ch = n_question_generator_chain 
    def post_processor(llm_result, trace_inputs=None, trace=None):
        for generation in llm_result.generations:
            ## update outputs with before and after for given trace
            generation_0_original = generation[0].text

            if generation[0].text.startswith("```json"):
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text[7:], "```")
            else:
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text, "```")
            if not('\"output\":' in trimmed_with_backtick_at_end or '"output":' in trimmed_with_backtick_at_end):
                generation[0].text = "```json\n{\n    \"output\":" + trimmed_with_backtick_at_end[:-4] + "\n}\n```" if len(trimmed_with_backtick_at_end) >= 4 else exec("raise ValueError('error backtick mismatch.')")
            else:
                generation[0].text = "```json" + trimmed_with_backtick_at_end
            
            generation_0_processed = generation[0].text
            ## update outputs with before and after for given trace
            if trace:
                trace_outputs = {
                    "generated": generation_0_original,
                    "processed": generation_0_processed
                }
                trace.add_inputs_and_outputs(inputs=trace_inputs, outputs=trace_outputs)
        output = [
            # Get the text of the top generated string.
            {
                ch.output_key: ch.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if ch.return_final_only:
            output = [{ch.output_key: r[ch.output_key]} for r in output]
        output = output[0][ch.output_key].dict()
        print(output)
        return output
    return {"chain": n_question_generator_chain, "payload": payload, "prompt": N_Q_PROMPT, "post_processor": post_processor}

def get_n_question_from_subgraph_chain_with_example():
    # Define your desired data structure.

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
    ]
    example_output = [
        "Can you list the papers authored by Michael A. Kochte?",
        "How many papers did Michael A. Kochte co-author with other researchers?",
        "Did Michael A. Kochte author a paper titled 'Trustworthy reconfigurable access to on-chip infrastructure' in 2017?",
    ]
    example_n = 3

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
        llm=llm, prompt=N_Q_PROMPT,
        verbose=False,
        output_parser=n_q_json_output_parser
    )
    payload = {
        "example_subgraph": example_subgraph,
        "example_n": example_n,
        "example_output": example_output,
    }
    return {"chain": n_question_generator_chain, "payload": payload}

def get_n_question_from_schema_chain_without_example(llm):
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
            "seed",
            "schema",
            "n",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""Generate a list of n questions based on a seed, and subgraph schema from a knowledge graph. Each question should relate to a shared entity (e) within the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triplets from the subgraph.
        {format_instructions}

        seed: {seed}
        input: {schema}
        n: {n}
        output: """,
    )

    n_question_generator_chain = LLMChain(
        llm=llm, prompt=N_Q_PROMPT,
        verbose=False,
        output_parser=n_q_json_output_parser
    )
    payload = {}
    return {"chain": n_question_generator_chain, "payload": payload}

def get_answer_from_question_and_triple_zero_shot(llm:dict):
    n_q_response_schemas = [
        ResponseSchema(
            name="sparql", description="a SPARQL query", type="string"
        )
    ]

    n_q_json_output_parser = StructuredOutputParser.from_response_schemas(
        n_q_response_schemas
    )

    n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()

    N_Q_PROMPT = PromptTemplate(
        input_variables=[
            "question",
            "triples",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGiven a question and set of triples used to generate this question. Create the SPARQL query representing the question. Do not include the answer in the query.\n{format_instructions}\n\nquestion: {question}\ntriples: {triples}\n\n### Response:```json""",
        # template="""### Instruction:\nGiven a question and set of triples of form (subject, predicate, object) where object is unknown, used to generate this question. Write the SPARQL query representing the question. You must use only the given URIs.\n{format_instructions}\n\nquestion: {question}\ntriples: {triples}\n\n### Response:```json""",
    )

    n_answer_generator_chain = LLMChain(
        llm=llm["llm"], prompt=N_Q_PROMPT,
        verbose=False,
        output_parser=n_q_json_output_parser
    )
    ch = n_answer_generator_chain

    def post_processor(llm_result, trace_inputs=None, trace=None):
        for generation in llm_result.generations:
            ## update outputs with before and after for given trace
            generation_0_original = generation[0].text

            if generation[0].text.startswith("```json"):
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text[7:], "```")
            else:
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text, "```")
#             if not('\"sparql\":' in generation[0].text or '"sparql":' in generation[0].text):
#                 generation[0].text = "```json\n{\n    \"sparql\":" + trimmed_with_backtick_at_end[:-4] + "\n}\n```" if len(trimmed_with_backtick_at_end) >= 4 else exec("raise ValueError('error backtick mismatch.')")
#             else:
            generation[0].text = "```json" + trimmed_with_backtick_at_end
            print("gen-text: ", generation[0].text)
            
            generation_0_processed = generation[0].text
            ## update outputs with before and after for given trace
            if trace:
                trace_outputs = {
                    "generated": generation_0_original,
                    "processed": generation_0_processed
                }
                trace.add_inputs_and_outputs(inputs=trace_inputs, outputs=trace_outputs)
        output = [
            # Get the text of the top generated string.
            {
                ch.output_key: ch.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if ch.return_final_only:
            output = [{ch.output_key: r[ch.output_key]} for r in output]
        output = output[0][ch.output_key]
        print(output)
        return output
    return {"chain": n_answer_generator_chain, "payload": {}, "prompt": N_Q_PROMPT, "post_processor": post_processor}

def get_target_answer_from_triples(llm:dict):
    n_q_response_schemas = [
        ResponseSchema(
            name="target", description="a part of given triple", type="string"
        )
    ]

    n_q_json_output_parser = StructuredOutputParser.from_response_schemas(
        n_q_response_schemas
    )

    n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()

    N_Q_PROMPT = PromptTemplate(
        input_variables=[
            "question",
            "triples",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""Given a question and set of triples used to generate this question. Return the target answer:
        {format_instructions}

        question: {question}
        triples: {triples}
        target: """,
    )

    n_answer_target_chain = LLMChain(
        llm=llm["llm"], prompt=N_Q_PROMPT,
        verbose=False,
        output_parser=n_q_json_output_parser
    )
    return {"chain": n_answer_target_chain, "payload": {}}

def get_n_question_from_summarized_subgraph_chain_without_example(llm):
    # n_q_response_schemas = [
    #     ResponseSchema(
    #         name="output", description="a list of questions", type="List[string]"
    #     )
    # ]
    #
    # n_q_json_output_parser = StructuredOutputParser.from_response_schemas(
    #     n_q_response_schemas
    # )
    # n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()
    # n_q_json_output_parser = PydanticOutputParser(pydantic_object=QuestionSet)
    n_q_json_output_parser = PydanticOutputParser(pydantic_object=LLMInput)
    n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()

    # N_Q_PROMPT = PromptTemplate(
    #     input_variables=[
    #         "subgraph",
    #         "n",
    #     ],
    #     partial_variables={"format_instructions": n_q_json_format_instructions},
    #     template="""Generate a list of n questions based on a subgraph from a knowledge graph, represented as a list of triples. Each triple consists of (subject, predicate, object). Each question should relate to a shared entity (e) within the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triples from the subgraph.  Return each question with the triple or triples used to generate the question.   {format_instructions}
    #
    #     input: {subgraph}
    #     n: {n}
    #     output: """,
    # )

    N_Q_PROMPT_1 = PromptTemplate(
        input_variables=[
            "subgraph",
            "n",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGenerate a list of n questions based on a subgraph from a knowledge graph, represented as a list of triples. Each question should relate to a shared entity (e) within the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triples from the subgraph. Return each question with the triple or triples used to generate the question. Maximum number of returned triples per questions is 5\n\n{format_instructions}.\n\ninput: {subgraph}\nn: {n}\n\n### Response:```json""",
    )

    N_Q_PROMPT_2 = PromptTemplate(
        input_variables=[
            "subgraph",
            "n",
            "entity"
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGenerate a list of n questions based on the given entity and its subgraph. The subgraph is represented as a list of triples. Each question should ask about a fact from the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should include the entity. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triples from the subgraph. Return each question with the triple or triples used to generate the question. Maximum number of returned triples per questions is 5\n\n{format_instructions}.\n\ninput: {subgraph}\nentity: {entity}\nn: {n}\n\n### Response:```json""",
    )

    N_Q_PROMPT_3 = PromptTemplate(
        input_variables=[
            "subgraph",
            "n",
            "entity"
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGenerate a list of n questions based on the given entity and its subgraph. The subgraph is represented as a list of triples. Each question must ask about a fact from the subgraph and must fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question must include the entity. Each question must be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triples from the subgraph. Return each question and the associated triple used to generate the question.\n\n{format_instructions}.\n\ninput: {subgraph}\nentity: "{entity}"\nn: {n}\n\n### Response:```json""",
    )

    N_Q_PROMPT_3_forced = PromptTemplate(
        input_variables=[
            "subgraph",
            "n",
            "entity"
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGenerate a list of n questions based on the given entity and its subgraph. The subgraph is represented as a list of triples. Each question must ask about a fact from the triples in the subgraph and must fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question must include the entity. Each question must be answerable solely from the information in the provided subgraph without explicitly mentioning it. For each question, choose the triples from the input subgraph which was used to generate the question. Return both the question and the exact triple from the subgraph that it was based on.\n\n{format_instructions}.\n\ninput: {subgraph}\nentity: "{entity}"\nn: {n}\n\n### Response:```json""",
    )

    # N_Q_PROMPT = N_Q_PROMPT_1 # before updates
    # N_Q_PROMPT = N_Q_PROMPT_2 # this is v1 in commit
    # N_Q_PROMPT = N_Q_PROMPT_3 # this is v2 in commit - revert the change of v2 while keeping must
    N_Q_PROMPT = N_Q_PROMPT_3_forced # this is v2 in commit - forced triple use

    if llm["config"] is not None:
        n_question_generator_chain = LLMChain(
            llm=llm["llm"], prompt=N_Q_PROMPT,
            verbose=False,
            output_parser=n_q_json_output_parser,
            llm_kwargs=llm["config"]
        )
    else:
        n_question_generator_chain = LLMChain(
            llm=llm["llm"], prompt=N_Q_PROMPT,
            verbose=False,
            output_parser=n_q_json_output_parser
        )

    payload = {"stop": "```\n\n"}
    ch = n_question_generator_chain 
    def post_processor(llm_result, trace_inputs=None, trace=None):
        for generation in llm_result.generations:
            ## update outputs with before and after for given trace
            generation_0_original = generation[0].text

            if generation[0].text.startswith("```json"):
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text[7:], "```")
            else:
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text, "```")

            if not('\"output\":' in trimmed_with_backtick_at_end or '"output":' in trimmed_with_backtick_at_end):
            # if not('\"output\":' in generation[0].text or '"output":' in generation[0].text):
                generation[0].text = "```json\n{\n    \"output\":" + trimmed_with_backtick_at_end[:-4] + "\n}\n```" if len(trimmed_with_backtick_at_end) >= 4 else exec("raise ValueError('error backtick mismatch.')")
            else:
                generation[0].text = "```json" + trimmed_with_backtick_at_end
            print("gen-text: ", generation[0].text)

            generation_0_processed = generation[0].text
            ## update outputs with before and after for given trace
            if trace:
                trace_outputs = {
                    "generated": generation_0_original,
                    "processed": generation_0_processed
                }
                trace.add_inputs_and_outputs(inputs=trace_inputs, outputs=trace_outputs)

        output = [
            # Get the text of the top generated string.
            {
                ch.output_key: ch.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if ch.return_final_only:
            output = [{ch.output_key: r[ch.output_key]} for r in output]
        output = output[0][ch.output_key].dict()
        print(output)
        return output
    return {"chain": n_question_generator_chain, "payload": payload, "prompt": N_Q_PROMPT, "post_processor": post_processor}

def get_n_question_from_summarized_subgraph_chain_without_example_without_triple(llm):
    n_q_response_schemas = [
        ResponseSchema(
            name="output", description="a list of questions", type="List[string]"
        )
    ]
   
    n_q_json_output_parser = StructuredOutputParser.from_response_schemas(
        n_q_response_schemas
    )
    n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()

    N_Q_PROMPT_1 = PromptTemplate(
        input_variables=[
            "subgraph",
            "n",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGenerate a list of n questions based on a subgraph from a knowledge graph, represented as a list of triples. Each question should relate to a shared entity (e) within the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triples from the subgraph.{format_instructions}\n\ninput: {subgraph}\nn: {n}\n\n### Response:```json""",
    )
    
    N_Q_PROMPT_2 = PromptTemplate(
        input_variables=[
            "subgraph",
            "n",
            "entity"
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGenerate a list of n questions based on the given entity and its subgraph. The subgraph is represented as a list of triples. Each question should ask about a fact from the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should include the entity. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triples from the subgraph.{format_instructions}\n\ninput: {subgraph}\nentity: {entity}\nn: {n}\n\n### Response:```json""",
    )

    N_Q_PROMPT_3 = PromptTemplate(
        input_variables=[
            "subgraph",
            "n",
            "entity"
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGenerate a list of n questions based on the given entity and its subgraph. The subgraph is represented as a list of triples. Each question must ask about a fact from the subgraph and must fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question must include the entity. Each question must be answerable solely from the information in the provided subgraph without explicitly mentioning it. Return list of questions.{format_instructions}\n\ninput: {subgraph}\nentity: {entity}\nn: {n}\n\n### Response:```json""",
    )
    
    N_Q_PROMPT_4 = PromptTemplate(
        input_variables=[
            "subgraph",
            "n",
            "entity"
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGenerate a list of n questions based on the given entity and its subgraph. The subgraph is represented as a list of triples. Each question must ask about a fact from the subgraph and must fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question must include the entity. Each question must be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triples from the subgraph. Return list of questions.{format_instructions}\n\ninput: {subgraph}\nentity: "{entity}"\nn: {n}\n\n### Response:```json""",
    )
    # N_Q_PROMPT = N_Q_PROMPT_1
    # N_Q_PROMPT = N_Q_PROMPT_2 # this is v1 in commit
    # N_Q_PROMPT = N_Q_PROMPT_3 # this is v2 in commit
    N_Q_PROMPT = N_Q_PROMPT_4 # this is v3 in commit

    if llm["config"] is not None:
        n_question_generator_chain = LLMChain(
            llm=llm["llm"], prompt=N_Q_PROMPT,
            verbose=False,
            output_parser=n_q_json_output_parser,
            llm_kwargs=llm["config"]
        )
    else:
        n_question_generator_chain = LLMChain(
            llm=llm["llm"], prompt=N_Q_PROMPT,
            verbose=False,
            output_parser=n_q_json_output_parser
        )
    payload = {"stop": "```\n\n"}
    ch = n_question_generator_chain
    
    def post_processor(llm_result, trace_inputs=None, trace=None):
        for generation in llm_result.generations:

            ## update outputs with before and after for given trace
            generation_0_original = generation[0].text

            if generation[0].text.startswith("```json"):
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text[7:], "```")
            else:
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text, "```")
            if not('\"output\":' in trimmed_with_backtick_at_end or '"output":' in trimmed_with_backtick_at_end):
            # if not('\"output\":' in generation[0].text or '"output":' in generation[0].text):
                generation[0].text = "```json\n{\n    \"output\":" + trimmed_with_backtick_at_end[:-4] + "\n}\n```" if len(trimmed_with_backtick_at_end) >= 4 else exec("raise ValueError('error backtick mismatch.')")
            else:
                generation[0].text = "```json" + trimmed_with_backtick_at_end
            print("gen-text: ", generation[0].text)

            generation_0_processed = generation[0].text
            ## update outputs with before and after for given trace
            if trace:
                trace_outputs = {
                    "generated": generation_0_original,
                    "processed": generation_0_processed
                }
                trace.add_inputs_and_outputs(inputs=trace_inputs, outputs=trace_outputs)

        output = [
            # Get the text of the top generated string.
            {
                ch.output_key: ch.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if ch.return_final_only:
            output = [{ch.output_key: r[ch.output_key]} for r in output]
        output = output[0][ch.output_key]
        print(output)
        return output
    return {"chain": n_question_generator_chain, "payload": payload, "prompt": N_Q_PROMPT,
    "post_processor": post_processor}


def get_triple_for_question_given_subgraph_chain_without_example(llm):
    n_q_json_output_parser_1 = PydanticOutputParser(pydantic_object=Triples_1)
    n_q_json_output_parser_2 = PydanticOutputParser(pydantic_object=Triples_2)
    n_q_json_output_parser_3 = PydanticOutputParser(pydantic_object=Triples_3)
    n_q_json_format_instructions = n_q_json_output_parser_1.get_format_instructions()

    N_Q_PROMPT_1 = PromptTemplate(
        input_variables=[
            "subgraph",
            "question",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGiven a question and a subgraph extracted from a knowledge graph, where the subgraph is represented as a list of triples, your task is to identify the specific triples within this subgraph that accurately represent the information needed to address the question. Each triple comprises a subject, a predicate, and an object, denoting a relationship between entities. Your objective is to discern and select the triples that contain relevant information essential for answering the question at hand. You must select triples from given triple list.\n\n{format_instructions}.\n\ninput: {subgraph}\nquestion: {question}\n\n### Response:```json""",
    )
    N_Q_PROMPT_2 = PromptTemplate(
        input_variables=[
            "subgraph",
            "question",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction:\nGiven a question and a list of triples, your task is to choose the specific triples within this list that were used to form the question. Each triple comprises a subject, a predicate, and an object, denoting a relationship between entities. You must choose triples from given triple list.\n\n{format_instructions}.\n\ninput: {subgraph}\nquestion: {question}\n\n### Response:```json""",
    )

    # N_Q_PROMPT = N_Q_PROMPT_1
    N_Q_PROMPT = N_Q_PROMPT_2 # commit with v2

    if llm['config'] is not None:
        triple_bind_chain = LLMChain(
            llm=llm["llm"], prompt=N_Q_PROMPT,
            verbose=False,
            output_parser=n_q_json_output_parser_1,
            llm_kwargs=llm["config"]
        )
    else:
        triple_bind_chain = LLMChain(
            llm=llm["llm"], prompt=N_Q_PROMPT,
            verbose=False,
            output_parser=n_q_json_output_parser_1)
    payload = {"stop": "```\n\n"}
    ch = triple_bind_chain
    

    def post_processor(llm_result, trace_inputs=None, trace=None):
        for generation in llm_result.generations:

            ## update outputs with before and after for given trace
            generation_0_original = generation[0].text

            if generation[0].text.startswith("```json"):
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text[7:], "```")
            else:
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text, "```")
            if not('\"triples\":' in trimmed_with_backtick_at_end or '"triples":' in trimmed_with_backtick_at_end):
                generation[0].text = "```json\n{\n    \"triples\":" + trimmed_with_backtick_at_end[:-4] + "\n}\n```" if len(trimmed_with_backtick_at_end) >= 4 else exec("raise ValueError('error backtick mismatch.')")
            else:
                generation[0].text = "```json" + trimmed_with_backtick_at_end
            print("gen-text: ", generation[0].text)

            # handle use of single quote - replace it with double quote
            generation[0].text = replace_single_quotes(generation[0].text)
            # if "'" in generation[0].text:
            #     generation[0].text = generation[0].text.replace("'", '"')

            # handle use of round braces - replace it with square braces
            if "(" in generation[0].text or ")" in generation[0].text:
                generation[0].text = generation[0].text.replace("(", "[")
                generation[0].text = generation[0].text.replace(")", "]")
            
            # handle single triple only - put it inside list of list form
            if "[[" not in generation[0].text and generation[0].text.count("[") == 1  and generation[0].text.count("]") == 1:
                generation[0].text = generation[0].text.replace("[", '[[')
                generation[0].text = generation[0].text.replace("]", ']]')
            
            if "[{" in generation[0].text:
                generation[0].text = generation[0].text.replace("[{", '{')
                generation[0].text = generation[0].text.replace("}]", '}')
            
            print("gen-text: ", generation[0].text)
            generation_0_processed = generation[0].text
            ## update outputs with before and after for given trace
            if trace:
                trace_outputs = {
                    "generated": generation_0_original,
                    "processed": generation_0_processed
                }
                trace.add_inputs_and_outputs(inputs=trace_inputs, outputs=trace_outputs)

        is_parsed = False
        try:
            if not is_parsed:
                output = [
                    # Get the text of the top generated string.
                    {
                        ch.output_key: n_q_json_output_parser_1.parse_result(generation),
                        "full_generation": generation,
                    }
                    for generation in llm_result.generations
                ]
                is_parsed = True
        except Exception as e:
            print("Parser 1 : Couldn't parse it")

        try:
            if not is_parsed:
                output = [
                    # Get the text of the top generated string.
                    {
                        ch.output_key: n_q_json_output_parser_2.parse_result(generation),
                        "full_generation": generation,
                    }
                    for generation in llm_result.generations
                ]
                is_parsed = True
        except Exception as e:
            print("Parser 2 : Couldn't parse it")

        try:
            if not is_parsed:
                output = [
                    # Get the text of the top generated string.
                    {
                        ch.output_key: n_q_json_output_parser_3.parse_result(generation),
                        "full_generation": generation,
                    }
                    for generation in llm_result.generations
                ]
                output = [{ch.output_key: [x[ch.output_key]]} for x in output]
                is_parsed = True
        except Exception as e:
            print("Parser 3 : Couldn't parse it")

        if not is_parsed:
            raise Exception("Not able to parse it")

        if ch.return_final_only:
            output = [{ch.output_key: r[ch.output_key]} for r in output]
        output = output[0][ch.output_key].dict()
        return output["triples"]

    return {"chain": triple_bind_chain, "payload": payload, "prompt": N_Q_PROMPT, "post_processor": post_processor}

def get_n_question_from_subgraph_chain_using_seed_entity(llm):
    n_q_json_output_parser = PydanticOutputParser(pydantic_object=LLMInput)
    n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()

    N_Q_PROMPT = PromptTemplate(
        input_variables=[
            "e",
            "subgraph",
            "n",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""Generate a list of n questions based on a subgraph from a knowledge graph, represented as a list of triples. Each question should relate to a shared entity (e) within the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triples from the subgraph. Return each question with the triple or triples used to generate the question. Maximum number of returned triples per questions is 5 {format_instructions}.

        e: {e}
        input: {subgraph}
        n: {n}
        output: """,
    )

    n_question_generator_chain = LLMChain(
        llm=llm["llm"], prompt=N_Q_PROMPT,
        verbose=True,
        output_parser=n_q_json_output_parser
    )
    payload = {}
    return {"chain": n_question_generator_chain, "payload": payload}

def get_n_question_from_subgraph_chain_using_seed_entity_and_type(llm):
    n_q_json_output_parser = PydanticOutputParser(pydantic_object=LLMInput)
    n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()

    N_Q_PROMPT = PromptTemplate(
        input_variables=[
            "e",
            "e_type",
            "subgraph",
            "n",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""Generate a list of n questions based on a subgraph from a knowledge graph, represented as a list of triples. Each question should relate to a shared entity (e) within the subgraph and should fall into one of the following categories: list, count, boolean, wh (open-ended), or date-related questions. Each question should be answerable solely from the information in the provided subgraph without explicitly mentioning it. The questions can be equivalent to one or two triples from the subgraph. Return each question with the triple or triples used to generate the question. Maximum number of returned triples per questions is 5 {format_instructions}.

        e: {e}
        e type: {e_type}
        input: {subgraph}
        n: {n}
        output: """,
    )

    n_question_generator_chain = LLMChain(
        llm=llm["llm"], prompt=N_Q_PROMPT,
        verbose=False,
        output_parser=n_q_json_output_parser
    )
    payload = {}
    return {"chain": n_question_generator_chain, "payload": payload}

def get_representative_label_for_type(llm:dict):
    n_q_response_schemas = [
        ResponseSchema(
            name="predicate", description="The most representative predicate", type="string"
        )
    ]

    n_q_json_output_parser = StructuredOutputParser.from_response_schemas(
        n_q_response_schemas
    )
    n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()

    N_Q_PROMPT = PromptTemplate(
        input_variables=[
            "node_type",
            "predicates",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template="""### Instruction: Given the specified node type {node_type}  and its associated
        predicates {predicates}, choose a suitable predicate to serve as a label for this type.
        {format_instructions}.\n\n### Response:```json""",
    )

    if llm["config"] is not None:
        n_question_generator_chain = LLMChain(
            llm=llm["llm"], prompt=N_Q_PROMPT,
            verbose=False,
            output_parser=n_q_json_output_parser,
            llm_kwargs=llm["config"]
        )
    else:
        n_question_generator_chain = LLMChain(
            llm=llm["llm"], prompt=N_Q_PROMPT,
            verbose=False,
            output_parser=n_q_json_output_parser)

    payload = {"stop": "```\n\n"}
    return {"chain": n_question_generator_chain, "payload": payload}


def get_pronoun_identification_and_substitution_chain_without_example(llm):
    p_sub_json_output_parser = PydanticOutputParser(pydantic_object=SchemaInput)
    p_sub_json_format_instructions = p_sub_json_output_parser.get_format_instructions()

    # P_SUB_PROMPT = PromptTemplate(
    #     input_variables=[
    #         "entity",
    #         "questions",
    #     ],
    #     partial_variables={"format_instructions": p_sub_json_format_instructions},
    #     template="""Given an entity and a set of questions or sentences focused on this entity, Choose the appropriate pronoun that refers to it. Return the questions with the entity replaced with its pronoun.
    # {format_instructions}
    #
    # entity: {entity}
    # input: "{questions}"
    # output: """,
    # )

    P_SUB_PROMPT = PromptTemplate(
        input_variables=[
            "entity",
            "questions",
        ],
        partial_variables={"format_instructions": p_sub_json_format_instructions},
        template="""### Instruction:\nGiven an entity and a set of questions or sentences focused on this entity, choose the appropriate pronoun that refers to it. Replace the entity with its pronoun in the questions and return the modified questions. Ensure that the modified questions do not contain the original entity and that the pronoun used in the modified questions is contextually appropriate and grammatically correct.
    {format_instructions}\n\nentity: {entity}\ninput: "{questions}"\n\n### Response:```json""",
    )

    if llm["config"] is not None:
        pronoun_substitution_chain = LLMChain(
            llm=llm["llm"],
            prompt=P_SUB_PROMPT,
            verbose=False,
            output_parser=p_sub_json_output_parser,
            llm_kwargs=llm["config"]
        )
    else:
        pronoun_substitution_chain = LLMChain(
            llm=llm["llm"],
            prompt=P_SUB_PROMPT,
            verbose=False,
            output_parser=p_sub_json_output_parser,
        )
    ch = pronoun_substitution_chain
    
    def post_processor(llm_result, trace_inputs=None, trace=None):
        for generation in llm_result.generations:

            ## update outputs with before and after for given trace
            generation_0_original = generation[0].text
            if generation[0].text.startswith("```json"):
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text[7:], "```")
            else:
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text, "```")
            if not('\"output\":' in trimmed_with_backtick_at_end or '"output":' in trimmed_with_backtick_at_end):
#            if not('\"output\":' in generation[0].text or '"output":' in generation[0].text):
                generation[0].text = "```json\n{\n    \"output\":" + trimmed_with_backtick_at_end[:-4] + "\n}\n```" if len(trimmed_with_backtick_at_end) >= 4 else exec("raise ValueError('error backtick mismatch.')")
            else:
                generation[0].text = "```json" + trimmed_with_backtick_at_end
            print("gen-text: ", generation[0].text)

            generation_0_processed = generation[0].text
            ## update outputs with before and after for given trace
            if trace:
                trace_outputs = {
                    "generated": generation_0_original,
                    "processed": generation_0_processed
                }
                trace.add_inputs_and_outputs(inputs=trace_inputs, outputs=trace_outputs)
        output = [
            # Get the text of the top generated string.
            {
                ch.output_key: ch.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if ch.return_final_only:
            output = [{ch.output_key: r[ch.output_key]} for r in output]
        output = output[0][ch.output_key].dict()
        print(output)
        return output

    return {
        "chain": pronoun_substitution_chain, "payload": {}, "prompt": P_SUB_PROMPT,"post_processor": post_processor
    }


def get_validate_question_quality_old(llm):

    n_q_response_schemas = [
        ResponseSchema(
            name="output", description="valid or not valid", type="string"
        )
    ]

    n_q_json_output_parser = StructuredOutputParser.from_response_schemas(
        n_q_response_schemas
    )
    n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()


    P_SUB_PROMPT = PromptTemplate(
        input_variables=[
            "questions",
        ],
        # partial_variables={"format_instructions": n_q_json_format_instructions},
        template=""" Given a set of questions about a specified entity, determine the validity of the input based on the following criteria:
        1. The first question must explicitly mention the specified entity and cannot use a pronoun to refer to it.
        2. Subsequent questions should be relevant and appropriately formulated.
        If both conditions are met, consider the input "Valid." Otherwise, consider it "Not Valid."

    Input:  "{questions}"
    output: """,
    )

    question_validation_chain = LLMChain(
        llm=llm,
        prompt=P_SUB_PROMPT,
        verbose=False,
        # output_parser=n_q_json_output_parser,
    )

    return {
        "chain": question_validation_chain, "payload": {}
    }

def get_validate_question_quality(llm):

    n_q_response_schemas = [
        ResponseSchema(
            name="output", description="valid or not valid", type="string"
        )
    ]

    n_q_json_output_parser = StructuredOutputParser.from_response_schemas(
        n_q_response_schemas
    )
    n_q_json_format_instructions = n_q_json_output_parser.get_format_instructions()


    P_SUB_PROMPT = PromptTemplate(
        input_variables=[
            "entity",
            "dialogue",
        ],
        partial_variables={"format_instructions": n_q_json_format_instructions},
        template = """
        Given a dialogue comprising a list of questions about the particular entity {entity}, assess whether the dialogue is valid or invalid according to the following guidelines:
        1. The initial question in the dialogue must directly reference the entity itself, avoiding generic references.
        2. Subsequent questions within the dialogue must adhere to grammatical correctness.
        
        "entity": {entity}
        "dialogue" : {dialogue}
        {format_instructions}
        
        output: 
        """
        # template="""Given an entity and a dialogue (defined as a list of questions) about that entity, determine the dialogue is valid or invalid based on the following criteria:
        #
        # 1. The first question in the list should be specific to the entity by mentioning it directly (not a generic entity class).
        # 2. The subsequent questions from the list must be grammatically correct.
        #
        # "entity" : {entity}
        # "dialogue" : {dialogue}
        # {format_instructions}
        #
        # output: """,
    )

    question_validation_chain = LLMChain(
        llm=llm,
        prompt=P_SUB_PROMPT,
        verbose=True,
        output_parser=PydanticOutputParser(pydantic_object=Item)
    )

    return {
        "chain": question_validation_chain, "payload": {}
    }

def singleshot_dialogue_chain(llm):
    print(llm)

    class QuestionItem(BaseModel):
        original: str
        transformed: str
        sparql: str
        triple: str

    class QuestionSet(BaseModel):
        output: List[QuestionItem]

    output_parser = PydanticOutputParser(pydantic_object=QuestionSet)
    format_instructions = output_parser.get_format_instructions()
    
    PROMPT = PromptTemplate(
        input_variables=[
            "entity",
            "label_subgraph",
            "query_subgraph",
            "n",
        ],
        partial_variables={"format_instructions": format_instructions},
        template = (
        "You are tasked with generating a set of questions based on provided entities and their corresponding subgraphs. Each question will be represented by an instance of the `QuestionItem` class. The process will involve forming both standalone and dialogue-contextual questions, and constructing SPARQL queries to retrieve answers from a knowledge graph."
        "**Instructions:**"
        "1. **Inputs:**"
        "- `entity`: A string representing the label for an entity."
        "- `label triples`: A list of triples, where each triple contains subject, predicate, and object labels. These labels will be used to form the questions."
        "- `query triples`: A list of triples, where each triple contains subject, predicate, and object URIs. These URIs will be used to form the SPARQL queries."
        "- `n`: An integer representing the number of questions to generate.\n"
        "2. **Question Formation:**"
        "- Each question will be based on the provided `label triples` and should incorporate the entity label."
        "- Questions should be clear, concise, and relevant to the subgraph information."
        "- For each question, create a 'transformed' version suitable for a dialogue by substituting the entity label with a pronoun.\n"
        "3. **SPARQL Query Formation:**"
        "- Each question should have an associated SPARQL query that uses the `query triples` to retrieve the correct answer."
        "- Ensure that the SPARQL query corresponds accurately to the question.\n"
        "4. **QuestionItem Structure:**"
        "- `original`: The original standalone question based on the entity and label triples."
        "- `transformed`: The transformed version of the question for use in a dialogue, substituting the entity label with a pronoun."
        "- `sparql`: The SPARQL query corresponding to the original question, using the query triples URIs."
        "- `triple`: The original triple from the label triples used to form the question."
        "\n'entity': {entity}\n'n': {n}\n'label triples' : {label_subgraph}\n'query triples' : {query_subgraph}\n{format_instructions}\n\n```json"
        )
    )
    llm_conf = {
        'max_new_tokens': 650,
        # 'stop_strings': ["```", "```\n\n", "```\n", "\n```"],
        'early_stopping': "```",
        'do_sample': True,
        # 'num_beams': 5,
        # 'num_return_sequences': 3
    }

    ch = None
    if llm["config"] is not None:
        singleshot_chain = LLMChain(
            llm=llm["llm"], 
            prompt=PROMPT,
            verbose=True,
            llm_kwargs=llm_conf,
            # llm["config"],
            output_parser=output_parser
        )
    else:
        singleshot_chain = LLMChain(
            llm=llm["llm"], 
            prompt=PROMPT,
            verbose=True,
            output_parser=output_parser
        )

    ch = singleshot_chain
    payload = {"stop": "```\n\n"}
    
    def post_processor(llm_result, trace_inputs=None, trace=None):
        for generation in llm_result.generations:

            ## update outputs with before and after for given trace
            generation_0_original = generation[0].text
            if generation[0].text.startswith("```json"):
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text[7:], "```")
            else:
                trimmed_with_backtick_at_end = trim_after_first_occurrence(generation[0].text, "```")
            if not('\"output\":' in trimmed_with_backtick_at_end or '"output":' in trimmed_with_backtick_at_end):
                generation[0].text = "```json\n{\n    \"output\":" + trimmed_with_backtick_at_end[:-4] + "\n}\n```" if len(trimmed_with_backtick_at_end) >= 4 else exec("raise ValueError('error backtick mismatch.')")
            else:
                generation[0].text = "```json" + trimmed_with_backtick_at_end
            print("gen-text: ", generation[0].text)

            generation_0_processed = generation[0].text
            ## update outputs with before and after for given trace
            if trace:
                trace_outputs = {
                    "generated": generation_0_original,
                    "processed": generation_0_processed
                }
                trace.add_inputs_and_outputs(inputs=trace_inputs, outputs=trace_outputs)
        output = [
            # Get the text of the top generated string.
            {
                ch.output_key: ch.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if ch.return_final_only:
            output = [{ch.output_key: r[ch.output_key]} for r in output]
        output = output[0][ch.output_key].dict()
        return output

    return {
        "chain": singleshot_chain, "payload": {}, "prompt": PROMPT, "payload": payload, "post_processor": post_processor
    }



def get_prompt_chains():
    prompt_chains = {
        "question_template_chain": get_question_template_chain,
        "pronoun_identification_chain": get_pronoun_identification_chain,
        "pronoun_substitution_chain": get_pronoun_substitution_chain,
        "n_question_from_subgraph_chain_with_example": get_n_question_from_subgraph_chain_with_example,
        "n_question_from_subgraph_chain_without_example": get_n_question_from_subgraph_chain_without_example,
        "n_question_from_schema_chain_without_example": get_n_question_from_schema_chain_without_example,
        "n_question_from_summarized_subgraph_chain_without_example": get_n_question_from_summarized_subgraph_chain_without_example,
        "get_answer_from_question_and_triple_zero_shot": get_answer_from_question_and_triple_zero_shot,
        "get_target_answer_from_triples": get_target_answer_from_triples,
        "get_n_question_from_subgraph_chain_using_seed_entity": get_n_question_from_subgraph_chain_using_seed_entity,
        "get_n_question_from_subgraph_chain_using_seed_entity_and_type": get_n_question_from_subgraph_chain_using_seed_entity_and_type,
        "get_representative_label_for_type": get_representative_label_for_type,
        "get_pronoun_identification_and_substitution_chain_without_example": get_pronoun_identification_and_substitution_chain_without_example,
        "get_validate_question_quality": get_validate_question_quality,
        "n_question_from_summarized_subgraph_chain_without_example_without_triple": get_n_question_from_summarized_subgraph_chain_without_example_without_triple,
        "get_triple_for_question_given_subgraph_chain_without_example":
        get_triple_for_question_given_subgraph_chain_without_example,
        "singleshot_dialogue_chain": singleshot_dialogue_chain
    }
    return prompt_chains
