from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
)

llm = OpenAI(model_name="text-davinci-003", temperature=0.5, streaming=False)


def get_question_template_chain():
    response_schemas = [
        ResponseSchema(
            name="output",
            description="a list of question templates",
            type="List[string]",
        )
    ]

    json_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    json_format_instructions = json_output_parser.get_format_instructions()

    QUESTION_TEMPLATE_PROMPT = PromptTemplate(
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

    example_input = ("person", "worksAt", "company")
    example_output = [
        "Which company employs {subject}?",
        "At which company does {subject} work?",
        "What is the workplace of {subject}?",
        "Who works at {object}?",
        "Which person is employed at {object}?",
        "At {object}, who is an employee?",
    ]

    question_template_chain = LLMChain(
        llm=llm,
        prompt=QUESTION_TEMPLATE_PROMPT,
        verbose=True,
        output_parser=json_output_parser,
    )
    payload = {"example_input": example_input, "example_output": example_output}
    return {
        "chain": question_template_chain,
        "payload": payload
    }


def get_pronoun_identification_chain():
    p_idf_response_schemas = [
        ResponseSchema(
            name="output",
            description="a tuple of Entity and its pronouns",
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
        llm=llm,
        prompt=P_IDF_PROMPT,
        verbose=True,
        output_parser=p_idf_json_output_parser,
    )
    # TODO: fix randomness as it will be always same while generation
    pronoun_examples = select_random_pronoun_examples()
    examples_dict = {f"e_{idx+1}_inp": x[0] for idx, x in enumerate(pronoun_examples)}
    examples_dict.update(
        {f"e_{idx+1}_out": list(x[1]) for idx, x in enumerate(pronoun_examples)}
    )
    return {
        "chain": pronoun_identification_chain,
        "payload": {"payload": example_dict},
    }

def get_pronoun_substitution_chain():
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
    examples_dict = {
        "example_inp": example_inp,
        "example_out": example_out,
        "example_entity": example_entity,
        "example_pronouns": example_pronouns,
    }
    return {
        "chain": pronoun_substitution_chain,
        "payload": {"payload": example_dict},
    }


def get_n_question_from_subgraph_chain():
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
        llm=llm, prompt=N_Q_PROMPT, verbose=True, output_parser=n_q_json_output_parser
    )
    payload = {
        "example_subgraph": example_subgraph,
        "example_n": example_n,
        "example_output": example_output,
    }
    return {
        'chain': n_question_generator_chain,
        'payload': payload
    }

def get_prompt_chains():
    prompt_chains = {
        "question_template_chain": get_question_template_chain(),
        "pronoun_identification_chain": get_pronoun_identification_chain(),
        "pronoun_substitution_chain": get_pronoun_substitution_chain(),
        "n_question_from_subgraph_chain": get_n_question_from_subgraph_chain()
    }
    return prompt_chains