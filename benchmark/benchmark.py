# this will be end to end pipeline
# first step is to get the kg's interface instance and parse schema in some format
from llm.prompt_chains import get_prompt_chains
from langchain.callbacks import get_openai_callback

prompt_chains = get_promtp_chains()
question_template_chain = prompt_chains.get('question_template_chain')
pronoun_identification_chain = prompt_chains.get('pronoun_identification_chain')
pronoun_substitution_chain = prompt_chains.get('pronoun_substitution_chain')
n_question_from_subgraph_chain = prompt_chains.get('n_question_from_subgraph_chain')

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

def categorize_questions(original_questions):
    # Dictionary to store categorized questions
    categorized_questions = {}

    # Categorize questions based on question type
    for question in original_questions:
        # Convert "Can you list" question to "List" only (case-insensitive)
        if question.lower().startswith("can you list"):
            question = question.replace("Can you list", "List", 1)
            question = question.replace("?", ".", 1)

        # Extract the question type (e.g., "what," "where," "did," "when," etc.)
        question_type = question.split()[0].lower()

        # Add the question to the corresponding category
        if question_type not in categorized_questions:
            categorized_questions[question_type] = [question]
        else:
            categorized_questions[question_type].append(question)

    return categorized_questions

def filter_and_select_questions(original_questions):
    categorized_questions = categorize_questions(original_questions)

    # List to store selected questions
    selected_questions = []

    # Iterate through question types and select one random question per type
    for question_type, questions in categorized_questions.items():
        selected_question = random.choice(questions)
        selected_questions.append(selected_question)

    return selected_questions

def generate_question_set(kg):
    """
    kg -> seeds
    seeds -> subgraphs
    approach 1. subgraphs -> question set | llm
    approach 2. subgraphs -> question set | question-templates
    """
    seeds = kg.select_seed_nodes(n=10)
    subgraphs = kg.extract_subgraphs(seeds)

    # for approach 2
    triple_list = kg.get_triple_list()
    triple_question_templates_map = {}
    # for triple generate question templates
    for triple in triple_list:
        triple_question_templates_map[triple] = question_template_from_triple(triple)

    for node_type, subgraphs in subgraphs_map.items():
        for subgraph in subgraphs:
            # approach 1 : subgraph to question set
            with get_openai_callback() as cb:
                payload_dict = n_question_from_subgraph_chain.get('payload')
                question_set = n_question_from_subgraph_chain.get('chain').run({**payload_dict, "subgraph": subgraph, "n": n})
                question_set = output["output"]
                question_0 = question_set[0]

                payload_dict = pronoun_identification_chain.get('payload')
                ent_pronoun = pronoun_identification_chain.get('chain').run(
                    {"query": question_0, **payload_dict}
                )
                question_0_ent_pron = ent_pronoun["output"]
                query_dict = {
                    "query_inp": question_set[1:],
                    "query_entity": question_0_ent_pron[0],
                    "query_pronouns": question_0_ent_pron[1],
                }
                payload_dict = pronoun_substitution_chain.get('payload')
                output = pronoun_substitution_chain.run(
                    {**query_dict, **payload_dict}
                )
                transformed_questions = output["output"]
                question_set_dialogue = [question_0, *transformed_questions]
                filtered_set = filter_and_select_questions(question_set_dialogue)
                cb_dict = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    # "successful_requests": cb.successful_requests,
                    # "total_cost": cb.total_cost
                }
                dialogue = {
                    "dialogue": question_set_dialogue,
                    "original": question_set,
                    "filtered": filtered_set,
                    "cost": cb_dict,
                }


        # approach 2: first get set of question from templated generated 
        question_sets = []
        for subgraph in subgraphs:
            question_answer_set = []
            seed_uri, seed_label, seed_type = subgraph.get("seed_node")
            # incoming type
            # in this types of question, subject will be an answer and object will be our seed node
            # so when filling the template, we will fill object and will have unknown for subject
            for triple in subgraph.get("incoming_predicates"):
                predicate, subject_type, subject_label = triple
                query_triple = (subject_type, predicate, seed_type.lower())
                print("query_triple", query_triple)
                # ideally we should use one question per predicate, which will make sure to have small subgraphs
                # for selection we can do randomaly for now as we don't have knowledge of other question type in subgraph
                # or just select all and then implement selction strategy at the end.
                # for now we will get just one random template in list
                ## temp fix: when we get preds with not main classes, ignore it.
                question_templates = triple_question_templates_map.get(query_triple)
                if question_templates is None:
                    continue
                templates = question_templates.get("object")
                for q_t in templates[:1]:  # limiting one question per predicate
                    values = {"object": seed_label}
                    qst = format_template_with_dict(q_t, values)
                    ans = subject_label
                    question_answer_set.append((qst, ans))

            # outgoing type
            # in this types of question, object will be an answer and subject will be our seed node
            # so when filling the template, we will fill subject and will have unknown for object
            for triple in subgraph.get("outgoing_predicates"):
                predicate, object_type, object_label = triple
                query_triple = (seed_type, predicate, object_type.lower())
                # print(query_triple)
                # ideally we should use one question per predicate, which will make sure to have small subgraphs
                # for selection we can do randomaly for now as we don't have knowledge of other question type in subgraph
                # or just select all and then implement selction strategy at the end.
                # for now we will get just one random template in list
                ## temp fix: when we get preds with not main classes, ignore it.
                question_templates = triple_question_templates_map.get(query_triple)
                if question_templates is None:
                    continue
                templates = question_templates.get("subject")
                for q_t in templates[:1]:  # limiting one question per predicate
                    values = {"subject": seed_label}
                    qst = format_template_with_dict(q_t, values)
                    ans = object_label
                    question_answer_set.append((qst, ans))

            # we got set of questions with an aswer label
            print(question_answer_set)
            if len(question_answer_set) > 0:
                question_sets.append(question_answer_set)
