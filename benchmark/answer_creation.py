from llm.prompt_chains import get_prompt_chains
from kg.kg.kg import defrag_uri

prompt_chains = get_prompt_chains()
get_answer_chain = prompt_chains.get("get_answer_from_question_and_triple_zero_shot")
get_target_chain = prompt_chains.get("get_target_answer_from_triples")


# Start Utils Function

def get_triple_for_summarized(triple, subgraph):
    # triple = triple.replace('"', '').replace("'", "")
    for el in subgraph.triples:
        # if str(subgraph.get_triple_representation_for_optimized(el)) == triple:
        # seralized_triple = str(subgraph.get_triple_representation_no_object(el)).replace('"', '').replace("'", "")
        seralized_triple = subgraph.get_triple_representation_no_object(el)
        if seralized_triple[0] == triple[0] and seralized_triple[1] == triple[1]:
            return subgraph.get_triple_with_uris_no_object(el)
    return None


def get_original_triple(triple, sub_graph):
    triple = triple.replace('"', '').replace("'", "")
    for el in sub_graph.triples:
        seralized_triple = str(sub_graph.get_triple_representation(el, 'uri')).replace('"', '').replace("'", "")
        if seralized_triple == triple:
            return el
    return None


def is_boolean(question):
    return (question.lower().startswith('is') or question.lower().startswith('are') or
            question.lower().startswith('was') or question.lower().startswith('were') or
            question.lower().startswith('did') or question.lower().startswith('do') or
            question.lower().startswith('does'))


def detect_is_count_query(question, triples):
    if not question.lower().startswith('how many'):
        return False
    if len(triples) > 1:
        return True
    elif len(triples) == 1:
        triple_components = triples[0][1:len(triples[0]) - 1].strip().split(',')
        subj, pred, obj = triple_components[0], triple_components[1], triple_components[2]
        return not obj.strip().replace("'", "").isnumeric()
    return True


def get_ask_query_from_triples(triples, subgraph):
    query = 'Ask where {'
    for triple in triples:
        original_triple = get_original_triple(triple, subgraph)
        for el in original_triple:
            if el.uri:
                query += f"<{el.__str__()}> "
            else:
                value = el.__str__().strip()
                query += f"\"{value}\" "
        query += '.'
    query += '}'
    return query


def detect_unknown_in_triple(question, triple):
    triple_splitted = triple[1:len(triple) - 1].strip().split(',')
    if len(triple_splitted) == 3:
        subj, pred, obj = triple_splitted
    elif len(triple_splitted) == 2:
        subj, pred = triple_splitted
        obj = ""
    else:
        subj = triple_splitted[0]
        pred, obj = "", ""

    subj = subj.strip().replace("'", "")
    obj = obj.strip().replace("'", "")
    if obj in question:
        return subj
    else:
        return obj


def get_unknown_from_llm(question, triples, subgraph):
    triples_list = list()
    for triple in triples:
        original_triple = get_original_triple(triple, subgraph)
        subject, predicate, object = original_triple
        triples_list.append((subject.__str__(), predicate.__str__(), object.__str__()))

    output = get_target_chain.get("chain").run({"question": question, "triples": triples_list})
    return output['target']


def get_select_query_with_target(triples, subgraph, target):
    query = ''
    used_predicates = set()
    for triple in triples:
        original_triple = get_original_triple(triple, subgraph)
        if original_triple[1].__str__() not in used_predicates:
            used_predicates.add(original_triple[1].__str__())
            if original_triple[0].__str__() == target:
                query += f"?uri <{original_triple[1].__str__()}> <{original_triple[2].__str__()}> ."
            elif original_triple[2].__str__() == target:
                query += f"<{original_triple[0].__str__()}> <{original_triple[1].__str__()}> ?uri."
            else:
                for el in original_triple:
                    if el.uri:
                        query += f"<{el.__str__()}> "
                    else:
                        value = el.__str__().strip()
                        query += f"\"{value}\" "
    query += '}'
    return query
# End utils Function

# Start Different Answer creation Approaches
# Inputs to LLM are question and triples and the output is the SPARQL query
def get_answer_LLM_based(question, triples, subgraph, approach):
    triples_list = list()
    for triple in triples:
        if approach == "optimized":
            returned_triple = get_triple_for_summarized(triple, subgraph)
        else:
            returned_triple = get_original_triple(triple, subgraph)
        subject, predicate, object = returned_triple
        triples_list.append((subject.__str__(), predicate.__str__(), object.__str__()))

    ch = get_answer_chain.get("chain")
    post_processor = get_answer_chain.get("post_processor")
    llm_result = ch.generate([{"question": question, "triples": triples_list}], None)
    output = post_processor(llm_result)
    return output['sparql']


# Fully Rule Based Approach
def get_answer_query_from_graph(triples, seed_entity, subgraph, question):
    is_boolean_question = is_boolean(question)
    is_count_query = detect_is_count_query(question, triples)
    if is_boolean_question:
        query = get_ask_query_from_triples(triples, subgraph)
    else:
        used_predicates = set()
        if is_count_query:
            query = "select count(?uri) as ?count where {"
        else:
            query = "select ?uri where { "
        for triple in triples:
            original_triple = get_original_triple(triple, subgraph)
            if original_triple[1].__str__() not in used_predicates:
                used_predicates.add(original_triple[1].__str__())
                target = detect_unknown_in_triple(question, triple)
                if original_triple[0].__str__() == target or defrag_uri(original_triple[0].__str__()) == target:
                    query += f"?uri <{original_triple[1].__str__()}> <{original_triple[2].__str__()}> ."
                elif original_triple[2].__str__() == target or defrag_uri(original_triple[2].__str__()) == target:
                    query += f"<{original_triple[0].__str__()}> <{original_triple[1].__str__()}> ?uri."
                else:
                    for el in original_triple:
                        if el.uri:
                            query += f"<{el.__str__()}> "
                        else:
                            value = el.__str__().strip()
                            query += f"\"{value}\" "
        query += '}'
    return query


# LLM Based Approach followed by post processing for count and boolean Errors
def get_LLM_based_postprocessed(question, triples, subgraph, approach):
    llm_query = get_answer_LLM_based(question, triples, subgraph, approach)
    if is_boolean(question) and not llm_query.lower().startswith("ask"):
        where_index = llm_query.lower().index("where")
        llm_query = 'ASK ' + llm_query[where_index:]
    else:
        if approach == "optimized":
            return llm_query

        is_count_query = detect_is_count_query(question, triples)
        if is_count_query and 'count(' not in llm_query.lower():
            variable_name = llm_query.split(' ')[1]
            where_index = llm_query.lower().index("where")
            llm_query = 'SELECT (COUNT(' + variable_name + ') as ?count) ' + llm_query[where_index:]
        elif not is_count_query and 'count(' in llm_query.lower():
            start = llm_query.lower().index('count(')
            end = llm_query.index(')')
            where_index = llm_query.lower().index("where")
            variable_name = llm_query[start + 6: end]
            llm_query = 'SELECT ' + variable_name + ' ' + llm_query[where_index:]
    return llm_query


# Rule Based Approach with getting the unknown using LLMs
def updated_get_answer_query_from_graph(triples, subgraph, question):
    is_boolean_question = is_boolean(question)
    is_count_query = detect_is_count_query(question, triples)
    if is_boolean_question:
        query = get_ask_query_from_triples(triples, subgraph)
    else:
        target = get_unknown_from_llm(question, triples, subgraph)
        if is_count_query:
            query = "select count(?uri) as ?count where {"
        else:
            query = "select ?uri where { "
        select_query = get_select_query_with_target(triples, subgraph, target)
        query += select_query
    return query
# End Answer creation Approaches
