import json
import pandas as pd

def count_starting_phrases(question_list):
    starting_phrases = ["How many", "What", "Is", "When", "Who", "other"]
    phrase_count = {phrase: 0 for phrase in starting_phrases}

    total = 0

    for question in question_list:
        for phrase in starting_phrases:
            if question.startswith(phrase):
                phrase_count[phrase] += 1
                total += 1
    
    phrase_count["other"] = len(question_list) - total
            

    return phrase_count


def question_type_distribution(data):
    exp_q_type = None
    for d in data:
        qset = d.get("filtered")
        if qset is None:
            continue
        q_types = count_starting_phrases(qset)
        if exp_q_type is None:
            exp_q_type = q_types.copy()
            continue
        for i in q_types.keys():
            exp_q_type[i] += q_types[i]
    return exp_q_type


def question_cost_distribution(data):
    # Convert the list of dictionaries into a DataFrame
    cost = []
    for d in data:
        if d['cost'] is not None:
            cost.append(d['cost'])

    df = pd.DataFrame(cost)

    # Calculate average, minimum, and maximum values for each field
    result = {
        "Average": df.mean(),
        "Minimum": df.min(),
        "Maximum": df.max(),
        "Total": df.sum()
    }

    # return pd.DataFrame(result)
    result_df = pd.DataFrame(result)
    return result_df.to_dict()


def analyze_benchmark_sample(benchmark_sample):
    question_type_dist = question_type_distribution(benchmark_sample)
    question_cost_dist = question_cost_distribution(benchmark_sample)
    
    analysis = {
        "types": question_type_dist,
        "cost": question_cost_dist
    }
    return analysis