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


def create_markdown_table(data):
    keys = list(data.keys())
    header = "|-|"
    separator = "|-|"
    
    # Extracting header from the first dictionary
    for key in data[keys[0]]:
        header += f"{key}|"
        separator += "-|"
    print(header)
    print(separator)
    
    # Printing rows for each experiment
    for key in keys:
        row = f"|{key}"
        for value in data[key].values():
            row += f"|{value}"
        print(row)

def question_type_distribution():
    exp_md_table = {}
    for idx in range(1,4):
        exp_q_type = None
        exp_data_file = f"results/e{idx}.json"
        data = None
        with open(exp_data_file, "r") as f:
            data = json.load(f)
        for d in data:
            qset = d.get("dialogue")
            q_types = count_starting_phrases(qset)
            # print(q_types)
            if exp_q_type is None:
                exp_q_type = q_types.copy()
                continue
            for i in q_types.keys():
                exp_q_type[i] += q_types[i]
            # exp_q_type.update(q_types)
        print(f"exp : {idx}", exp_q_type)
        exp_md_table[f"e{idx}"] = exp_q_type
    create_markdown_table(exp_md_table)


def analyze_cost(data):
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame([d['cost'] for d in data])

    # Calculate average, minimum, and maximum values for each field
    result = {
        "Average": df.mean(),
        "Minimum": df.min(),
        "Maximum": df.max(),
        "Total": df.sum()
    }

    # return pd.DataFrame(result)
    result_df = pd.DataFrame(result)
    return result_df.T.to_markdown()

def question_cost_distribution():
    exp_md_table = {}
    for idx in range(1,4):
        exp_q_type = None
        exp_data_file = f"results/e{idx}.json"
        data = None
        with open(exp_data_file, "r") as f:
            data = json.load(f)
        
        costinfo = analyze_cost(data)
        exp_md_table[f"e{idx}"] = costinfo
        print(f"exp{idx}", costinfo)

    # print(exp_md_table)
    # create_markdown_table(exp_md_table)

# # output
# exp : 1 {'How many': 25, 'What': 45, 'Is': 15, 'When': 17, 'Who': 7, 'other': 16}
# exp : 2 {'How many': 31, 'What': 35, 'Is': 12, 'When': 4, 'Who': 1, 'other': 77}
# exp : 3 {'How many': 22, 'What': 44, 'Is': 5, 'When': 4, 'Who': 17, 'other': 18}

if __name__ == '__main__':
    question_type_distribution()
    question_cost_distribution()