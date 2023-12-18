import utils
import json
import pandas as pd
import random


class Node:
    uri: str
    type: str

    def __init__(self, uri, type):
        self.uri = uri
        self.type = type

    def __str__(self):
        return f"Type:{self.type}, URI:{self.uri}"


# This is an initial implementation, which search for any character is the string
def check_is_human_readable(label):
    for char in label:
        if char.isalpha():
            return True
    return False


# Number of samples should be less than len rare types
def get_samples_for_rare_types(rare_types, num_samples, prefix):
    included_types = random.sample(rare_types, num_samples)
    node_samples = list()
    for type in included_types:
        file_name = utils.get_file_name_from_type(type)
        human_readable_type = f"index_data/{prefix}/{file_name}.txt"
        with open(human_readable_type, "r") as file:
            lines = file.readlines()
        data = [line.strip().split('\t') for line in lines]
        sampled = False
        while not sampled:
            sampled_entities = random.sample(data, k=1)
            if check_is_human_readable(utils.get_name(sampled_entities[0][0])) and int(sampled_entities[0][1]) > 2:
                sampled = True
                node_samples.append(Node(sampled_entities[0][0], type))
    return node_samples


def get_samples_for_type(type, num_samples, prefix):
    file_name = utils.get_file_name_from_type(type)
    human_readable_type = f"index_data/{prefix}/{file_name}.txt"
    with open(human_readable_type, "r") as file:
        lines = file.readlines()

    data = [line.strip().split('\t') for line in lines]
    samples = list()
    sample_count = num_samples
    while len(samples) < num_samples:
        sampled_entities = random.sample(data, k=sample_count)
        for entity in sampled_entities:
            if check_is_human_readable(utils.get_name(entity[0])) and entity[0] not in samples and int(entity[1]) >= 2:
                samples.append(entity[0])
        sample_count = num_samples - len(samples)

    node_samples = list()
    for sample in samples:
        node_samples.append(Node(sample, type))
    return node_samples


def return_seed_nodes(samples_per_type, rare_types, prefix):
    samples = list()
    for key, value in samples_per_type.items():
        if key == "Merged":
            merged_samples = get_samples_for_rare_types(rare_types, value, prefix)
            samples.extend(merged_samples)
        else:
            type_samples = get_samples_for_type(key, value, prefix)
            samples.extend(type_samples)
    return samples

def merge_rare_types(input_df):
    json_object = json.loads(input_df)
    threshold = 1
    new_obj = []
    rare_types = list()
    total_percentage = 0
    total_count = 0
    for inst in json_object:
        if inst['percentage'] > threshold:
            new_obj.append(inst)
        else:
            total_percentage += inst['percentage']
            total_count += inst['Count']
            rare_types.append(inst['Type'])
    new_obj.append({"Type": "Merged", 'Count': total_count, 'percentage': total_percentage})
    return new_obj, rare_types


def get_sample_distribution(input, total_samples):
    samples_per_type = {}
    used_samples = 0
    for inst in input:
        if inst['Type'] == 'Merged':
            continue
        samples = round((inst['percentage'] / 100) * total_samples)
        samples_per_type[inst['Type']] = samples
        used_samples += samples

    samples_per_type['Merged'] = total_samples - used_samples
    return samples_per_type

def calculate_class_importance(input_df):
    # counts = pd.DataFrame(input)
    input_df['Count'] = input_df['Num_Entities'].astype(int)
    total_count = input_df['Num_Entities'].sum()
    percentage_df = input_df.copy()
    percentage_df['percentage'] = (percentage_df['Num_Entities'] / total_count) * 100
    return percentage_df.to_json(orient='records')


def remove_low_richness(file_name):
    df = pd.read_csv(file_name, sep='\t')
    df.columns = df.columns.str.strip()
    filtered_df = df[df.iloc[:, -1] > 2]
    result_df = filtered_df.iloc[:, [0, 1, 3]]
    # result_df_reverse = result_df.iloc[::-1]
    return result_df


def get_seed_nodes(knowledge_graph_prefix, num_samples = 100):
    average_richness_file = f"index_data/{knowledge_graph_prefix}/average_per_type.txt"
    # Removed richness less than 2
    filtered_df = remove_low_richness(average_richness_file)
    # Calculate importance
    percentage_df = calculate_class_importance(filtered_df)
    # Merge types less than one percent
    update_df, rare_types = merge_rare_types(percentage_df)
    # num_samples = 100
    sample_distribution = get_sample_distribution(update_df, num_samples)
    print(sample_distribution)
    seed_nodes = return_seed_nodes(sample_distribution, rare_types, knowledge_graph_prefix)
    return seed_nodes


if __name__ == '__main__':
    knowledge_graph_to_uri = {
        "dbpedia": ("http://206.12.95.86:8890/sparql", "dbpedia"),
        # "lc_quad": "http://206.12.95.86:8891/sparql",
        "microsoft_academic": ("http://206.12.97.159:8890/sparql", "makg"),
        "yago": ("http://206.12.95.86:8892/sparql", "yago"),
        "dblp": ("http://206.12.95.86:8894/sparql", "dblp"),
    }
    kg = "dblp"
    knowledge_graph_uri = knowledge_graph_to_uri[kg][0]
    knowledge_graph_prefix = knowledge_graph_to_uri[kg][1]

    average_richness_file = f"index_data/{knowledge_graph_prefix}/average_per_type.txt"
    # Removed richness less than 2
    filtered_df = remove_low_richness(average_richness_file)
    # Calculate importance
    percentage_df = calculate_class_importance(filtered_df)
    # Merge types less than one percent
    update_df, rare_types = merge_rare_types(percentage_df)
    num_samples = 100
    sample_distribution = get_sample_distribution(update_df, num_samples)
    print(sample_distribution)
    seed_nodes = return_seed_nodes(sample_distribution, rare_types, knowledge_graph_prefix)
    for seed_node in seed_nodes:
        print(str(seed_node))


