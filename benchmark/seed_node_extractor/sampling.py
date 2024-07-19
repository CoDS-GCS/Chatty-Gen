import sys
# sys.path.append('../benchmark/')
import seed_node_extractor.utils as utils
import json
import pandas as pd
import random
from kg.kg.kg import Node
from rdflib import URIRef


# This is an initial implementation, which search for any character is the string
def check_is_human_readable(label):
    for char in label:
        if char.isalpha():
            return True
    return False


# Number of samples should be less than len rare types
# For random sampling
# def get_samples_for_rare_types(rare_types, num_samples, prefix):
#     included_types = random.sample(rare_types, num_samples)
#     node_samples = list()
#     for type in included_types:
#         file_name = utils.get_file_name_from_type(type)
#         human_readable_type = f"index_data/{prefix}/{file_name}.txt"
#         with open(human_readable_type, "r") as file:
#             lines = file.readlines()
#         data = [line.strip().split('\t') for line in lines]
#         sampled = False
#         while not sampled:
#             sampled_entities = random.sample(data, k=1)
#             if check_is_human_readable(utils.get_name(sampled_entities[0][0])) and int(sampled_entities[0][1]) > 2:
#                 sampled = True
#                 node_samples.append(Node(uri=URIRef(sampled_entities[0][0]), nodetype=URIRef(type.strip())))
#     return node_samples

# For Random Sampling
# def get_samples_for_type(type, num_samples, prefix):
#     file_name = utils.get_file_name_from_type(type)
#     human_readable_type = f"index_data/{prefix}/{file_name}.txt"
#     with open(human_readable_type, "r") as file:
#         lines = file.readlines()
#
#     data = [line.strip().split('\t') for line in lines]
#     samples = list()
#     sample_count = num_samples
#     while len(samples) < num_samples:
#         sampled_entities = random.sample(data, k=sample_count)
#         for entity in sampled_entities:
#             if check_is_human_readable(utils.get_name(entity[0])) and entity[0] not in samples and int(entity[1]) >= 2:
#                 samples.append(entity[0])
#         sample_count = num_samples - len(samples)
#
#     node_samples = list()
#     for sample in samples:
#         node_samples.append(Node(uri=URIRef(sample), nodetype=URIRef(type.strip())))
#     return node_samples

def get_samples_for_rare_types(rare_types, num_samples, prefix, str_samples_so_far):
    included_types = random.sample(rare_types, num_samples)
    node_samples = list()
    for type in included_types:
        file_name = utils.get_file_name_from_type(type)
        human_readable_type = f"index_data/{prefix}/{file_name}.txt"
        with open(human_readable_type, "r") as file:
            lines = file.readlines()
        data = [line.strip().split('\t') for line in lines]
        data = [(type_, int(count)) for type_, count in data]
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
        sorted_count = 0
        while sorted_data[sorted_count][0] in str_samples_so_far:
            sorted_count += 1
        node_samples.append(Node(uri=URIRef(sorted_data[0][0]), nodetype=URIRef(type.strip())))
    return node_samples


#  For top k
def get_samples_for_type(type, num_samples, prefix, str_samples_so_far):
    file_name = utils.get_file_name_from_type(type)
    human_readable_type = f"index_data/{prefix}/{file_name}.txt"
    with open(human_readable_type, "r") as file:
        lines = file.readlines()

    data = [line.strip().split('\t') for line in lines]
    data = [(type_, int(count)) for type_, count in data]
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    node_samples = list()
    str_samples = list()
    sorted_count = 0
    for i in range(num_samples):
        while sorted_data[sorted_count][0] in str_samples_so_far:
            sorted_count += 1
        node_samples.append(Node(uri=URIRef(sorted_data[sorted_count][0]), nodetype=URIRef(type.strip())))
        str_samples.append(sorted_data[sorted_count][0])
        sorted_count += 1
    return node_samples, str_samples


def return_seed_nodes(samples_per_type, prefix):
    node_samples = list()
    str_samples = list()
    for key, value in samples_per_type.items():
        if value > 0:
            # if key == "Merged":
            #     merged_samples = get_samples_for_rare_types(rare_types, value, prefix, str_samples)
            #     node_samples.extend(merged_samples)
            #     str_samples.extend(str_type_samples)
            # else:
            node_type_samples, str_type_samples = get_samples_for_type(key, value, prefix, str_samples)
            node_samples.extend(node_type_samples)
            str_samples.extend(str_type_samples)
    return node_samples

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

def remove_rare_types(input_df):
    threshold = 1
    filtered_df = input_df[input_df['percentage'] > threshold]
    filtered_df = filtered_df.drop(columns=['percentage'])
    total_count = filtered_df['Num_Entities'].sum()
    percentage_df = filtered_df.copy()
    percentage_df['percentage'] = (percentage_df['Num_Entities'] / total_count) * 100
    # return percentage_df.to_json(orient='records')
    return percentage_df

def get_sample_distribution(input, total_samples):
    samples_per_type = {}
    used_samples = 0
    for inst in input:
        # if inst['Type'] == 'Merged':
        #     continue
        samples = round((inst['percentage'] / 100) * total_samples)
        samples_per_type[inst['Type']] = samples
        used_samples += samples

    remaining_samples = total_samples - used_samples
    for node_type in samples_per_type:
        if remaining_samples > 0 and samples_per_type[node_type] == 0:
            samples_per_type[node_type] += 1
            remaining_samples -= 1

    index = 0
    while remaining_samples > 0:
        node_type = list(samples_per_type.keys())[index]
        samples_per_type[node_type] += 1
        remaining_samples -= 1
        index = (index + 1) % len(samples_per_type)

    return samples_per_type

def calculate_class_importance(input_df):
    input_df['Count'] = input_df['Num_Entities'].astype(int)
    total_count = input_df['Num_Entities'].sum()
    percentage_df = input_df.copy()
    percentage_df['percentage'] = (percentage_df['Num_Entities'] / total_count) * 100
    return percentage_df
    # return percentage_df.to_json(orient='records')


def remove_low_richness(file_name):
    df = pd.read_csv(file_name, sep='\t')
    df.columns = df.columns.str.strip()
    filtered_df = df[df.iloc[:, -1] > 2]
    result_df = filtered_df.iloc[:, [0, 1, 3]]
    # result_df_reverse = result_df.iloc[::-1]
    return result_df


def get_seed_nodes(kg_name, num_samples = 100):
    knowledge_graph_uri = utils.knowledge_graph_to_uri[kg_name][0]
    knowledge_graph_prefix = utils.knowledge_graph_to_uri[kg_name][1]
    average_richness_file = f"index_data/{knowledge_graph_prefix}/average_per_type.txt"
    # Removed richness less than 2
    filtered_df = remove_low_richness(average_richness_file)
    # Calculate importance
    percentage_df = calculate_class_importance(filtered_df)
    # Merge types less than one percent
    update_df = remove_rare_types(percentage_df)
    # update_df, rare_types = merge_rare_types(percentage_df)
    cleaned_df = eliminate_dominated_parents(update_df, knowledge_graph_uri)
    # num_samples = 100
    json_object = json.loads(cleaned_df)
    sample_distribution = get_sample_distribution(json_object, num_samples)
    print(sample_distribution)
    seed_nodes = return_seed_nodes(sample_distribution, knowledge_graph_prefix)
    return seed_nodes, sample_distribution


def get_parents(children_names, knowledge_graph_uri):
    parents = list()
    for child in children_names:
        query = ("select ?parent where {"
                 f"<{child.strip()}> "
                 "<http://www.w3.org/2000/01/rdf-schema#subClassOf> ?parent}")
        results = utils.send_sparql_query(knowledge_graph_uri, query)
        parent_exist = False
        for binding in results['results']['bindings']:
            parent = binding.get('parent', {}).get('value', None)
            if parent in children_names:
                parents.append(parent)
                parent_exist = True
        if not parent_exist:
            parents.append(None)
    return parents



def eliminate_dominated_parents(df, knowledge_graph_uri):
    df['Type'] = df['Type'].str.strip()
    children_names = [x.strip() for x in df['Type'].values]
    parents = get_parents(children_names, knowledge_graph_uri)
    types_to_remove = list()
    for child, parent in zip(children_names, parents):
        if parent is not None:
            count_child = df[df['Type'] == child]['Num_Entities'].values[0]
            count_parent = df[df['Type'] == parent]['Num_Entities'].values[0]
            if count_child / (count_parent * 1.0) > 0.99:
                types_to_remove.append(parent)
    df_cleaned = df[~df['Type'].isin(types_to_remove)]
    print(df_cleaned)
    return df_cleaned.to_json(orient='records')




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
    # # Merge types less than one percent
    # # update_df, rare_types = merge_rare_types(percentage_df)
    print(percentage_df)
    update_df = remove_rare_types(percentage_df)
    print(update_df)
    # Remove Dominated Parents i.e. once child forms more than 99% of the parent
    cleaned_df = eliminate_dominated_parents(update_df, knowledge_graph_uri)
    num_samples = 100
    json_object = json.loads(cleaned_df)
    sample_distribution = get_sample_distribution(json_object, num_samples)
    print(sample_distribution)
    seed_nodes = return_seed_nodes(sample_distribution, knowledge_graph_prefix)
    for seed_node in seed_nodes:
        print(str(seed_node))


