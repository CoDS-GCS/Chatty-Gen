import json
import pandas as pd
import matplotlib.pyplot as plt
from seed_node_extractor import utils, sampling
import numpy as np
from collections import OrderedDict


def get_types_for_nodes(seed_nodes, kg_name, non_rare_types):
    types_distribution = dict()
    endpoint = utils.knowledge_graph_to_uri[kg_name][0]
    for node in seed_nodes:
        query = ("Select ?type where {"
                 f"<{node}> rdf:type ?type"
                 "}")
        result = utils.send_sparql_query(endpoint, query)
        for binding in result["results"]["bindings"]:
            type = binding.get('type', {}).get('value', None)
            if type not in non_rare_types:
                continue
            if type in types_distribution:
                types_distribution[type] += 1
            else:
                types_distribution[type] = 1

    json_for_df = []
    for key, value in types_distribution.items():
        json_for_df.append({"Type": key, "Count": value})
    df = pd.DataFrame(json_for_df)
    total_count = df['Count'].sum()
    percentage_df = df.copy()
    percentage_df['percentage'] = (percentage_df['Count'] / total_count) * 100
    return percentage_df

def calculate_class_importance(input_df):
    input_df['Count'] = input_df['Count'].astype(int)
    total_count = input_df['Count'].sum()
    percentage_df = input_df.copy()
    percentage_df['percentage'] = (percentage_df['Count'] / total_count) * 100
    return percentage_df

def eliminate_dominated_parents(df, knowledge_graph_uri):
    df['Type'] = df['Type'].str.strip()
    children_names = [x.strip() for x in df['Type'].values]
    parents = sampling.get_parents(children_names, knowledge_graph_uri)
    types_to_remove = list()
    for child, parent in zip(children_names, parents):
        if parent is not None:
            count_child = df[df['Type'] == child]['Count'].values[0]
            count_parent = df[df['Type'] == parent]['Count'].values[0]
            if count_child / (count_parent * 1.0) > 0.99:
                types_to_remove.append(parent)
    df_cleaned = df[~df['Type'].isin(types_to_remove)]
    print(df_cleaned)
    return df_cleaned.to_json(orient='records')
    # return df_cleaned

def remove_rare_types(input_df):
    threshold = 1
    filtered_df = input_df[input_df['percentage'] > threshold]
    filtered_df = filtered_df.drop(columns=['percentage'])
    total_count = filtered_df['Count'].sum()
    percentage_df = filtered_df.copy()
    percentage_df['percentage'] = (percentage_df['Count'] / total_count) * 100
    # return percentage_df.to_json(orient='records')
    return percentage_df
def get_types_for_kg(kg_name):
    knowledge_graph_prefix = utils.knowledge_graph_to_uri[kg_name][1]
    knowledge_graph_uri = utils.knowledge_graph_to_uri[kg_name][0]
    kg_type_distribution = utils.get_type_distrubution(knowledge_graph_uri, knowledge_graph_prefix)
    distribution = pd.DataFrame(kg_type_distribution)
    percentage_df = calculate_class_importance(distribution)
    update_df = remove_rare_types(percentage_df)
    cleaned_df = eliminate_dominated_parents(update_df, knowledge_graph_uri)
    json_object = json.loads(cleaned_df)
    return pd.DataFrame(json_object)

def plot_comparison(benchmark_data, kg_data, kg_name):

    benchmark_data = OrderedDict(sorted(benchmark_data.items(), key=lambda item: item[1], reverse=True))
    kg_data = OrderedDict(sorted(kg_data.items(), key=lambda item: item[1], reverse=True))


    labels = list()

    for item in kg_data.keys():
        if item not in labels:
            labels.append(item)

    for item in benchmark_data.keys():
        if item not in labels:
            labels.append(item)


    # Create an array of indices for the labels
    x = np.arange(len(labels))

    # Get the values for each label from both approaches, with 0 if the label is not present in an approach
    values_benchmark = [benchmark_data.get(label, 0) for label in labels]
    values_kg = [kg_data.get(label, 0) for label in labels]

    # Define bar width
    bar_width = 0.4
    font_size = 13
    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 7))
    bar1 = ax.bar(x - bar_width / 2, values_benchmark, bar_width, label='CoKG', color='tab:green')
    bar2 = ax.bar(x + bar_width / 2, values_kg, bar_width, label='KG',  color='steelblue')

    for i, value in enumerate(values_benchmark):
        ax.text(i - bar_width / 2, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom', fontsize=10)

    for i, value in enumerate(values_kg):
        ax.text(i + bar_width / 2, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom', fontsize=10)

    # Set labels and title
    ax.set_xlabel(f"Node Types for {kg_name.upper()}", labelpad=10., fontsize=font_size)
    ax.set_ylabel('Percentage', fontsize=font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=font_size)
    # ax.legend()
    ax.legend(loc='upper right', fontsize=13)

    fig.tight_layout()
    output_file = f"Figures/{kg_name}_diversity.pdf"
    plt.subplots_adjust(left=0.2, bottom=0.2, right=1.35)
    plt.savefig(output_file, bbox_inches='tight')



if __name__ == '__main__':
    file = '../output_26/dblp_e11_20_5.json'
    kg_name = file.split('/')[-1].split('_')[0]
    file = open(file, 'r')
    file_data = json.load(file)
    seed_nodes = list()
    for inst in file_data['data']:
        seed_nodes.append(inst['seed_entity'])

    kg_type_distribution = get_types_for_kg(kg_name)
    # print(" Knowledge graph type distribution:")
    # print(kg_type_distribution)
    json_data = kg_type_distribution.to_json(orient='records')
    json_object = json.loads(json_data)
    kg_data = dict()
    for obj in json_object:
        kg_data[utils.get_name(obj['Type']).strip()] = obj['percentage']

    non_rare_types = list(kg_type_distribution['Type'].values)
    non_rare_types = [x.strip() for x in non_rare_types]
    benchmark_type_distribution = get_types_for_nodes(seed_nodes, kg_name, non_rare_types)
    # print("Benchmark type distribution:")
    # print(benchmark_type_distribution)
    json_data = benchmark_type_distribution.to_json(orient='records')
    json_object = json.loads(json_data)
    benchmark_data = dict()
    for obj in json_object:
        benchmark_data[utils.get_name(obj['Type']).strip()] = obj['percentage']
    plot_comparison(benchmark_data, kg_data, kg_name)



