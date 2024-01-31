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


def get_types_for_kg(kg_name):
    knowledge_graph_prefix = utils.knowledge_graph_to_uri[kg_name][1]
    knowledge_graph_uri = utils.knowledge_graph_to_uri[kg_name][0]
    average_richness_file = f"../index_data/{knowledge_graph_prefix}/average_per_type.txt"
    filtered_df = sampling.remove_low_richness(average_richness_file)
    percentage_df = sampling.calculate_class_importance(filtered_df)
    update_df = sampling.remove_rare_types(percentage_df)
    cleaned_df = sampling.eliminate_dominated_parents(update_df, knowledge_graph_uri)
    json_object = json.loads(cleaned_df)
    return pd.DataFrame(json_object)


def plot_comparison(benchmark_data, kg_data, kg_name):
    benchmark_data = OrderedDict(sorted(benchmark_data.items(), key=lambda item: item[1], reverse=True))
    kg_data = OrderedDict(sorted(kg_data.items(), key=lambda item: item[1], reverse=True))

    labels = list(set(benchmark_data.keys()) | set(kg_data.keys()))

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
    bar_width = 0.35

    # Create the grouped bar chart
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - bar_width / 2, values_benchmark, bar_width, label='Benchmark')
    bar2 = ax.bar(x + bar_width / 2, values_kg, bar_width, label='KG')

    for i, value in enumerate(values_benchmark):
        ax.text(i - bar_width / 2, value + 0.1, str(round(value, 1)), ha='center', va='bottom')

    for i, value in enumerate(values_kg):
        ax.text(i + bar_width / 2, value + 0.1, str(round(value, 1)), ha='center', va='bottom')

    # Set labels and title
    ax.set_xlabel(f"Node Types for {kg_name.upper()}")
    ax.set_ylabel('Percentage')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.legend()

    output_file = f"{kg_name}_diversity.pdf"
    plt.savefig(output_file, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    file = '../output_26/dblp_e11_20_5.json'
    kg_name = file.split('/')[-1].split('_')[0]
    file = open(file, 'r')
    file_data = json.load(file)
    seed_nodes = list()
    for inst in file_data['data']:
        seed_nodes.append(inst['seed_entity'])

    kg_type_distribution = get_types_for_kg(kg_name)
    print(" Knowledge graph type distribution:")
    print(kg_type_distribution)
    json_data = kg_type_distribution.to_json(orient='records')
    json_object = json.loads(json_data)
    kg_data = dict()
    for obj in json_object:
        kg_data[utils.get_name(obj['Type']).strip()] = obj['percentage']

    non_rare_types = list(kg_type_distribution['Type'].values)
    non_rare_types = [x.strip() for x in non_rare_types]
    benchmark_type_distribution = get_types_for_nodes(seed_nodes, kg_name, non_rare_types)
    print("Benchmark type distribution:")
    print(benchmark_type_distribution)
    json_data = benchmark_type_distribution.to_json(orient='records')
    json_object = json.loads(json_data)
    benchmark_data = dict()
    for obj in json_object:
        benchmark_data[utils.get_name(obj['Type']).strip()] = obj['percentage']
    plot_comparison(benchmark_data, kg_data, kg_name)



