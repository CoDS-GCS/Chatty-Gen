import os
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def get_unique_types_per_file(json_data):
    unique_types = dict()
    for instance in json_data:
        type = instance["seedType_withPrefix"]
        if type not in unique_types:
            unique_types[type] = 1
        else:
            unique_types[type] += 1
    return unique_types

def get_num_unique_entities(json_data):
    unique_entities = set()
    for instance in json_data:
        seed = instance["seed_withPrefix"]
        unique_entities.add(seed)
    return len(unique_entities)

def get_question_coverage_distribution(json_data, exisiting_dist):
    for instance in json_data:
        question = instance["questionString"]
        word = question.split(" ")[0]
        if word in exisiting_dist:
            exisiting_dist[word] += 1
        else:
            exisiting_dist[word] = 1
    return exisiting_dist

def get_questions(json_data):
    questions = list()
    for instance in json_data:
        question = instance["questionString"]
        questions.append(question)
    return questions

def get_total_type_distribution(current_dist, exisiting_dist):
    for key, value in current_dist.items():
        if key in exisiting_dist:
            exisiting_dist[key] += value
        else:
            exisiting_dist[key] = value
    return exisiting_dist

def plot_dist(dist, output_file):
    plt.clf()
    sorted_data = sorted(dist.items(), key=lambda item: item[1], reverse=True)

    labels, values = zip(*sorted_data)

    # Create a bar chart
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="center")

    for i, v in enumerate(values):
        plt.text(i, v + 0.1, str(v), ha="center", va="bottom")

    # Add a title and labels
    # plt.title("Bar Chart of Dictionary Values")
    plt.xlabel("Question prefix")
    plt.ylabel("Count")
    plt.savefig(output_file, bbox_inches='tight')
    plt.tight_layout()
    # Show the chart
    # plt.show()

def plot_percentage(data, output_file):
    plt.clf()
    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)
    labels, values = zip(*sorted_data)
    total = sum(values)
    new_labels = list()
    new_values = list()
    for label in labels:
        new_labels.append(label.split('/')[-1])
    for value in values:
        new_values.append(round((value/ total)* 100, 1))
    labels = new_labels
    values = new_values

    # plt.bar(labels, values)
    plt.bar(labels, values, width=0.5)  # Adjust width as needed

    plt.xticks(rotation=45, ha="right")

    for i, v in enumerate(values):
        plt.text(i, v + 0.1, str(v), ha="center", va="bottom")

    # Add a title and labels
    # plt.title("Bar Chart of Dictionary Values")
    plt.xlabel("Node Types")
    plt.ylabel("Percentage")
    plt.savefig(output_file, bbox_inches='tight')
    # Show the chart
    plt.tight_layout()
    # plt.show()

def sunburst_plot(all_questions, output_file):
    sentences = list()
    for question in all_questions:
        sentences.append(tuple(question.split()[:2]))

    labels = []
    parents = []
    values = []

    for sentence in sentences:
        # Adjust First Layer
        if sentence[0] in labels:
            index = labels.index(sentence[0])
            if parents[index] == '':
                values[index] += 1
            else:
                labels.append(sentence[0])
                parents.append('')
                values.append(1)
        else:
            labels.append(sentence[0])
            parents.append('')
            values.append(1)

        # Adjust second layer
        if sentence[1] in labels:
            index = labels.index(sentence[1])
            if parents[index] == sentence[0]:
                values[index] += 1
            else:
                labels.append(sentence[1])
                parents.append(sentence[0])
                values.append(1)
        else:
            labels.append(sentence[1])
            parents.append(sentence[0])
            values.append(1)

    sunburst_data = dict(
        type='sunburst',
        labels=labels,
        parents=parents,
        values=values,
    )

    layout = dict(
        margin=dict(l=0, r=0, b=0, t=0),
        sunburstcolorway=["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#19d3f3", "#ffa15a", "#ff6692"],
        font=dict(size=20),

    )

    fig = go.Figure(go.Sunburst(**sunburst_data), layout=layout)
    fig.update_layout(title_text='Multi-level Trigrams Sunburst Chart')
    # fig.show()
    fig.write_image(output_file)


if __name__ == '__main__':
    file_name = "/home/rehamomar/Project/Maestro_Intelligi_Reham/Smart_dblp_1_LIKE_QALD_450_pruned.json"
    coverage_dist = dict()
    type_dist = dict()
    total_num_questions = 0
    all_questions = []
    with open(file_name, 'r') as f:
        data = json.load(f)
        total_num_questions = len(data)
        current_dist = get_unique_types_per_file(data)
        coverage_dist = get_question_coverage_distribution(data, coverage_dist)
        type_dist = get_total_type_distribution(current_dist, type_dist)
        all_questions.extend(get_questions(data))

    print(total_num_questions)
    plot_dist(coverage_dist, "yago_complexity.pdf")
    plot_percentage(type_dist, "yago_diversity.pdf")
    # sunburst_plot(all_questions, "sunburst.png")


