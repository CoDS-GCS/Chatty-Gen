import json
import plotly.graph_objects as go
def generate_sunburst_chart(sentences, file_name):
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
    fig.write_image(file_name)
    # fig.show()

if __name__ == '__main__':
    baseline_file = '../output_26/dblp_e1_20_5.json'
    approach_file = '../output_26/dblp_e11_20_5.json'
    baseline_data = list()
    with open(baseline_file, 'r') as file:
        baseline = json.load(file)
        for inst in baseline['data']:
            if inst['dialogue'] is not None:
                for question in inst['dialogue']:
                    baseline_data.append(tuple(question.split()[:2]))

    generate_sunburst_chart(baseline_data, "baseline_sunburst.png")
    approach_data = list()
    with open(approach_file, 'r') as file:
        approach = json.load(file)
        for inst in approach['data']:
            if inst['dialogue'] is not None:
                for question in inst['dialogue']:
                    approach_data.append(tuple(question.split()[:2]))

    generate_sunburst_chart(approach_data, "approach_sunburst.png")
    qald_file = 'qald-9-test-multilingual.json'
    qald_data = list()
    with open(qald_file) as f:
        qald9_testset = json.load(f)
        for question in qald9_testset['questions']:
            for language_variant_question in question['question']:
                if language_variant_question['language'] == 'en':
                    question_text = language_variant_question['string'].strip()
                    break
            qald_data.append(tuple(question_text.split()[:2]))

    generate_sunburst_chart(qald_data, "qald_sunburst.png")
