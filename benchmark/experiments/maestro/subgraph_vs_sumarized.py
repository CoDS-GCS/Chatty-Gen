import matplotlib.pyplot as plt
import numpy as np

# Sample data for the grouped bar charts
labels_cost = ['DBpedia', 'YAGO', 'DBLP', 'MAG']
labels_percentage = ['DBpedia', 'YAGO', 'DBLP', 'MAG']

# Values from analysis[cost][Total]
data1 = {
    'Summarized': [50, 46, 55, 49],
    'FULL': [69, 63, 62, 89],
}

#Values from analysis[execution][Correct] / Sum(analysis[execution])
data2 = {
    'Summarized': [94.9, 94.5, 93.8, 90.9],
    'FULL': [81.4, 89.0, 82.9, 86.6],

}

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

def plot_cost_bar_chart(ax, labels, data, title):
    # Create an array of indices for the labels
    x = np.arange(len(labels))
    bar_width = 0.3
    font_size = 15
    values_font_size = 10

    bars1 = ax.bar(x - bar_width / 2, data[list(data.keys())[0]], bar_width, label=list(data.keys())[0], color='tab:green')
    bars2 = ax.bar(x + bar_width / 2, data[list(data.keys())[1]], bar_width, label=list(data.keys())[1], color='steelblue')

    for i, value in enumerate(data[list(data.keys())[0]]):
        ax.text(i - bar_width / 2, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=values_font_size)

    for i, value in enumerate(data[list(data.keys())[1]]):
        ax.text(i + bar_width /2, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=values_font_size)

    ax.set_xlabel(title,  labelpad=10., fontsize=font_size)
    ax.set_ylabel('Number of tokens(K)', fontsize=font_size)
    ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=font_size)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.legend(loc='best', fontsize=values_font_size)
    fig.tight_layout()

def plot_sparql_bar_chart(ax, labels, data, title):
    # Create an array of indices for the labels
    x = np.arange(len(labels))
    bar_width = 0.3
    font_size = 15
    values_font_size = 10

    bars1 = ax.bar(x - bar_width / 2, data[list(data.keys())[0]], bar_width, label=list(data.keys())[0], color='tab:green')
    bars2 = ax.bar(x + bar_width / 2, data[list(data.keys())[1]], bar_width, label=list(data.keys())[1], color='steelblue')

    for i, value in enumerate(data[list(data.keys())[0]]):
        ax.text(i - bar_width / 2, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=values_font_size)

    for i, value in enumerate(data[list(data.keys())[1]]):
        ax.text(i + bar_width /2, value + 0.1, str(int(round(value, 0))), ha='center', va='bottom',  fontsize=values_font_size)

    ax.set_xlabel(title,  labelpad=10., fontsize=font_size)
    ax.set_ylabel('Correct Queries (%)', fontsize=font_size)
    ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=font_size)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.legend(loc='best', fontsize=values_font_size)
    fig.tight_layout()
    return bars1, bars2

# Function to plot grouped bar chart
def plot_grouped_bar_chart(ax, labels, data, title):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    bars1 = ax.bar(x - width/2, data[list(data.keys())[0]], width, label=list(data.keys())[0])
    bars2 = ax.bar(x + width/2, data[list(data.keys())[1]], width, label=list(data.keys())[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_xlabel('Groups')
    # ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # output_file = f"../Final_Benchmarks/subgraph_vs_summarized.pdf"
    # plt.savefig(output_file, bbox_inches='tight')

# Plot the first grouped bar chart in the first subplot
# plot_grouped_bar_chart(ax1, labels_cost, data1, '(A)')
plot_cost_bar_chart(ax1, labels_cost, data1, '(A)')
ax1.set_ylim(30, 95)
# Plot the second grouped bar chart in the second subplot
# plot_grouped_bar_chart(ax2, labels_percentage, data2, '(B)')
bars1, bars2 = plot_sparql_bar_chart(ax2, labels_percentage, data2, '(B)')
ax2.set_ylim(30, 105)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
# plt.show()
fig.legend([bars1, bars2], [list(data1.keys())[0], list(data1.keys())[1]], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=13, )
#
output_file = f"../Final_Benchmarks/subgraph_vs_summarized_v2.pdf"
plt.savefig(output_file, bbox_inches='tight')

