"""
first read the dblp prompts then get inference from the model and write to file
"""
import json
from llama_singleton import ModelSingleton

llama_instance = ModelSingleton()

with open("dblp_test_prompts.json", "r") as f:
    data = json.load(f)

for graph in data:
    prompts = graph["prompts"]
    outputs = []
    for p in prompts:
        output = llama_instance.generate_response(p)
        outputs.append(output)
    graph["outputs"] = outputs

with open("dblp_subgraphs_prompts_output.json", "w") as f:
    json.dump(data, f)
