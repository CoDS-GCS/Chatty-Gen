"""
dblp_test.py

generate test data for LLM finetuned on dblp subgraphs - to - text data.
uses the gpg-3.5 to generate 5 template prompts from the templates, and actual subgraph's text data for dblp.
"""
import random
import json
from pprint import pprint


def generate_test_prompts(subgraph):
    prompts = []
    for triple in subgraph["triples"]:
        if triple["predicate"] == "primary affiliation":
            author_name = triple["subject"]
            prompts.append(
                f"Instruction: Compose a 100-word summary about {author_name}. Output: "
            )
            prompts.append(
                f"Instruction: Identify the affiliation of {author_name}. Output: "
            )

        elif triple["predicate"] == "authored by":
            paper_title = triple["subject"]
            author_name = triple["object"]
            prompts.append(
                f'Instruction: Who authored the paper titled "{paper_title}"? Output: '
            )
            prompts.append(
                f'Instruction: Which author is affiliated with the organization in the paper "{paper_title}"? Output: '
            )
            prompts.append(
                f"Instruction: List the titles of the top 3 papers, separated by commas, that were authored by {author_name}. Output: "
            )
    return {"author_name": author_name, "prompts": prompts}


def generate_dblp_test_prompts():
    # first read the file
    with open("dblp_subgraphs.jsonl", "r") as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    print(len(data))
    sample_size = int(len(data) * 0.02)
    print(sample_size)
    sampled = random.sample(data, sample_size)
    generated_data = []
    for idx, subgraph in enumerate(sampled):
        d = generate_test_prompts(subgraph)
        generated_data.append(d)
    with open("dblp_test_prompts.json", "w") as f:
        json.dump(generated_data, f, indent=4)


if __name__ == "__main__":
    generate_dblp_test_prompts()
