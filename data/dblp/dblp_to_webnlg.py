"""
dblp_to_webnlg.py

from dblp subgraphs to webnlg format required for jointGT model.
"""
import jsonlines
import os
import json
from pprint import pprint

base_dir = os.path.dirname(os.path.abspath(__file__))

filepath = os.path.join(base_dir, "../subgraphs.jsonl")


def process_jsonl_file(file_path):
    output_data = []
    with jsonlines.open(file_path) as reader:
        for idx, line in enumerate(reader, start=1):
            triplets = line.get("triples", [])
            kbs = {}
            kb_id = 1
            kb = []
            for triplet in triplets:
                subject = triplet.get("subject", "")
                predicate = triplet.get("predicate", "")
                obj = triplet.get("object", "")
                kb.append([predicate, obj])
                kbs[str(kb_id)] = [subject, subject, kb]
                kb_id += 1
                kb = []
            if kb:
                kbs[str(kb_id)] = [subject, subject, kb]
            output_data.append({"id": idx, "kbs": kbs, "text": [""]})
    return output_data


output_data = process_jsonl_file(filepath)


output_json_file_path = os.path.join(base_dir, "../subgraphs_webnlg_format.json")
# Save the output data to a JSON file
with open(output_json_file_path, "w") as output_json_file:
    json.dump(output_data, output_json_file, indent=4)
