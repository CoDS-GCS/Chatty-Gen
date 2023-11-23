import json
from typing import List, Any


def read_json(filename: str):
    with open(filename, "r") as f:
        json_data = json.load(f)
        return json_data


def read_jsonl(filename: str) -> List[Any]:
    with open(filename, "r") as f:
        data = f.readlines()
        json_data = [json.loads(d) for d in data]
        return json_data


def replace_placeholders(string_template: str, replacement_dict: dict) -> str:
    """
    Replace placeholders in a string template with values from a dictionary.
    """
    formatted_string = string_template % replacement_dict
    return formatted_string
