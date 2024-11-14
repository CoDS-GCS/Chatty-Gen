import yaml
import os
from itertools import product


def generate_yml_file(output_path, schema, values):
    """
    Generates a YAML file based on the schema and given values.

    Parameters:
    - output_path (str): Path where the YAML file will be saved.
    - schema (dict): Fixed schema for the YAML file.
    - values (dict): Key-value pairs to replace in the schema.
    """
    # Create a copy of the schema and update it with the values
    data = schema.copy()
    data.update(values)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the YAML data to a file
    with open(output_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"YAML file generated at: {output_path}")

def generate_configs(schema, output_dir, variations):
    """
    Generates YAML files based on all combinations of provided variations.

    Parameters:
    - schema (dict): Fixed schema for the YAML file.
    - output_dir (str): Base output directory for generated YAML files.
    - variations (dict): Dictionary with keys as fields to vary and values as lists of possible values.
    """
    # Extract field names and value lists
    fields, value_lists = zip(*variations.items())

    # Generate all combinations of values
    for combo in product(*value_lists):
        # Map the combination to field names
        values = dict(zip(fields, combo))

        # Prepare specific output path
        kgname = values['kgname']
        approach = values['approach'][0]
        # model_name = values['comman_model']['model_name']
        # model_name = "multi-llm-1"
        model_name = "multi-llm-2"
        values['outputdir'] = f"./results/{kgname}/{approach}/{model_name}"
        filename = f"{kgname}/{approach}/config_{model_name}.yml"
        output_path = os.path.join(output_dir, filename)

        # Generate the YAML file with current combination of values
        generate_yml_file(output_path, schema, values)

# Fixed schema with default values
schema = {
    'kgname': 'dblp',
    'temperature': 0.5,
    'kghost': "206.12.95.86",
    'kgport': 8894,
    'redishost': 'localhost',
    'outputdir': "",
    'dataset_size': 20,
    'dialogue_size': 5,
    'wandb_project': 'chatty-gen-benchmark',
    'approach': None,
    'pipeline_type': 'original',
    'comman_model': {
        'model_type': 'google',
        'model_name': 'gemini-1.5-pro-preview-0409',
        'model_endpoint': '',
        'model_apikey': ''
    },
    'prompt': 1,
    'use_label': True,
    'tracing': True,
    'logging': True
}

multi_llm_schema = {
    'kgname': 'dblp',
    'temperature': 0.5,
    'kghost': "206.12.95.86",
    'kgport': 8894,
    'redishost': 'localhost',
    'outputdir': "",
    'dataset_size': 20,
    'dialogue_size': 5,
    'wandb_project': 'chatty-gen-benchmark',
    'approach': None,
    'pipeline_type': 'original',
    'question_generation_model': {
        'model_type': 'google',
        'model_name': 'gemini-1.5-pro-preview-0409',
        'model_endpoint': '',
        'model_apikey': ''
    },
    'sparql_generation_model': {
        'model_type': 'google',
        'model_name': 'gemini-1.5-pro-preview-0409',
        'model_endpoint': '',
        'model_apikey': ''
    },
    'dialogue_generation_model': {
        'model_type': 'google',
        'model_name': 'gemini-1.5-pro-preview-0409',
        'model_endpoint': '',
        'model_apikey': ''
    },
    'prompt': 1,
    'use_label': True,
    'tracing': True,
    'logging': True
}

# Configurations to override specific fields
variations = {
    'kgname': ['dblp', 'yago', 'dbpedia', 'mag'],
    'approach': [
        ['original'],
        ['subgraph-summarized'],
        ['single-shot']
    ],
    'pipeline_type': ['original'],
    'comman_model': [
        {
            'model_type': 'google',
            'model_name': 'gemini-1.0-pro-002',
            'model_endpoint': '',
            'model_apikey': ''
        },
        {
            'model_type': 'google',
            'model_name': 'gemini-1.5-pro-preview-0409',
            'model_endpoint': '',
            'model_apikey': ''
        },
        {
            'model_type': 'openai',
            'model_name': 'gpt-3.5',
            'model_endpoint': '',
            'model_apikey': ''
        },
        {
            'model_type': 'openai',
            'model_name': 'gpt-4',
            'model_endpoint': '',
            'model_apikey': ''
        },
        {
            'model_type': 'openai',
            'model_name': 'gpt-4o',
            'model_endpoint': '',
            'model_apikey': ''
        },
        {
            'model_type': 'openllm',
            'model_name': 'llama-2-13b',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        },
        {
            'model_type': 'openllm',
            'model_name': 'llama-3-8b',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        },
        {
            'model_type': 'openllm',
            'model_name': 'llama-3-8b-instruct',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        },
        {
            'model_type': 'openllm',
            'model_name': 'codellama-7b',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        },
        {
            'model_type': 'openllm',
            'model_name': 'codellama-13b',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        },
        {
            'model_type': 'openllm',
            'model_name': 'mistral-7b-v0.1',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        }
    ]
}

multi_llm_1_variations = {
    'kgname': ['dblp', 'yago', 'dbpedia', 'mag'],
    'approach': [
        ['original'],
        ['subgraph-summarized'],
        ['single-shot']
    ],
    'pipeline_type': ['original'],
    'question_generation_model': 
        [{
            'model_type': 'openllm',
            'model_name': 'llama-3-8b-instruct',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        }],
    'sparql_generation_model': 
        [{
            'model_type': 'openllm',
            'model_name': 'codellama-13b',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        }],
    'dialogue_generation_model': 
        [{
            'model_type': 'openllm',
            'model_name': 'llama-3-8b-instruct',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        }]
}

multi_llm_2_variations = {
    'kgname': ['dblp', 'yago', 'dbpedia', 'mag'],
    'approach': [
        ['original'],
        ['subgraph-summarized'],
        ['single-shot']
    ],
    'pipeline_type': ['original'],
    'question_generation_model': 
        [{
            'model_type': 'openai',
            'model_name': 'gpt-4',
            'model_endpoint': '',
            'model_apikey': ''
        }],
    'sparql_generation_model': 
        [{
            'model_type': 'openllm',
            'model_name': 'codellama-13b',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        }],
    'dialogue_generation_model': 
        [{
            'model_type': 'openllm',
            'model_name': 'llama-3-8b-instruct',
            'model_endpoint': 'http://localhost:5000/',
            'model_apikey': ''
        }],
}



if __name__ == '__main__':
    output_base_dir = 'benchmark_configs'
    # generate_configs(schema, output_base_dir, variations)
    generate_configs(multi_llm_schema, output_base_dir, multi_llm_2_variations)