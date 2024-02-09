import sys
sys.path.append('../')
import json
from llm.prompt_chains import get_prompt_chains

def evaluate_question_quality_old(data, chain):
    valid, not_valid = 0, 0
    for inst in data:
        dialogue = inst['dialogue']
        output = chain.get("chain").run({"questions": dialogue})
        output = output.lower().strip().replace('.', '')
        # result = output["output"]
        if output == 'valid':
            valid += 1
        elif output == 'not valid':
            not_valid += 1
        else:
            print("Result:\n", output)
    return valid, not_valid

def evaluate_question_quality(data, chain):
    valid, not_valid = 0, 0
    for inst in data:
        entity = inst['seed_entity']
        dialogue = inst['dialogue']
        output = chain.get("chain").run({"entity": entity, "dialogue": dialogue})
        print(output)
        if output == 'valid':
            valid += 1
        elif output == 'not valid':
            not_valid += 1
        else:
            print("Result:\n", output)
    return valid, not_valid

if __name__ == '__main__':
    file = '../Debug/dblp_e11_20_5.json'
    file = open(file, 'r')
    file_data = json.load(file)
    prompt_chains = get_prompt_chains()
    # chain = prompt_chains.get("get_validate_question_quality_old")
    # valid, not_valid = evaluate_question_quality_old(file_data['data'], chain)
    # print("Valid: ", valid, '\t Not valid:', not_valid)
    chain = prompt_chains.get("get_validate_question_quality")
    valid, not_valid = evaluate_question_quality(file_data['data'], chain)
    print("Valid: ", valid, '\t Not valid:', not_valid)