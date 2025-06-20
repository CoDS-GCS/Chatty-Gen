import argparse
import time
from appconfig import config
from benchmark2 import generate_dialogues

def main():
    print(config)

    kg_name = config.kgname
    dataset_size = config.dataset_size
    dialogue_size = config.dialogue_size
    approach = config.approach
    out_dir = config.outputdir
    prompt = config.prompt
    use_label = config.use_label
    seed_nodes_file = config.seed_nodes_file

    # Generating dialogues using the provided arguments
    generate_dialogues(kg_name, dataset_size, dialogue_size, approach, out_dir, prompt, use_label, seed_nodes_file)

if __name__ == "__main__":
    print("starting benchmark generation....")
    start_time = time.time()
    print("start time in epoch ", start_time)
    main()
    end_time = time.time()
    print("end time in epoch ", end_time)
    print("total time in seconds ", end_time - start_time)