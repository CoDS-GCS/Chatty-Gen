import argparse
from benchmark2 import generate_dialogues

def main():
    # Creating the argument parser
    parser = argparse.ArgumentParser(description="KG dialogue benchmark generation")

    # Adding argument for kg with choices
    kg_choices = ["yago", "dbpedia", "dblp", "mag", "other"]
    parser.add_argument(
        "--kg",
        type=str,
        required=True,
        choices=kg_choices,
        help=f"Choose a value for kg ({', '.join(kg_choices)})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Specify the output directory path",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=2,
        help="Specify the dataset size",
    )
    parser.add_argument(
        "--dialogue-size",
        type=int,
        default=5,
        help="Specify the dialogue size",
    )
    parser.add_argument(
        "--approach",
        nargs='+',  # Allows multiple values
        type=str,
        default=["subgraph"],  # Default value if not provided
        choices=["subgraph", "subgraph-summarized"],  # Available choices
        help="Specify the approach (options: 'subgraph', 'subgraph-summarized')",
    )
    # parser.add_argument(
    #     "--label-predicate",
    #     type=str,
    #     default="http://www.w3.org/2000/01/rdf-schema#label",
    #     help="Specify the representative label predicate in KG",
    # )
    parser.add_argument(
        "--label-predicates-file-name",
        type=str,
        default="dblp_predicates_for_types.json",
        help="Specify the representative label predicate in KG",
    )

    # Parsing the arguments
    args = parser.parse_args()

    # Accessing the values of arguments
    kg_name = args.kg
    dataset_size = args.dataset_size
    dialogue_size = args.dialogue_size
    approach = args.approach
    out_dir = args.output_dir
    # label_predicate = args.label_predicate
    label_predicate = args.label_predicates_file_name

    # Generating dialogues using the provided arguments
    generate_dialogues(kg_name, dataset_size, dialogue_size, approach, label_predicate, out_dir)

if __name__ == "__main__":
    print("starting benchmark generation....")
    main()