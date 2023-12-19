import argparse
import os
from benchmark2 import  generate_dialogues


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
        help=f"Choose a value for kg ({kg_choices})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./out",
        help="Specify the output directory path",
    )

    # Parsing the arguments
    args = parser.parse_args()

    # Accessing the value of kg argument
    kg_name = args.kg

    # Checking if kg value is valid
    if kg_name is None:
        parser.print_help()

    generate_dialogues(kg_name)



if __name__ == "__main__":
    print("starting benchmark generation....")
    main()
