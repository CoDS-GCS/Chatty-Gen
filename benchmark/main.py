import argparse
import os
from benchmark import generate_question_set
from kg.yago.yago import YAGO
from kg.dblp.dblp import DBLP 


def get_kg_instance(kg_name):
    kgs = {"yago": YAGO(), "dblp": DBLP()}
    kg = kgs.get(kg_name, None)
    if kg is None:
        raise ValueError(f"kg : {kg_name} not supported")
    return kg


# approach 1

# approach 2


def main():
    # Creating the argument parser
    parser = argparse.ArgumentParser(description="KG dialogue benchmark generation")

    # Adding argument for kg with choices
    kg_choices = ["yago", "dbpedia", "dblp", "mag", "other"]
    parser.add_argument(
        "--kg",
        type=str,
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

    kg = get_kg_instance(kg_name)
    generate_question_set(kg)


if __name__ == "__main__":
    print("starting benchmark generation....")
    main()
