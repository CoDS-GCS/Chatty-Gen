import time
import json
import os


# Function 1: Extract and yield idx + string + time
def extract(input_data):
    for idx, string in enumerate(input_data):
        yield f"{idx} - {string} - {time.time()}"


# Function 2: Transform to lowercase
def transform(data):
    for item in data:
        yield item.lower()


# Function 3: Load into a JSONL file
def load(data, filename):
    with open(filename, "a") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


# Simulate a list of input strings
input_strings = ["Hello", "World", "Python", "ETL", "Example"]

# Define the number of workers for each step
extract_workers = 5
transform_workers = 2
load_workers = 4

# Specify the output file
output_file = "output.jsonl"


# Function to handle the ETL pipeline and file creation
def etl_pipeline(
    input_strings, extract_workers, transform_workers, load_workers, output_file
):
    try:
        # Step 1: Extract
        extract_generator = extract(input_strings)

        # Step 2: Transform
        transform_generator = transform(extract_generator)

        # Step 3: Load
        load_generators = load(transform_generator, output_file)

    except FileNotFoundError:
        # If the file doesn't exist, create it and retry the pipeline
        with open(output_file, "w"):
            pass
        etl_pipeline(
            input_strings, extract_workers, transform_workers, load_workers, output_file
        )


# Execute the ETL pipeline
if __name__ == "__main__":
    etl_pipeline(
        input_strings, extract_workers, transform_workers, load_workers, output_file
    )
