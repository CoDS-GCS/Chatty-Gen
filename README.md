# Chatty-Gen Resources

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Run Experiments](#run-experiments)
5. [Contact](#contact)

## Introduction
<!-- Add-abstract -->

## System Requirements
The system requirements needed to run the codebase.

- Operating System: [e.g., Ubuntu 20.04, macOS, Windows 10]
- Python Version: [e.g., 3.8+]
- Optional Software: [e.g., Docker, Docker Compose]

## Installation 

### (without docker)
1. git clone the repo
2. `cd repo-directory`
3. `python3 -m venv .venv` create new python virtual environment
4. `sudo apt install python3-pip` make sure pip is installed
    `python3 -m pip install --upgrade pip`
5. `source .venv/bin/activate` - activate virtual environment
6. `pip3 install -r requirements.txt`

### (with docker)
1. make sure you have docker installed `docker --version`, install it if not found - [docker installation guide](https://docs.docker.com/engine/install/)
2. git clone the repo
3. `cd repo-directory`
4. `docker compose up --build`

## Run Experiments
To run the experiments you need to first configure the run-config yaml file

- example runconfig.yaml looks like below.
```yaml
kghost: 206.12.95.86 # knowledge graph sparql endpoint
kgport: 8894
redishost: localhost
outputdir: ./results/docker-test/dblp/singleshot/gpt-3.5-turbo # output directory path for generated benchmark data
kgname: dblp # the knowledge graph name
pipeline_type: original
dataset_size: 1
dialogue_size: 5
wandb_project: cov-kg-benchmark
approach: 
  - single-shot
  - subgraph-summarized
comman_model:
  model_type: "openai"
  model_name: "gpt-3.5-turbo"
  model_endpoint: ""
  model_apikey: "<OPENAI_API_KEY>"
use_label: true
tracing: true
logging: true
```

- update `benchmark/appconfig.py` with location of your runconfig.yaml file

### Run experiments without docker
- activate virtual environment `source .venv/bin/activate`
- install dependecies `pip install -r requirements.txt`
- make sure you have updated the `runconfig-yaml` and its path in `benchmark/appconfig.py`
- run `python3 benchmark/main.py`, will store the generated data at outputdir path in runconfig-yaml

### Run experiments with docker
- make sure you have updated the `runconfig-yaml` and its path in `benchmark/appconfig.py`
- run `docker compose up --build`