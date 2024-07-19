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