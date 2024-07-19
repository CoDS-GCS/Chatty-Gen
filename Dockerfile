FROM ubuntu:22.04
WORKDIR /app
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY benchmark /app/benchmark
CMD [ "python3", "benchmark/main.py" ]