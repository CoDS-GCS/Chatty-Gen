FROM ubuntu:22.04
WORKDIR /app
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip
COPY benchmark /app/benchmark
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD [ "python3", "benchmark/main.py" ]