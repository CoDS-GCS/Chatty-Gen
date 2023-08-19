# generate_finetune_app.py
# Author: @ozx1812
# Date: 2023/08/18

# The following code has been adopted and modified from [alpaca-lora].
# Source: https://github.com/tloen/alpaca-lora

# Modifications:
# Flask endpoint for LLAMA + PEFT text generation


from flask import Flask, request, jsonify
from llama_singleton import ModelSingleton
from queue import Queue
from threading import Thread

app = Flask(__name__)
model_instance = ModelSingleton()


class InferenceWorker(Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            instruction, response_queue = self.queue.get()
            try:
                # Perform inference using the preloaded model instance
                # ...
                generated_text = model_instance.generate_response(instruction)

                # Add the generated response to the response queue
                response_queue.put(generated_text)
            except Exception as e:
                error_response = {"error": str(e)}
                response_queue.put(error_response)
            self.queue.task_done()


inference_queue = Queue()
for _ in range(2):  # Number of worker threads
    worker = InferenceWorker(inference_queue)
    worker.daemon = True
    worker.start()


@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.json
        instruction = data.get("instruction")
        # Add your parameters here
        # ...
        response_queue = Queue()

        # Add the request to the inference queue
        inference_queue.put((instruction, response_queue))

        # Wait for the response from the worker thread
        generated_response = response_queue.get()
        response_queue.task_done()

        return jsonify({"response": generated_response})
    except Exception as e:
        error_response = {"error": str(e)}
        return jsonify(error_response), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
