# Chatbot-Resources


# How to fine-tune llama model

- Before understanding fine-tuning lets walk through some basics.

## LLM - finetuning
**Q**: What are the methods used in the fine tuning process of LLMs?

- Few-shot learning method
- In this approach, the model is provided with a very limited number of examples (i.e., “few shots”) from the new task, and it uses this information to adapt and perform well on that task.
- In the context of LLMs, fine-tuning with a small dataset related to the new task is an example of few-shot learning.

- *Fine-tuning methods*
- Transfer learning
- Sequential Fine-tuning - pre-trained model is fine-tuned on multiple related tasks or domains sequentially.
- Task-specific Fine-tuning - the pre-trained model is fine-tuned on a specific task or domain using a task-specific dataset.
- Multi-task Learning - the pre-trained model is fine-tuned on multiple tasks simultaneously.
- Adapter Training - training lightweight modules that are plugged into the pre-trained model, allowing for fine-tuning on a specific task without affecting the original model’s performance on other tasks.


Each method is effective in given situation and depends on task, dataset, resources at hand.

- To train the LLAMA models, it requires huge resources even for a 7B model. That's where the PEFT fine-tuning comes into the picture.

**Q**: what is PEFT?
- " Parameter-Efficient Fine-Tuning (PEFT) methods: enable efficient adaptation of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model's parameters. Fine-tuning large-scale PLMs is often prohibitively costly. In this regard, PEFT methods only fine-tune a small number of (extra) model parameters, thereby greatly decreasing the computational and storage costs. Recent State-of-the-Art PEFT techniques achieve performance comparable to that of full fine-tuning." - [PEFT-HF](https://github.com/huggingface/peft)

- Now to train the LLAMA model using LORA-PEFT method, we will use [alpaca-lora](https://github.com/tloen/alpaca-lora). This repository contains code for reproducing the Stanford Alpaca results using low-rank adaptation (LoRA).
- It uses LORA method from [PEFT by Huggingface](https://github.com/huggingface/peft) and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for 8-bit matrix multiplication.
- This makes the fine-tuning process too much faster, while keeping up with original model's performance.

### Steps to fine-tune
1) Gather data
    - The training data can be vary based on the task and domain, it decides the outcome or ability of fine-tuned model.
    - Examples., boolean maths question answering data, NER data, sentiment data, financial data (domain-specific), etc.
    - One such example is alpaca-instruction-format data.
    - "We introduce [Alpaca 7B](https://github.com/tatsu-lab/stanford_alpaca), a model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations." - [blog](https://crfm.stanford.edu/2023/03/13/alpaca.html)
    - We used the following prompts for fine-tuning the Alpaca model - [Alpaca-LLAMA](https://github.com/tatsu-lab/stanford_alpaca)
    
    - for examples with a non-empty input field:
    
     ```
     Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
     
     ### Instruction:
     {instruction}
     
     ### Input:
     {input}
     
     ### Response:
     ```
    
    - for examples with an empty input field:
     ```
     Below is an instruction that describes a task. Write a response that appropriately completes the request.
     
     ### Instruction:
     {instruction}
     
     ### Response:
     ```
    - We will go with ~1% of alpaca instruction-following dataset.
2) Fine tune model
    - Train the model using alpaca-lora repo's `finetune.py`.
        ```bash
        python finetune.py \
            --base_model 'decapoda-research/llama-7b-hf' \
            --data_path 'yahma/alpaca-cleaned' \
            --output_dir './lora-alpaca'
        ```
    - This will train the adapter and save it, Now to use this adapter we need to concate it with original weights.
3) Concate with LLAMA-original
    - There are two types of model weights - 1) sharded and 2) consolidated.
    - 1) sharded weights or huggingface version weights are format for uploading it to huggingface or loading it from.
    - 2) consolidated weights - pytorch's bin file or original pytorch model weights from META.
    - First we Concate the Adapter's Weights to LLAMA's Huggingface Weights, this will allow us to make our model published to huggingface hub or to use it from transformer library.
        ```bash
        export BASE_MODEL=BASE_MODEL=decapoda-research/llama-7b-hf
        python export_hf_checkpoint.py
        ```
        - This will concate adapter to LLAMA and will save it to `hf_ckpt` directory (sharded version).
4) Consolidate Weights
    - Now Huggingface's LLAMA model can use sharded version of its weights or model on huggingface-hub. But in resource scar environment (or only CPU) we need to use 8-bit or 4-bit inference. For that first we need to consolidate sharded weights of LLAMA model (from huggingface version to pytorch version).
        ```bash
        python export_state_dict_checkpoint.py
        ```
        - This will consolidate weights into multiple `consolidated-0*.bin` files.
5) CPU-only Inference
    - After getting consolidated weights, for CPU-only inference we will use [llama.cpp](https://github.com/ggerganov/llama.cpp), "The main goal of llama.cpp is to run the LLaMA model using 4-bit integer quantization on a MacBook".
        ```bash
        python convert.py /saved/path/of/consolidated/weights/from/previous/step
        ```
        - This will convert the model to ggml FP16 format.
    - Quantize the model to 4-bits (using q4_0 method)
        ```bash
        ./quantize /converted/model/ggml-model-f16.bin /convered/model/ggml-model-q4_0.bin q4_0
        ```
    - Run the inference
        ```bash
        ./main -m /convered/model/ggml-model-q4_0.bin -n 128 -p "Building a website can be done in 10 simple steps:"
        ```
        ```bash
        Building a website can be done in 10 simple steps:
        1) Select a domain name and web hosting plan
        2) Complete a sitemap
        3) List your products
        4) Write product descriptions
        5) Create a user account
        6) Build the template
        7) Start building the website
        8) Advertise the website
        9) Provide email support
        10) Submit the website to search engines
        ...........
        ```

