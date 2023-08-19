Use requirements_cc.txt for compute canada, also load this modules.
```module load python/3.10 arrow/8 gcc/9.3```

# Finetuning

- To finetune llama model using PEFT method LORA, follow below steps.

- Instruction data
    - [alpaca-data.json](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json)
    ```python finetune.py \
        --base_model 'decapoda-research/llama-7b-hf' \
        --data_path 'alpaca-data.json' \
        --output_dir './lora-alpaca'
    ```
- Only Text data `no_instruction_text.json`
    ```json
    ["input": "i am an example of text data...",]
    ```
    ```python finetune.py \
        --base_model 'decapoda-research/llama-7b-hf' \
        --data_path 'no_instruction_text.json' \
        --output_dir './lora-alpaca-no-instruction' \
        --batch_size 128 \
        --micro_batch_size 4 \
        --num_epochs 3 \
        --learning_rate 1e-4 \
        --cutoff_len 512 \
        --val_set_size 20 \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
        --train_on_inputs \
        --group_by_length \
        --no_prompt True # mandatory to use only `input`
    ```

# Generation
- To generate text from original llama model. Update with custom examples.
    `python generate_original.py`


- To generate text from finetuned model.
- 1) Gradio App
    ```python generate.py \
        --base_model 'local-path or huggingface model name' \
        --lora_weights 'local-path or huggingface PEFT weights' \
    ```

- 2) Test locally using flask server
    - generic template example
        ```bash
        export BASE_MODEL="local-path or huggingface model name"
        export PEFT_MODEL="local-path or huggingface PEFT weights"
        python generate_finetune_app.py
        curl -X POST \
          -H "Content-Type: application/json" \
          -d '{
            "instruction": "Tell me about alpacas.",
          }' \
          http://localhost:5000/generate
        ```

    - working example for llama trained on dblp text data.
        ```bash
        export BASE_MODEL="decapoda-research/llama-7b-hf"
        export PEFT_MODEL="0mij/llama-dblp-adapter-v1"
        python generate_finetune_app.py
        curl -X POST \
          -H "Content-Type: application/json" \
          -d '{
            "instruction": "Instruction: answer in yes/no, Question: is Parisa Memarmosherfi author of Hang Zang et al.: Investing the Learning Phase of an autonomus authentication in mobile ad-hoc networks.(2016)? Output: ",
          }' \
          http://localhost:5000/generate
        ```
