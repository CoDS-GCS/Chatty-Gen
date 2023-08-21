# llama_singleton.py
# Author: @ozx1812
# Date: 2023/08/18

# The following code has been adopted and modified from [alpaca-lora].
# Source: https://github.com/tloen/alpaca-lora

# Modifications:
# Singleton LLAMA + PEFT Adapter

import os
import sys
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


class ModelSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            load_8bit = False,
            base_model = "decapoda-research/llama-7b-hf"
            lora_weights = "0mij/llama-dblp-adapter-v1"
            base_model = base_model or os.environ.get("BASE_MODEL", "")
            assert (
                base_model
            ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

            cls._instance = super().__new__(cls)
            cls._instance.tokenizer = LlamaTokenizer.from_pretrained(base_model)

            if device == "cuda":
                model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    load_in_8bit=load_8bit,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                cls._instance.model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    torch_dtype=torch.float16,
                )
            elif device == "mps":
                model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    device_map={"": device},
                    torch_dtype=torch.float16,
                )
                cls._instance.model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    device_map={"": device},
                    torch_dtype=torch.float16,
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    base_model, device_map={"": device}, low_cpu_mem_usage=True
                )
                cls._instance.model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    device_map={"": device},
                )

            # unwind broken decapoda-research config
            cls._instance.model.config.pad_token_id = cls._instance.tokenizer.pad_token_id = 0  # unk
            cls._instance.model.config.bos_token_id = 1
            cls._instance.model.config.eos_token_id = 2

            if not load_8bit:
                cls._instance.model.half()  # seems to fix bugs for some users.

            cls._instance.model.eval()
            if torch.__version__ >= "2" and sys.platform != "win32":
                cls._instance.model = torch.compile(cls._instance.model)

        return cls._instance

    def generate_response(
        self,
        instruction,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = instruction
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return output.strip()

    def generate_response_batch(
        self,
        instruction_batch,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        encodings = self.tokenizer(instruction_batch, return_tensors="pt").to(self.model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_outputs = self.model.generate(
                **encodings,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.batch_decode(generation_outputs)
# Usage:
# model_instance = ModelSingleton()
