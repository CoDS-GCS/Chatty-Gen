import os
import tiktoken
from appconfig import config, LLMInfo, ModelType
from llm.openllm_local import OpenLLM
from langchain.llms import OpenAI

tiktoken_cache_dir = "../tiktoken-cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

openllm_config = {
    'max_new_tokens': 512,
    'early_stopping': "```",
    'do_sample': True
}

def get_llm_from_config(model_config: LLMInfo):
    llm = None
    try:
        if model_config.model_type == ModelType.OPENAI:
            llm = OpenAI(model_name=model_config.model_name, temperature=0.5, streaming=False, api_key=model_config.model_apikey)
            llm_config = None
        if model_config.model_type == ModelType.OPENLLM:
            llm = OpenLLM(server_url=model_config.model_endpoint)
            llm_config = openllm_config
        if model_config.model_type == ModelType.GOOGLE:
            pass
        return {"llm": llm, "config": llm_config}
    except Exception as e:
        print(e)

llms_dict = {
    "question_generation_model": get_llm_from_config(config.question_generation_model),
    "sparql_generation_model": get_llm_from_config(config.sparql_generation_model),
    "dialogue_generation_model": get_llm_from_config(config.dialogue_generation_model),
}

openai_embedding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def get_num_tokens(prompt):
    return len(openai_embedding.encode(prompt))