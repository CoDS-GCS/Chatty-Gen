import yaml
import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional
from enum import Enum
import os

class ModelType(Enum):
    OPENLLM = "openllm"
    OPENAI = "openai"
    GOOGLE = "google"

URL_PATTERN = r'^(?:([A-Za-z]+):)?(\/{0,3})([0-9.\-A-Za-z]+)(?::(\d+))?(?:\/([^?#]*))?(?:\?([^#]*))?(?:#(.*))?$'

@dataclass 
class LLMInfo:
    model_name: str = None
    model_type: ModelType = None
    model_endpoint: Optional[str] = None
    model_apikey: Optional[str] = None

    def __post_init__(self):
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not self.model_type:
            raise ValueError("Model type cannot be empty")
        if self.model_endpoint and not re.match(URL_PATTERN, self.model_endpoint):
            raise ValueError("Model endpoint must be a valid URL")
    
    @classmethod
    def from_dict(cls, data: dict):
        data["model_type"] = ModelType(data["model_type"])
        return cls(**data)

@dataclass
class Config:
    kgname: str = "default_kg"
    temperature: float = 0.5
    kghost : str = "localhost"
    kgport : str = "default_kg_port"
    outputdir: str = "default_output_dir"
    dataset_size: int = 1
    dialogue_size: int = 5
    approach: List[str] = field(default_factory=list)
    pipeline_type: str = None
    prompt: int = 1
    use_label: bool = True
    seed_nodes_file = None
    tracing: bool = True
    logging: bool = True

    comman_model: Optional[LLMInfo] = None

    question_generation_model: LLMInfo = None
    sparql_generation_model: LLMInfo = None
    dialogue_generation_model: LLMInfo = None

    # redis
    redishost: str = "localhost"
    redisport: int = 6379

    # wandb
    wandb_project: str = "cov-kg-benchmark-3"
    wandb_mode: str = "offline"


    @classmethod
    def from_yaml(cls, file_path: str) -> 'Config':
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)

        # Convert LLMInfo dictionaries to LLMInfo instances
        for field_name in ['question_generation_model', 'sparql_generation_model', 'dialogue_generation_model']:
            model_dict = config_data.get(field_name)
            if model_dict:
                config_data[field_name] = LLMInfo.from_dict(model_dict)

        return cls(**config_data)

    def to_yaml(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            yaml.dump(self.__dict__, file)
    
    def config_for_wandb(self) -> dict:
        _config = {
            "kgname": self.kgname,
            "dataset_size": self.dataset_size,
            "approach": self.approach,
            "pipeline_type": self.pipeline_type,
            "use_label": self.use_label,
            "llm_models": {
                "question_generation": self.question_generation_model.model_name,
                "sparql_generation": self.sparql_generation_model.model_name,
                "dialogue_generation": self.dialogue_generation_model.model_name,
            }
        }
        return _config
    
    def used_llms(self) -> str:
        return f"{self.question_generation_model.model_name}-{self.sparql_generation_model.model_name}-{self.dialogue_generation_model.model_name}"
        

# Example usage:
# yamlfile = "./run_configs/dblp/original/codellama13b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/dblp/simplified/codellama13b-config.yaml"
# config = Config.from_yaml(yamlfile)
yamlfile = "./run_configs/yago/original/codellama13b-config.yaml"
config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/yago/simplified/codellama13b-config.yaml"
# config = Config.from_yaml(yamlfile)

# yamlfile = "./run_configs/dblp/original/codellama7b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/dblp/simplified/codellama7b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/yago/original/codellama7b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/yago/simplified/codellama7b-config.yaml"
# config = Config.from_yaml(yamlfile)


# yamlfile = "./run_configs/dblp/original/mistral-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/dblp/simplified/mistral-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/yago/original/mistral-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/yago/simplified/mistral-config.yaml"
# config = Config.from_yaml(yamlfile)


# yamlfile = "./run_configs/dblp/original/llama7b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/dblp/simplified/llama7b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/yago/original/llama7b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/yago/simplified/llama7b-config.yaml"
# config = Config.from_yaml(yamlfile)


# yamlfile = "./run_configs/dblp/original/llama13b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/dblp/simplified/llama13b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/yago/original/llama13b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/yago/simplified/llama13b-config.yaml"
# config = Config.from_yaml(yamlfile)


# ## openai model
# yamlfile = "config.yaml"
# config = Config.from_yaml(yamlfile)

if config.wandb_project != "":
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_MODE"] = "offline"
print(config)
