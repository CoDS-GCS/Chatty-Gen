import yaml
import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional
from enum import Enum

class ModelType(Enum):
    OPENLLM = "openllm"
    OPENAI = "openai"
    GOOGLE = "google"

URL_PATTERN = r'^(?:([A-Za-z]+):)?(\/{0,3})([0-9.\-A-Za-z]+)(?::(\d+))?(?:\/([^?#]*))?(?:\?([^#]*))?(?:#(.*))?$'

@dataclass 
class LLMInfo:
    model_name: str = None
    model_type: ModelType = None
    model_endpoint: str = None
    model_apikey: Optional[str] = None

    def __post_init__(self):
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not self.model_type:
            raise ValueError("Model type cannot be empty")
        if not self.model_endpoint:
            raise ValueError("Model endpoint cannot be empty")
        if not re.match(URL_PATTERN, self.model_endpoint):
            raise ValueError("Model endpoint must be a valid URL")
    
    @classmethod
    def from_dict(cls, data: dict):
        data["model_type"] = ModelType(data["model_type"])
        return cls(**data)

@dataclass
class Config:
    openllm: bool = True
    modelname: str = "default_model"
    openllm_endpoint: str = "default_endpoint"
    openai_api_key: str = "default_api_key"
    kgname: str = "default_kg"
    temperature: float = 0.5
    kghost : str = "localhost"
    kgport : str = "default_kg_port"
    outputdir: str = "default_output_dir"
    dataset_size: int = 1
    dialogue_size: int = 5
    approach: List[str] = field(default_factory=list)
    pipeline_type: str = "default_pipeline_type"
    prompt: int = 1
    use_label: bool = True
    seed_nodes_file = None
    tracing: bool = True
    logging: bool = True
    question_generation_model: LLMInfo = None
    sparql_generation_model: LLMInfo = None
    dialogue_generation_model: LLMInfo = None

    # redis
    redishost: str = "localhost"
    redisport: int = 6379


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


# Example usage:
config = Config.from_yaml('config.yaml')
print(config)
