import yaml
from dataclasses import dataclass, field
from typing import Any, List, Tuple

@dataclass
class Config:
    openllm: bool = True
    modelname: str = "default_model"
    openllm_endpoint: str = "default_endpoint"
    openai_api_key: str = "default_api_key"
    kgname: str = "default_kg"
    temperature: float = 0.5
    kghost : str = "default_kg_url"
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
    pipeline_stages: List[Tuple[int, str, str]] = None

    # redis
    redishost: str = "localhost"
    redisport: int = 6379


    @classmethod
    def from_yaml(cls, file_path: str) -> 'Config':
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
            return cls(**config_data)

    def to_yaml(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            yaml.dump(self.__dict__, file)


# Example usage:
config = Config.from_yaml('config.yaml')
print(config)
