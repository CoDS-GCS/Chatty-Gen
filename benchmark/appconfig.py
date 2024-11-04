import yaml
import re
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import os


class ModelType(Enum):
    OPENLLM = "openllm"
    OPENAI = "openai"
    GOOGLE = "google"


URL_PATTERN = r"^(?:([A-Za-z]+):)?(\/{0,3})([0-9.\-A-Za-z]+)(?::(\d+))?(?:\/([^?#]*))?(?:\?([^#]*))?(?:#(.*))?$"


@dataclass
class LLMInfo:
    model_name: str = ""
    model_type: ModelType = ""
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
    question_generation_model: LLMInfo
    sparql_generation_model: LLMInfo
    dialogue_generation_model: LLMInfo

    kgname: str = "default_kg"
    temperature: float = 0.5
    kghost: str = "localhost"
    kgport: str = "default_kg_port"
    outputdir: str = "default_output_dir"
    dataset_size: int = 1
    dialogue_size: int = 5
    approach: List[str] = field(default_factory=list)
    pipeline_type: str = ""
    prompt: int = 1
    use_label: bool = True
    seed_nodes_file: str = None
    tracing: bool = True
    logging: bool = True

    time_sleep: bool = False

    comman_model: Optional[LLMInfo] = None

    # redis
    redishost: str = "localhost"
    redisport: int = 6379
    redis_url: str = "redis://default:admin@localhost:6379/0"

    # wandb
    wandb_project: str = "cov-kg-benchmark-27april"
    wandb_mode: str = "offline"

    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

            # Convert LLMInfo dictionaries to LLMInfo instances
            if "comman_model" in config_data:
                comman_model_dict = config_data.get("comman_model")
            else:
                if "question_generation_model" not in config_data or "sparql_generation_model" not in config_data or "dialogue_generation_model" not in config_data:
                    raise ValueError("If comman_model is not provided, all three models must be present.")
                comman_model_dict = None

            for field_name in [
                "question_generation_model",
                "sparql_generation_model",
                "dialogue_generation_model",
            ]:
                config_data[field_name] = LLMInfo.from_dict(
                    comman_model_dict if comman_model_dict else config_data.get(field_name)
                    )

                if config_data[field_name].model_type == ModelType.GOOGLE:
                    config_data["time_sleep"] = config_data["time_sleep"] or True
                else:
                    config_data["time_sleep"] = False

            return cls(**config_data)

    def to_yaml(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            yaml.dump(self.__dict__, file)

    def config_for_wandb(self) -> dict:
        _llm_models = None
        _llm_models = {
                "question_generation": self.question_generation_model.model_name,
                "sparql_generation": self.sparql_generation_model.model_name,
                "dialogue_generation": self.dialogue_generation_model.model_name,
            }
        _config = {
            "kgname": self.kgname,
            "dataset_size": self.dataset_size,
            "approach": self.approach,
            "pipeline_type": self.pipeline_type,
            "use_label": self.use_label,
            "llm_models": _llm_models
        }
        return _config

    def used_llms(self) -> str:
        return f"{self.question_generation_model.model_name}-{self.sparql_generation_model.model_name}-{self.dialogue_generation_model.model_name}"


# Example usage:
# yamlfile = "./run_configs/dblp/original/codellama13b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/dblp/simplified/codellama13b-config.yaml"
# config = Config.from_yaml(yamlfile)
# yamlfile = "./run_configs/yago/original/codellama13b-config.yaml"
# config = Config.from_yaml(yamlfile)
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
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
# yamlfile = "config.yaml"
# config = Config.from_yaml(yamlfile)

# yamlfile = "./run_configs/yago-comman.yaml"
# config = Config.from_yaml(yamlfile)

yamlfile = "run_configs/test_config.yaml"
yamlfile = os.path.join(CURR_DIR, yamlfile)
config = Config.from_yaml(yamlfile)

if config.wandb_project != "":
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_MODE"] = "offline"
