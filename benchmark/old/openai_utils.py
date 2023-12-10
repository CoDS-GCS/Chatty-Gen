import openai
import tiktoken

# Encoding name	OpenAI models
# cl100k_base	gpt-4, gpt-3.5-turbo, text-embedding-ada-002
# p50k_base	Codex models, text-davinci-002, text-davinci-003
# r50k_base (or gpt2)	GPT-3 models like davinci

encoding = tiktoken.get_encoding("p50k_base")


def encode_and_count(text: str) -> int:
    tokens = len(encoding.encode(text))
    return tokens


def openai_generate(text: str) -> str:
    pass


if __name__ == "__main__":
    assert encode_and_count("hello how are you?") == 5
