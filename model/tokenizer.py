import tiktoken
from config.config import load_config

config = load_config()
# TODO: which tokenizer should we use?
ENCODING = tiktoken.get_encoding(config.get("encoding"))

def tokenize(text)-> list[int]:
    return ENCODING.encode(text)

def detokenize(tokens: list[int]) -> str:
    return ENCODING.decode(tokens)

if __name__ == "__main__":
    sample_text = "Hello, world!"
    tokens = tokenize(sample_text)
    print(f"Tokens for '{sample_text}': {tokens}")
    detokenized_text = detokenize(tokens)
    print(f"Detokenized text: {detokenized_text}")