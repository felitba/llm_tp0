import torch
import torch.nn as nn
import config.config as config
import model.tokenizer as tk

# TODO: which embedding should we use?
class TokenEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        config_data = config.load_config()

        self.embedding = nn.Embedding(
            config_data["vocab_size"],
            config_data["d_model"]
        )

    def forward(self, tokens):
        return self.embedding(tokens)

if __name__ == "__main__":
    embedding = TokenEmbedding()
    sample_tokens = torch.tensor(tk.tokenize("Hello, world!"))
    embedded_output = embedding(sample_tokens)
    print("Embedded output shape:", embedded_output)  