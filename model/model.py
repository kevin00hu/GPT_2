import torch
import torch.nn as nn

from .attention import attention_block

class GPT2(nn.Module):
    def __init__(self, token_size:int, sequence_length:int=1024, dim:int=768, block_num:int=12):
        super().__init__()

        self.token_embed = nn.Embedding(token_size, dim)
        self.position_embed = nn.Embedding(sequence_length, dim)

        self.attention_block_list = nn.ModuleList()

        for _ in range(block_num):
            self.attention_block_list.append(
                attention_block()
            )

        self.ln_final = nn.LayerNorm(dim)
        self.decode_mapping = nn.Linear(dim, token_size)
        
    def forward(self, x):
        B, T = x.shape
        # x: (B, T)
        token_embedding = self.token_embed(x)
        positional_embedding = self.position_embed(torch.arange(0, T, dtype=int).to(x.device))
        # x: (B, T, C)
        x = positional_embedding + token_embedding

        for attn_layer in self.attention_block_list:
            x = attn_layer(x)

        x = self.ln_final(x) # (B, T, C)
        x = self.decode_mapping(x) # (B, T, V)
        # x = nn.functional.softmax(x, dim=-1) 
        return x

if __name__ == "__main__":
    model = GPT2(50257)

    import torch

    x = torch.randint(1, 1000, (2, 512))
    output = model(x)

    print(x.shape)
    print(output.shape)
