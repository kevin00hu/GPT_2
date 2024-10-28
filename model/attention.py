import math

import torch
import torch.nn as nn

class multihead_attention(nn.Module):
    def __init__(self, dim:int=768, n_head:int=8, dropout:float=0.1):
        super().__init__()
        self.n_head = n_head

        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)

        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.concat_feedforward = nn.Linear(dim, dim)
        

    def forward(self, x):
        B, T, C = x.size()

        temp = self.ln_1(x)
        # (B, T, C)
        Q = self.proj_q(temp)
        K = self.proj_k(temp)
        V = self.proj_v(temp)

        # (B, T, H, C/H) - > (B, H, T, C/H)
        Q = Q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        K = K.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        V = V.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attn -> (B, H, T, T)
        attn_result = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(C // self.n_head)
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(Q.device)
        attn_result = attn_result.masked_fill(mask == 0, float("-inf"))
        attn_result = nn.functional.softmax(attn_result, dim=-1)
        attn_result = self.attn_dropout(attn_result)

        # @ V -> (B, H, T, C/H)
        attn_result = torch.matmul(attn_result, V)
        # concat -> (B, T, H, C/H) -> (B, T, C)
        attn_result = attn_result.transpose(1, 2).reshape(B, T, C)
        # feedforward -> (B, T, C)
        attn_result = self.concat_feedforward(attn_result)

        return attn_result 
        
class feedforward(nn.Module):
    def __init__(self, dim:int=768, rate:int=4, dropout:float=0.1):
        super().__init__()

        self.encoder = nn.Linear(dim, dim * rate)
        self.decoder = nn.Linear(dim * rate, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = x * nn.functional.sigmoid(x)
        x = self.dropout(x)
        x = self.decoder(x)
        return x

class attention_block(nn.Module):
    def __init__(self, dim:int=768, n_head:int=8, rate:int=4, dropout:float=0.1):
        super().__init__()
        
        self.attention_layer = multihead_attention(dim, n_head, dropout)
        self.feedforward_layer = feedforward(dim, rate, dropout)

        self.residual_dropout = nn.Dropout(dropout)

        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)

    def forward(self, x):
        # attention
        temp = self.ln_1(x)
        temp = self.attention_layer(temp)
        x = x + temp
        # dropout after residual
        x = self.residual_dropout(x)
        # feedforward
        temp = self.ln_2(x)
        temp = self.feedforward_layer(temp)
        x = x + temp
        return x

if __name__ == "__main__":
    attention_block_example = attention_block()
    x = torch.rand((1, 1024, 768))
    print(x.shape)
    result = attention_block_example(x)
    print(result.shape)
