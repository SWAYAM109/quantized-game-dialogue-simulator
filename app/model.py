import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention written from scratch (plain nn.Linear layers)."""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)   # Q, K, V in one projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = self.attn_drop(F.softmax(att.masked_fill(mask, float("-inf")), dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    """Pre-norm transformer block: attention + feed-forward, both with residuals."""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.fc1, self.fc2 = nn.Linear(n_embd, 4 * n_embd), nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class GPTLanguageModel(nn.Module):
    """Compact decoder-only GPT built from scratch with causal self-attention."""
    def __init__(self, vocab_size, block_size=96, n_embd=96, n_head=4, n_layer=2, dropout=0.1):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        self.vocab_size, self.block_size = vocab_size, block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size={self.block_size}.")
        pos = torch.arange(seq_len, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos)[None, :, :])
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        if temperature <= 0: raise ValueError("temperature must be greater than zero.")
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
