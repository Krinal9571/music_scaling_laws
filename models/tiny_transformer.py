import torch
import torch.nn as nn
import math

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, num_layers=2, dim_ff=64, max_len=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.register_buffer('causal_mask', 
                           torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x, targets=None):
        B, T = x.shape
        
        x_emb = self.dropout(self.embedding(x) + self.pos_encoding[:, :T, :])
        
        mask = self.causal_mask[:T, :T]
        out = self.transformer(x_emb, mask=mask, is_causal=True)
        out = self.ln_f(out)
        logits = self.fc_out(out)
        
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=0
            )
            return logits, loss
        
        return logits
    
    @torch.no_grad()
    def generate(self, start_ids, max_new=100, temp=0.8, top_k=None):
        self.eval()
        generated = start_ids.clone()
        
        for _ in range(max_new):
            if generated.size(1) >= self.max_len:
                generated = generated[:, -self.max_len:]
            
            logits = self(generated)
            next_logits = logits[:, -1, :] / temp
            
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_id], dim=1)
            
            if next_id.item() == 0:
                break
        
        return generated

def create_transformer(vocab_size, config_name='tiny'):
    configs = {
        'tiny': {'d_model': 32, 'nhead': 2, 'num_layers': 2, 'dim_ff': 64},
        'small': {'d_model': 64, 'nhead': 4, 'num_layers': 3, 'dim_ff': 128},
        'medium': {'d_model': 96, 'nhead': 6, 'num_layers': 4, 'dim_ff': 192},
        'large': {'d_model': 128, 'nhead': 8, 'num_layers': 5, 'dim_ff': 256},
        'xlarge': {'d_model': 160, 'nhead': 8, 'num_layers': 6, 'dim_ff': 320},
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    
    config = configs[config_name]
    return MiniTransformer(vocab_size, **config)