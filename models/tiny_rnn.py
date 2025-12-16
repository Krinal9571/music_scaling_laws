import torch
import torch.nn as nn

class MiniLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x, hidden=None):
        x_emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(x_emb, hidden)
        out = self.ln(out)
        logits = self.fc_out(out)
        return logits, hidden
    
    @torch.no_grad()
    def generate(self, start_ids, max_new=100, temp=0.8, top_k=None):
        self.eval()
        generated = start_ids.clone()
        hidden = None
        
        for _ in range(max_new):
            logits, hidden = self(generated[:, -1:], hidden)
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

def create_rnn(vocab_size, config_name='tiny'):
    configs = {
        'tiny': {'embed_dim': 32, 'hidden_dim': 64, 'num_layers': 2},
        'small': {'embed_dim': 64, 'hidden_dim': 128, 'num_layers': 2},
        'medium': {'embed_dim': 96, 'hidden_dim': 192, 'num_layers': 3},
        'large': {'embed_dim': 128, 'hidden_dim': 256, 'num_layers': 3},
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    
    config = configs[config_name]
    return MiniLSTM(vocab_size, **config)