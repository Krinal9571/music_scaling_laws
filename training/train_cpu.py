import os
import sys
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.tiny_transformer import create_transformer
from models.tiny_rnn import create_rnn

class SimpleDataset(Dataset):
    def __init__(self, sequences, block_size=64):
        self.sequences = sequences
        self.block_size = block_size
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        if len(seq) > self.block_size + 1:
            start = random.randint(0, len(seq) - self.block_size - 1)
            chunk = seq[start:start + self.block_size + 1]
        else:
            chunk = seq + [0] * (self.block_size + 1 - len(seq))
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_model(model, model_name, train_loader, val_loader, epochs=2):
    print(f"\nTraining {model_name}:")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            
            if hasattr(model, 'transformer'):
                _, loss = model(x, y)
            else:
                logits, _ = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    y.reshape(-1)
                )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        train_loss = total_loss / batch_count
        train_losses.append(train_loss)
        
        model.eval()
        total_val = 0
        batch_count = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                if hasattr(model, 'transformer'):
                    _, loss = model(x, y)
                else:
                    logits, _ = model(x)
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, model.vocab_size),
                        y.reshape(-1)
                    )
                
                total_val += loss.item()
                batch_count += 1
        
        val_loss = total_val / batch_count
        val_losses.append(val_loss)
        
        print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    return {
        'model': model,
        'name': model_name,
        'params': num_params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_val': val_loss
    }

def main():
    print("="*70)
    print("TRAINING ALL MODELS")
    print("="*70)
    
    with open('data/train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open('data/tokenizer.pkl', 'rb') as f:
        tokenizer_data = pickle.load(f)
    
    vocab_size = tokenizer_data['vocab_size']
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Training sequences: {len(train_data)}")
    print(f"Validation sequences: {len(val_data)}")
    
    train_dataset = SimpleDataset(train_data, block_size=64)
    val_dataset = SimpleDataset(val_data, block_size=64)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    os.makedirs('checkpoints', exist_ok=True)
    
    configs = [
        ('transformer', 'tiny'),
        ('transformer', 'small'),
        ('transformer', 'medium'),
        ('lstm', 'tiny'),
        ('lstm', 'small'),
        ('lstm', 'medium'),
    ]
    
    all_results = []
    
    for model_type, size in configs:
        model_name = f"{model_type}_{size}"
        
        if model_type == 'transformer':
            model = create_transformer(vocab_size, size)
        else:
            model = create_rnn(vocab_size, size)
        
        result = train_model(model, model_name, train_loader, val_loader, epochs=2)
        all_results.append(result)
        
        torch.save({
            'state_dict': model.state_dict(),
            'vocab_size': vocab_size,
            'config': {'type': model_type, 'size': size}
        }, f'checkpoints/{model_name}.pt')
        
        print(f"  Saved checkpoint to checkpoints/{model_name}.pt")
    
    with open('results/scaling_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    for r in all_results:
        print(f"{r['name']:20} {r['params']:8,} params  Val Loss: {r['final_val']:.4f}")

if __name__ == "__main__":
    main()