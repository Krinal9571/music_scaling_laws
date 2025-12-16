import os
import pickle
import random
import numpy as np

class SimpleTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        sorted_chars = ['<PAD>', '<UNK>'] + sorted(list(all_chars))
        self.char_to_idx = {ch: i for i, ch in enumerate(sorted_chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        print(f"  Vocabulary size: {self.vocab_size}")
        
    def encode(self, text):
        return [self.char_to_idx.get(ch, 1) for ch in text]
    
    def decode(self, ids):
        return ''.join([self.idx_to_char.get(i, '') for i in ids])
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls()
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = data['idx_to_char']
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer

def main():
    print("="*70)
    print("TOKENIZING ABC DATASET")
    print("="*70)
    
    abc_dir = 'data/abc_files'
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    all_tunes = []
    abc_files = sorted([f for f in os.listdir(abc_dir) if f.endswith('.abc')])
    
    print(f"\nLoading {len(abc_files)} ABC files...")
    
    for filename in abc_files:
        with open(os.path.join(abc_dir, filename), 'r') as f:
            content = f.read()
            all_tunes.append(content)
    
    print(f"✓ Loaded {len(all_tunes)} ABC files")
    
    print("\nBuilding vocabulary...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_tunes)
    
    tokenizer.save(os.path.join(output_dir, 'tokenizer.pkl'))
    print(f"✓ Saved tokenizer to {output_dir}/tokenizer.pkl")
    
    print("\nEncoding sequences...")
    encoded_tunes = [tokenizer.encode(tune) for tune in all_tunes]
    
    valid_tunes = [t for t in encoded_tunes if 30 < len(t) < 500]
    
    print(f"✓ Valid sequences: {len(valid_tunes)}")
    
    random.seed(42)
    random.shuffle(valid_tunes)
    
    split1 = int(0.80 * len(valid_tunes))
    split2 = int(0.90 * len(valid_tunes))
    
    train_data = valid_tunes[:split1]
    val_data = valid_tunes[split1:split2]
    test_data = valid_tunes[split2:]
    
    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    
    with open(os.path.join(output_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"\nDataset splits created:")
    print(f"  Train: {len(train_data)} sequences")
    print(f"  Validation: {len(val_data)} sequences")
    print(f"  Test: {len(test_data)} sequences")
    
    total_tokens = sum(len(seq) for seq in train_data + val_data + test_data)
    train_tokens = sum(len(seq) for seq in train_data)
    
    print(f"\nToken statistics:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Training tokens: {train_tokens:,}")
    print(f"  Tokens per sequence (avg): {total_tokens // len(valid_tunes)}")
    
    print("\n✓ Tokenization complete!")

if __name__ == "__main__":
    main()