import os
import sys
import pickle
import torch
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.tiny_transformer import create_transformer
from models.tiny_rnn import create_rnn
from data_preprocessing.tokenize_small import SimpleTokenizer

def generate_samples():
    print("GENERATING MUSIC SAMPLES FROM ALL MODELS")
    
    tokenizer = SimpleTokenizer.load('data/tokenizer.pkl')
    
    print(f"\nTokenizer loaded:")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    
    os.makedirs('generated', exist_ok=True)
    
    checkpoint_files = [
        'transformer_tiny.pt',
        'transformer_small.pt',
        'transformer_medium.pt',
        'transformer_large.pt',
        'transformer_xlarge.pt',
        'lstm_tiny.pt',
        'lstm_small.pt',
        'lstm_medium.pt',
        'lstm_large.pt',
    ]
    
    all_samples = {}
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = f'checkpoints/{checkpoint_file}'
        
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠ Skipping {checkpoint_file} - not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Generating from: {checkpoint_file}")
        print(f"{'='*70}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        vocab_size = checkpoint['vocab_size']
        config = checkpoint['config']
        
        model_type = config['type']
        model_size = config['size']
        
        if model_type == 'transformer':
            model = create_transformer(vocab_size, model_size)
        else:
            model = create_rnn(vocab_size, model_size)
        
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        model_samples = []
        
        for i in range(10):
            start_seq = "X:1\nT:"
            start_ids = torch.tensor([tokenizer.encode(start_seq)], dtype=torch.long)
            
            generated = model.generate(start_ids, max_new=150, temp=0.8, top_k=50)
            text = tokenizer.decode(generated[0].tolist())
            
            output_file = f'generated/{model_type}_{model_size}_sample_{i+1}.abc'
            with open(output_file, 'w') as f:
                f.write(text)
            
            model_samples.append({
                'sample_id': i+1,
                'text': text,
                'length': len(text),
                'file': output_file
            })
            
            if (i + 1) % 3 == 0:
                print(f"  Generated {i+1}/10 samples")
        
        all_samples[f'{model_type}_{model_size}'] = model_samples
        print(f"✓ Completed {len(model_samples)} samples for {model_type}_{model_size}")
    
    with open('generated/all_samples_metadata.json', 'w') as f:
        json.dump(all_samples, f, indent=2)
    

    print("SAMPLE GENERATION COMPLETE")
   
    print(f"\n Generated samples from {len(all_samples)} models")
    print(f" Total samples: {sum(len(samples) for samples in all_samples.values())}")
    print(f" Samples saved to: generated/")
    print(f" Metadata saved to: generated/all_samples_metadata.json")
    
    print("QUALITY ANALYSIS")
 
    
    for model_name, samples in all_samples.items():
        valid_count = 0
        has_header_count = 0
        has_bars_count = 0
        
        for sample in samples:
            text = sample['text']
            
            if 'X:' in text and 'T:' in text:
                has_header_count += 1
            
            if 'X:' in text and 'T:' in text and 'M:' in text and 'K:' in text:
                valid_count += 1
            
            if '|' in text:
                has_bars_count += 1
        
        print(f"\n{model_name}:")
        print(f"  Valid headers: {has_header_count}/10 ({has_header_count*10}%)")
        print(f"  Complete ABC: {valid_count}/10 ({valid_count*10}%)")
        print(f"  Has bar lines: {has_bars_count}/10 ({has_bars_count*10}%)")
    

if __name__ == "__main__":
    generate_samples()