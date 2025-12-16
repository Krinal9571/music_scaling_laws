import os
import random

class MusicGenerator:
    def __init__(self):
        self.notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'c', 'd', 'e', 'f', 'g', 'a', 'b']
        self.octaves = ['', "'", "''", ',', ',,']
        self.durations = ['', '2', '3', '4', '6', '8', '/2', '/4']
        self.keys = ['C', 'G', 'D', 'A', 'F', 'Bb', 'Eb', 'Am', 'Em', 'Dm', 'Bm']
        self.meters = ['4/4', '3/4', '6/8', '2/4', '3/8', '9/8']
        
    def generate_abc(self, idx):
        key = random.choice(self.keys)
        meter = random.choice(self.meters)
        length = random.choice(['1/8', '1/16', '1/4'])
        
        abc = f"X:{idx}\n"
        abc += f"T:Synthetic Tune {idx}\n"
        abc += f"M:{meter}\n"
        abc += f"L:{length}\n"
        abc += f"K:{key}\n"
        
        num_measures = random.randint(8, 16)
        for m in range(num_measures):
            notes = []
            notes_per_measure = random.randint(4, 12)
            
            for _ in range(notes_per_measure):
                if random.random() < 0.9:
                    note = random.choice(self.notes)
                    if random.random() < 0.3:
                        note += random.choice(self.octaves)
                    if random.random() < 0.4:
                        note += random.choice(self.durations)
                    notes.append(note)
                else:
                    notes.append('z' + random.choice(['', '2', '4']))
            
            abc += ' '.join(notes)
            if m < num_measures - 1:
                if random.random() < 0.8:
                    abc += ' |'
                else:
                    abc += ' ||'
            else:
                abc += ' |]'
            abc += '\n'
        
        return abc

def main():

    print("GENERATING SYNTHETIC ABC DATASET")
    
    os.makedirs('data/abc_files', exist_ok=True)
    
    generator = MusicGenerator()
    num_files = 500
    
    print(f"\nGenerating {num_files} ABC files...")
    
    for i in range(num_files):
        abc_content = generator.generate_abc(i + 1)
        
        with open(f'data/abc_files/tune_{i:04d}.abc', 'w') as f:
            f.write(abc_content)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_files} files...")
    
    print(f"\n Generated {num_files} ABC files in data/abc_files/")
    print("\nDataset statistics:")
    
    total_chars = 0
    for i in range(num_files):
        with open(f'data/abc_files/tune_{i:04d}.abc', 'r') as f:
            total_chars += len(f.read())
    
    print(f"  Total characters: {total_chars:,}")
    print(f"  Average per file: {total_chars // num_files}")
    print("\nDataset creation complete!")

if __name__ == "__main__":
    main()