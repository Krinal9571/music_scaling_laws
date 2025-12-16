
"""
Trains 5 Transformers + 4 LSTMs and generates complete analysis
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")

def run_command(command, step_name):
    print(f"\n Running: {step_name}...")
    result = subprocess.run(command, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"\n Error in step: {step_name}")
        sys.exit(1)
    print(f" {step_name} completed successfully")
    return True

def main():
    print_header("MUSIC SCALING LAWS PROJECT - COMPLETE EXECUTION")
    
    print("Project Overview:")
    print("  • 5 Transformer models (tiny → xlarge)")
    print("  • 4 LSTM models (tiny → large)")
    print("  • Synthetic ABC dataset (500 files)")
    print("  • Character-level tokenization")
    print("  • 1 epoch training for fair comparison")
    print("  • Power law scaling analysis")
    print("  • Music sample generation")
    print("\nEstimated time: 20-30 minutes on Mac CPU\n")
    
    input("Press Enter to start...")
    
    steps = [
        {
            'name': 'Installing Dependencies',
            'command': f'{sys.executable} -m pip install -q torch numpy matplotlib scipy pandas'
        },
        {
            'name': 'Step 1/5: Generating Synthetic ABC Dataset (500 files)',
            'command': f'{sys.executable} data_preprocessing/download_small_dataset.py'
        },
        {
            'name': 'Step 2/5: Tokenizing Dataset (character-level)',
            'command': f'{sys.executable} data_preprocessing/tokenize_small.py'
        },
        {
            'name': 'Step 3/5: Training ALL Models (5 Transformers + 4 LSTMs)',
            'command': f'{sys.executable} training/train_cpu.py'
        },
        {
            'name': 'Step 4/5: Analyzing Scaling Laws & Creating Plots',
            'command': f'{sys.executable} scripts/analyze_small.py'
        },
        {
            'name': 'Step 5/5: Generating Music Samples (10 per model)',
            'command': f'{sys.executable} evaluation/generate_small.py'
        }
    ]
    
    for step in steps:
        run_command(step['command'], step['name'])
    
    print("\n MODELS TRAINED (9 total):")
    print("  Transformers:")
    print("    • transformer_tiny (~15K params)")
    print("    • transformer_small (~80K params)")
    print("    • transformer_medium (~250K params)")
    print("    • transformer_large (~540K params)")
    print("    • transformer_xlarge (~940K params)")
    print("  LSTMs:")
    print("    • lstm_tiny (~15K params)")
    print("    • lstm_small (~70K params)")
    print("    • lstm_medium (~200K params)")
    print("    • lstm_large (~410K params)")
    
    print("\n GENERATED FILES:")
    print("   checkpoints/ - 9 trained model files (.pt)")
    print("   results/")
    print("    • scaling_laws.png - Power law fits")
    print("    • architecture_comparison.png - Transformer vs LSTM")
    print("    • training_curves.png - Loss curves")
    print("    • experiment_results.json - Complete data")
    print("   generated/ - 90 music samples (.abc files)")
    print("   data/ - Dataset and tokenizer")
    
    print("\n1. VIEW RESULTS:")
    print("   Open results/scaling_laws.png")
    print("   Open results/architecture_comparison.png")
    print("   Open results/training_curves.png")
     
    print("\n3. CHECK SAMPLES:")
    print("   Review generated/*.abc files")
    print("   Try online ABC player: https://abcjs.net/abcjs-editor.html")   

if __name__ == "__main__":
    main()