# Music Scaling Laws Project

Empirical study of scaling laws for language models trained on symbolic music data (ABC notation).

## Project Structure

```
MUSIC_SCALING_LAWS/
├── data_preprocessing/
│   ├── download_small_dataset.py    # Generate synthetic ABC dataset
│   └── tokenize_small.py            # Character-level tokenization
├── models/
│   ├── tiny_transformer.py          # Transformer implementation
│   └── tiny_rnn.py                  # LSTM implementation
├── training/
│   └── train_cpu.py                 # Training script for all models
├── evaluation/
│   └── generate_small.py            # Music sample generation
├── scripts/
│   └── analyze_small.py             # Scaling analysis and plots
├── data/                            # Generated datasets
├── checkpoints/                     # Trained model checkpoints
├── results/                         # Plots and analysis
├── generated/                       # Generated music samples
├── requirements.txt                 # Python dependencies
├── run_step_by_step.py             # Automated execution
└── README.md                        # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run all steps automatically:

```bash
python run_step_by_step.py
```

## Manual Execution

Step 1: Generate dataset

```bash
python data_preprocessing/download_small_dataset.py
```

Step 2: Tokenize

```bash
python data_preprocessing/tokenize_small.py
```

Step 3: Train models

```bash
python training/train_cpu.py
```

Step 4: Analyze results

```bash
python scripts/analyze_small.py
```

Step 5: Generate samples

```bash
python evaluation/generate_small.py
```

## Models Trained

### Transformers

- **Tiny**: 16-dim, 2 heads, 2 layers (~5K params)
- **Small**: 32-dim, 2 heads, 2 layers (~20K params)
- **Medium**: 48-dim, 4 heads, 3 layers (~60K params)

### LSTMs

- **Tiny**: 16-dim embed, 32-dim hidden, 1 layer (~3K params)
- **Small**: 32-dim embed, 64-dim hidden, 2 layers (~15K params)
- **Medium**: 48-dim embed, 96-dim hidden, 2 layers (~40K params)

## Results

All results are saved to the `results/` directory:

- `scaling_laws.png` - Transformer and LSTM scaling plots with power law fits
- `architecture_comparison.png` - Direct comparison of both architectures
- `training_curves.png` - Training loss curves for all models
- `experiment_results.json` - Complete experimental data

## Dataset

- **Type**: Synthetic ABC notation
- **Size**: 300 tunes
- **Train/Val/Test**: 80%/10%/10%
- **Tokenization**: Character-level
- **Vocabulary**: ~70 unique characters

## Key Findings

1. **Scaling Laws**: Both architectures follow power law: L = a·N^(-α) + c
2. **Transformer vs LSTM**: Transformers show better parameter efficiency
3. **Musical Structure**: Models learn basic ABC syntax and note patterns

## System Requirements

- **CPU**: Works on Mac/Linux/Windows
- **RAM**: 4GB minimum
- **Disk**: <200MB total
- **Runtime**: 15-30 minutes

## Citation

If you use this code, please cite:

- Kaplan et al. (2020): "Scaling Laws for Neural Language Models"

## License

MIT License
