import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import json
from datetime import datetime

def power_law(N, a, alpha, c):
    return a * (N ** (-alpha)) + c

def fit_and_plot_scaling(results, model_type, ax, color):
    params = np.array([r['params'] for r in results])
    losses = np.array([r['final_val'] for r in results])
    
    ax.scatter(params, losses, s=150, c=color, alpha=0.7, edgecolors='black', linewidth=1.5, 
               label=f'{model_type.upper()} data', zorder=3)
    
    try:
        popt, pcov = curve_fit(power_law, params, losses, 
                              p0=[10.0, 0.1, min(losses)], 
                              maxfev=10000,
                              bounds=([0, 0, 0], [1000, 1, 10]))
        
        param_range = np.logspace(np.log10(params.min() * 0.8), 
                                 np.log10(params.max() * 1.2), 200)
        loss_fit = power_law(param_range, *popt)
        
        ax.plot(param_range, loss_fit, '--', linewidth=2.5, alpha=0.8, color=color,
                label=f'Fit: L={popt[0]:.2f}N^(-{popt[1]:.3f})+{popt[2]:.2f}')
        
        perr = np.sqrt(np.diag(pcov))
        
        print(f"\n{model_type.upper()} Scaling Law:")
        print(f"  L(N) = {popt[0]:.3f} × N^(-{popt[1]:.4f}) + {popt[2]:.3f}")
        print(f"  Scaling exponent α = {popt[1]:.4f} ± {perr[1]:.4f}")
        print(f"  Coefficient a = {popt[0]:.3f} ± {perr[0]:.3f}")
        print(f"  Constant c = {popt[2]:.3f} ± {perr[2]:.3f}")
        
        r_squared = 1 - (np.sum((losses - power_law(params, *popt))**2) / 
                        np.sum((losses - losses.mean())**2))
        print(f"  R² = {r_squared:.4f}")
        
        return popt, pcov
        
    except Exception as e:
        print(f"\n⚠ Could not fit power law for {model_type}: {e}")
        return None, None

def main():
    print("="*70)
    print("ANALYZING SCALING RESULTS")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    with open('results/scaling_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    
    trans_results = [r for r in all_results if r['type'] == 'transformer']
    lstm_results = [r for r in all_results if r['type'] == 'lstm']
    
    print(f"\nLoaded results:")
    print(f"  Transformer models: {len(trans_results)}")
    print(f"  LSTM models: {len(lstm_results)}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    trans_popt, trans_pcov = None, None
    lstm_popt, lstm_pcov = None, None
    
    if trans_results:
        trans_popt, trans_pcov = fit_and_plot_scaling(trans_results, 'transformer', 
                                                       axes[0], '#2E86AB')
        
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Number of Parameters', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
        axes[0].set_title('Transformer Scaling Law', fontsize=15, fontweight='bold', pad=15)
        axes[0].legend(fontsize=10, framealpha=0.9)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        for r in trans_results:
            axes[0].annotate(r['size'], (r['params'], r['final_val']),
                           xytext=(8, 8), textcoords='offset points', 
                           fontsize=9, alpha=0.8, fontweight='bold')
    
    if lstm_results:
        lstm_popt, lstm_pcov = fit_and_plot_scaling(lstm_results, 'lstm', 
                                                     axes[1], '#E63946')
        
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Number of Parameters', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
        axes[1].set_title('LSTM Scaling Law', fontsize=15, fontweight='bold', pad=15)
        axes[1].legend(fontsize=10, framealpha=0.9)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        for r in lstm_results:
            axes[1].annotate(r['size'], (r['params'], r['final_val']),
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=9, alpha=0.8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/scaling_laws.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: results/scaling_laws.png")
    
    fig2, ax2 = plt.subplots(figsize=(11, 7))
    
    if trans_results:
        params = np.array([r['params'] for r in trans_results])
        losses = np.array([r['final_val'] for r in trans_results])
        ax2.scatter(params, losses, s=200, c='#2E86AB', alpha=0.7, 
                   edgecolors='black', linewidth=2, label='Transformer', zorder=3)
        
        if trans_popt is not None:
            param_range = np.logspace(np.log10(params.min() * 0.8), 
                                     np.log10(params.max() * 1.2), 200)
            loss_fit = power_law(param_range, *trans_popt)
            ax2.plot(param_range, loss_fit, '--', linewidth=2.5, alpha=0.8, 
                    color='#2E86AB', label='Transformer fit')
    
    if lstm_results:
        params = np.array([r['params'] for r in lstm_results])
        losses = np.array([r['final_val'] for r in lstm_results])
        ax2.scatter(params, losses, s=200, c='#E63946', alpha=0.7,
                   edgecolors='black', linewidth=2, label='LSTM', zorder=3)
        
        if lstm_popt is not None:
            param_range = np.logspace(np.log10(params.min() * 0.8),
                                     np.log10(params.max() * 1.2), 200)
            loss_fit = power_law(param_range, *lstm_popt)
            ax2.plot(param_range, loss_fit, '--', linewidth=2.5, alpha=0.8,
                    color='#E63946', label='LSTM fit')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Parameters (log scale)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_title('Architecture Comparison: Transformer vs LSTM', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, framealpha=0.9, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/architecture_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/architecture_comparison.png")
    
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    colors_trans = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087']
    colors_lstm = ['#f95d6a', '#ff7c43', '#ffa600', '#ffbe0b']
    
    for idx, r in enumerate(trans_results):
        ax3.plot(r['train_losses'], label=f"Transformer-{r['size']} ({r['params']:,})",
                linewidth=2.5, color=colors_trans[idx % len(colors_trans)])
    
    for idx, r in enumerate(lstm_results):
        ax3.plot(r['train_losses'], label=f"LSTM-{r['size']} ({r['params']:,})",
                linewidth=2.5, linestyle='--', color=colors_lstm[idx % len(colors_lstm)])
    
    ax3.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax3.set_title('Training Curves: All Models', fontsize=16, fontweight='bold', pad=20)
    ax3.legend(fontsize=10, framealpha=0.9, ncol=2)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/training_curves.png")
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(all_results),
        'transformer_models': len(trans_results),
        'lstm_models': len(lstm_results),
        'models': []
    }
    
    if trans_popt is not None:
        report_data['transformer_scaling_law'] = {
            'coefficient_a': float(trans_popt[0]),
            'exponent_alpha': float(trans_popt[1]),
            'constant_c': float(trans_popt[2])
        }
    
    if lstm_popt is not None:
        report_data['lstm_scaling_law'] = {
            'coefficient_a': float(lstm_popt[0]),
            'exponent_alpha': float(lstm_popt[1]),
            'constant_c': float(lstm_popt[2])
        }
    
    for r in all_results:
        report_data['models'].append({
            'name': r['name'],
            'type': r['type'],
            'size': r['size'],
            'parameters': r['params'],
            'final_validation_loss': r['final_val'],
            'training_losses': r['train_losses'],
            'validation_losses': r['val_losses'],
            'training_time_seconds': r['epoch_times'][0] if r['epoch_times'] else 0
        })
    
    with open('results/experiment_results.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print("✓ Saved: results/experiment_results.json")
    
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)
    
    if trans_popt is not None and lstm_popt is not None:
        print(f"\nScaling Exponent Comparison:")
        print(f"  Transformer α = {trans_popt[1]:.4f}")
        print(f"  LSTM α = {lstm_popt[1]:.4f}")
        print(f"  Difference = {abs(trans_popt[1] - lstm_popt[1]):.4f}")
        
        if trans_popt[1] > lstm_popt[1]:
            ratio = trans_popt[1] / lstm_popt[1]
            print(f"\n→ Transformers scale {ratio:.2f}x better than LSTMs")
        else:
            ratio = lstm_popt[1] / trans_popt[1]
            print(f"\n→ LSTMs scale {ratio:.2f}x better than Transformers")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()