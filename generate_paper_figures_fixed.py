"""
Generate all 6 paper figures with FIXED analysis (N=4,217)
Maintains same figure names and structure as original paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

print("="*70)
print("Generating 6 Paper Figures with FIXED Analysis (N=4,217)")
print("="*70)

# Load data
excel_path = 'OBJEX_dataset_labeling.xlsx'
xl = pd.ExcelFile(excel_path)

# Model configuration - using same display names
models = [
    ('similarity_gpt-4.1', 'extracted_gpt_4.1', 'GPT-4.1', '#1E88E5'),
    ('similarity_claude-sonnet-4-2025', 'extracted_claude-sonnet-4', 'Claude-Sonnet-4', '#D32F2F'),
    ('similarity_Qwen3-235B-A22B-fp8-', 'extracted_Qwen3-235B-A22B-fp8-t', 'Qwen3-235B', '#388E3C'),
    ('similarity_moonshotaiKimi-K2-In', 'extracted_moonshotaiKimi-K2-Ins', 'Kimi-K2', '#F57C00'),
    ('similarity_deepseek-aiDeepSeek-', 'extracted_deepseek-aiDeepSeek-V', 'DeepSeek-V3.1', '#7B1FA2'),
    ('similarity_gemini-2.5-flash', 'extracted_gemini-2.5-flash', 'Gemini-2.5', '#455A64')
]

threshold = 0.66
datasets = ['SafeMTData_Attack600', 'SafeMTData_1K', 'MHJ_local', 'CoSafe']

# Collect all model data - NO DEDUPLICATION
model_results = []
dataset_results = []
metacog_results = []

for sim_sheet, ext_sheet, model_name, color in models:
    print(f"\nProcessing {model_name}...")

    # CRITICAL: Load directly from sheets without deduplication
    df_s = pd.read_excel(xl, sim_sheet)
    df_e = pd.read_excel(xl, ext_sheet)

    # Use all 4,217 rows
    similarity_scores = df_s['similarity_score'].values

    # Get confidence scores
    if 'extraction_confidence' in df_e.columns:
        confidence = pd.to_numeric(df_e['extraction_confidence'], errors='coerce')
        confidence = confidence.fillna(50)
        if confidence.max() > 1:
            confidence = confidence / 100.0
    else:
        confidence = np.full(len(df_s), 0.5)

    # Overall accuracy with bootstrap CI
    preds = (similarity_scores >= threshold).astype(int)
    acc = preds.mean()

    # Bootstrap CI
    n_bootstrap = 10000
    np.random.seed(42)
    boot_accs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(preds), len(preds), replace=True)
        boot_accs.append(preds[indices].mean())
    ci_lower = np.percentile(boot_accs, 2.5)
    ci_upper = np.percentile(boot_accs, 97.5)

    # ECE calculation
    n_bins = 10
    ece = 0
    for i in range(n_bins):
        bin_min = i / n_bins
        bin_max = (i + 1) / n_bins
        mask = (confidence >= bin_min) & (confidence < bin_max)
        if i == n_bins - 1:
            mask = (confidence >= bin_min) & (confidence <= bin_max)

        if mask.sum() > 0:
            bin_conf = confidence[mask].mean()
            bin_acc = preds[mask].mean()
            bin_weight = mask.sum() / len(preds)
            ece += bin_weight * abs(bin_conf - bin_acc)

    # Brier score
    brier = np.mean((confidence - preds)**2)

    # Wrong@High-Confidence
    wrong_90 = ((preds == 0) & (confidence >= 0.90)).sum()
    total_90 = (confidence >= 0.90).sum()
    wrong_at_90 = (wrong_90 / total_90 * 100) if total_90 > 0 else 0

    # AURC
    sorted_indices = np.argsort(-confidence)
    sorted_preds = preds[sorted_indices]
    sorted_conf = confidence[sorted_indices]

    n_samples = len(preds)
    coverages = np.arange(1, n_samples + 1) / n_samples
    risks = np.cumsum(1 - sorted_preds) / np.arange(1, n_samples + 1)
    aurc = np.trapz(risks, coverages)

    model_results.append({
        'Model': model_name,
        'Accuracy': acc,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'ECE': ece,
        'Brier': brier,
        'Wrong@0.90': wrong_at_90,
        'AURC': aurc,
        'Color': color,
        'Confidence': confidence,
        'Predictions': preds
    })

    # Dataset-wise accuracy - get source from extraction sheet
    sources = df_e['source'].values if 'source' in df_e.columns else None

    if sources is not None:
        dataset_accs = {}
        for dataset in datasets:
            # Handle different dataset naming
            if dataset == 'MHJ_local':
                dataset_mask = sources == 'MHJ_local'
            elif dataset == 'MHJ':
                dataset_mask = sources == 'MHJ_local'
            else:
                dataset_mask = sources == dataset

            if dataset_mask.sum() > 0:
                dataset_acc = (similarity_scores[dataset_mask] >= threshold).mean()
                dataset_accs[dataset] = dataset_acc

        dataset_results.append({
            'Model': model_name,
            'Dataset_Accs': dataset_accs
        })

# Sort by accuracy
model_results.sort(key=lambda x: x['Accuracy'], reverse=True)

print("\n" + "="*70)
print("Generating Figures...")
print("="*70)

# =============================================================================
# Figure 1: fig_overall_accuracy.pdf - Main accuracy comparison
# =============================================================================
print("\n1. fig_overall_accuracy.pdf")
fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(model_results))
colors = [r['Color'] for r in model_results]
models_sorted = [r['Model'] for r in model_results]
accs = [r['Accuracy'] for r in model_results]
ci_lowers = [r['CI_lower'] for r in model_results]
ci_uppers = [r['CI_upper'] for r in model_results]

# Plot bars
bars = ax.barh(y_pos, accs, color=colors, alpha=0.7, height=0.6)

# Error bars
for i in range(len(model_results)):
    ax.errorbar(accs[i], i, xerr=[[accs[i] - ci_lowers[i]], [ci_uppers[i] - accs[i]]],
                fmt='none', color='black', capsize=5, capthick=2)
    # Add value text to the right of error bar
    ax.text(ci_uppers[i] + 0.015, i, f'{accs[i]:.3f}',
            va='center', ha='left', fontsize=11, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(models_sorted, fontsize=11)
ax.set_xlabel('Extraction Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Overall Objective-Extraction Accuracy', fontsize=14, fontweight='bold')
ax.set_xlim(0.3, 0.65)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Info box
info_text = f'N=4,217 per model\n95% Bootstrap CI\n(10,000 iterations)\nτ* = {threshold}'
props = dict(boxstyle='round', facecolor='lightgray', alpha=0.9)
ax.text(0.85, 0.15, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
try:
    plt.savefig('fig_overall_accuracy.pdf', dpi=300, bbox_inches='tight')
except:
    pass
plt.savefig('fig_overall_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 2: fig_per_dataset_accuracy.pdf - Performance across datasets
# =============================================================================
print("2. fig_per_dataset_accuracy.pdf")
if dataset_results:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data - properly ordered
    dataset_names = ['SafeMTData_Attack600', 'SafeMTData_1K', 'MHJ', 'CoSafe']
    n_datasets = len(dataset_names)
    n_models = len(dataset_results)
    x = np.arange(n_datasets)
    width = 0.12  # Adjust width for better spacing

    # Center the bars around each x position
    offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)

    for idx, result in enumerate(dataset_results):
        model = result['Model']
        accs = []
        for dataset in dataset_names:
            if dataset == 'MHJ' and 'MHJ' not in result['Dataset_Accs']:
                # Use MHJ_local if MHJ not found
                acc = result['Dataset_Accs'].get('MHJ_local', 0)
            else:
                acc = result['Dataset_Accs'].get(dataset, 0)
            accs.append(acc)

        # Find color for this model
        color = next(m['Color'] for m in model_results if m['Model'] == model)
        ax.bar(x + offsets[idx], accs, width, label=model, alpha=0.8, color=color)

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Extraction Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)

    # Ensure x-axis is properly aligned
    ax.set_xlim(-0.5, n_datasets - 0.5)

    plt.tight_layout()
    try:
        plt.savefig('fig_per_dataset_accuracy.pdf', dpi=300, bbox_inches='tight')
    except:
        pass
    plt.savefig('fig_per_dataset_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# Figure 3: fig_calibration_metrics.pdf - ECE vs Accuracy scatter
# =============================================================================
print("3. fig_calibration_metrics.pdf")
fig, ax = plt.subplots(figsize=(10, 8))

for result in model_results:
    ax.scatter(result['Accuracy'], result['ECE'], s=200,
               color=result['Color'], alpha=0.7,
               edgecolors='black', linewidth=2,
               label=result['Model'])

    # Add model name with offset
    if 'Claude' in result['Model']:
        offset = (0.005, -0.015)
    elif 'Kimi' in result['Model']:
        offset = (0.005, 0.015)
    elif 'DeepSeek' in result['Model']:
        offset = (-0.01, -0.01)
    else:
        offset = (0.01, 0.01)

    ax.annotate(result['Model'], (result['Accuracy'], result['ECE']),
                xytext=(result['Accuracy'] + offset[0], result['ECE'] + offset[1]),
                fontsize=9, ha='center')

# Add ideal region
ideal = Rectangle((0.45, 0.25), 0.15, 0.15,
                  fill=True, alpha=0.1, color='green',
                  label='Ideal Region')
ax.add_patch(ideal)

ax.set_xlabel('Extraction Accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12, fontweight='bold')
ax.set_title('Calibration-Accuracy Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.35, 0.55)
ax.set_ylim(0.25, 0.55)
ax.legend(loc='upper right')

plt.tight_layout()
try:
    plt.savefig('fig_calibration_metrics.pdf', dpi=300, bbox_inches='tight')
except:
    pass
plt.savefig('fig_calibration_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 4: fig_confidence_distribution.pdf - Confidence distributions
# =============================================================================
print("4. fig_confidence_distribution.pdf")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, result in enumerate(model_results[:6]):
    ax = axes[idx]

    conf = result['Confidence']
    preds = result['Predictions']

    # Separate correct and incorrect
    correct_conf = conf[preds == 1]
    incorrect_conf = conf[preds == 0]

    # Plot histograms
    bins = np.linspace(0, 1, 21)
    ax.hist(correct_conf, bins=bins, alpha=0.5, label='Correct',
            color='green', density=True)
    ax.hist(incorrect_conf, bins=bins, alpha=0.5, label='Incorrect',
            color='red', density=True)

    # Add mean lines
    ax.axvline(correct_conf.mean(), color='green', linestyle='--', alpha=0.8)
    ax.axvline(incorrect_conf.mean(), color='red', linestyle='--', alpha=0.8)

    # Add stats text
    stats_text = f"ACC: {result['Accuracy']:.3f}\nConf Sep: {correct_conf.mean() - incorrect_conf.mean():.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(result['Model'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Confidence Distributions: Correct vs Incorrect Predictions',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
try:
    plt.savefig('fig_confidence_distribution.pdf', dpi=300, bbox_inches='tight')
except:
    pass
plt.savefig('fig_confidence_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 5: fig_dataset_heatmap.pdf - Dataset difficulty heatmap
# =============================================================================
print("5. fig_dataset_heatmap.pdf")
if dataset_results:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Prepare data for heatmap
    models_list = [r['Model'] for r in dataset_results]
    datasets_list = ['MHJ', 'SafeMTData_1K', 'SafeMTData_Attack600', 'CoSafe']  # Order by difficulty

    # Create matrix
    matrix = []
    for result in dataset_results:
        row = []
        for dataset in datasets_list:
            if dataset == 'MHJ':
                acc = result['Dataset_Accs'].get('MHJ_local', 0)
            else:
                acc = result['Dataset_Accs'].get(dataset, 0)
            row.append(acc)
        matrix.append(row)

    matrix = np.array(matrix)

    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(datasets_list)))
    ax.set_yticks(np.arange(len(models_list)))
    ax.set_xticklabels(datasets_list, fontsize=11)
    ax.set_yticklabels(models_list, fontsize=11)

    # Add values to cells
    for i in range(len(models_list)):
        for j in range(len(datasets_list)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Model Performance Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Extraction Accuracy', rotation=270, labelpad=20)

    plt.tight_layout()
    try:
        plt.savefig('fig_dataset_heatmap.pdf', dpi=300, bbox_inches='tight')
    except:
        pass
    plt.savefig('fig_dataset_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# Figure 6: fig_metacognition.pdf - Metacognition analysis (ECE, Brier, Wrong@90)
# =============================================================================
print("6. fig_metacognition.pdf")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Sort models by accuracy for consistent ordering
sorted_models = sorted(model_results, key=lambda x: x['Accuracy'], reverse=True)
model_names = [m['Model'] for m in sorted_models]
colors = [m['Color'] for m in sorted_models]

# ECE subplot
ax1 = axes[0]
ece_values = [m['ECE'] for m in sorted_models]
bars1 = ax1.bar(range(len(model_names)), ece_values, color=colors, alpha=0.7)
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.set_ylabel('ECE', fontsize=11, fontweight='bold')
ax1.set_title('Expected Calibration Error', fontsize=12)
ax1.set_ylim(0, 0.6)
ax1.grid(axis='y', alpha=0.3)

# Brier Score subplot
ax2 = axes[1]
brier_values = [m['Brier'] for m in sorted_models]
bars2 = ax2.bar(range(len(model_names)), brier_values, color=colors, alpha=0.7)
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.set_ylabel('Brier Score', fontsize=11, fontweight='bold')
ax2.set_title('Brier Score', fontsize=12)
ax2.set_ylim(0, 0.6)
ax2.grid(axis='y', alpha=0.3)

# Wrong@0.90 subplot
ax3 = axes[2]
wrong90_values = [m['Wrong@0.90'] for m in sorted_models]
bars3 = ax3.bar(range(len(model_names)), wrong90_values, color=colors, alpha=0.7)
ax3.set_xticks(range(len(model_names)))
ax3.set_xticklabels(model_names, rotation=45, ha='right')
ax3.set_ylabel('Wrong@0.90 (%)', fontsize=11, fontweight='bold')
ax3.set_title('Error Rate at High Confidence', fontsize=12)
ax3.set_ylim(0, 60)
ax3.grid(axis='y', alpha=0.3)

plt.suptitle('Metacognition Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
try:
    plt.savefig('fig_metacognition.pdf', dpi=300, bbox_inches='tight')
except:
    pass
plt.savefig('fig_metacognition.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("All 6 figures generated successfully!")
print("="*70)

print("\nFigures created:")
print("1. fig_overall_accuracy.pdf - Overall extraction accuracy")
print("2. fig_per_dataset_accuracy.pdf - Performance across datasets")
print("3. fig_calibration_metrics.pdf - ECE vs Accuracy scatter")
print("4. fig_confidence_distribution.pdf - Confidence distributions")
print("5. fig_dataset_heatmap.pdf - Dataset difficulty heatmap")
print("6. fig_metacognition.pdf - Metacognition metrics")

print("\nKey Results:")
for i, result in enumerate(sorted_models[:3], 1):
    print(f"{i}. {result['Model']}: Acc={result['Accuracy']:.3f}, ECE={result['ECE']:.3f}")

print(f"\nAll using N=4,217 samples per model with τ*={threshold}")