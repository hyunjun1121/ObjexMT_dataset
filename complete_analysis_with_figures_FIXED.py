"""
Complete Analysis for OBJEX Dataset with Figure Generation - FIXED VERSION
NO DEDUPLICATION - Each jailbreak attempt is evaluated separately
Includes all figures for the paper
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPLETE OBJEX ANALYSIS WITH FIGURES - FIXED VERSION")
print("="*80)
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
excel_path = 'OBJEX_dataset_labeling.xlsx'
xl = pd.ExcelFile(excel_path)

# Load labeling data
df_label = pd.read_excel(xl, 'Labeling')
print(f"\nTotal labeled samples: {len(df_label)}")

# ============================================================================
# PART 1: OPTIMAL THRESHOLD CALCULATION
# ============================================================================
print("\n" + "="*80)
print("PART 1: OPTIMAL THRESHOLD CALCULATION")
print("="*80)

# Convert human labels to binary
label_mapping = {
    'Exact match': 1,
    'High similarity': 1,
    'Moderate similarity': 0,
    'Low similarity': 0
}
df_label['human_binary'] = df_label['human_label'].map(label_mapping)

# Calculate F1 scores for different thresholds
thresholds = np.arange(0.0, 1.01, 0.01)
best_f1 = 0
best_threshold = 0
f1_scores = []

for threshold in thresholds:
    predictions = (df_label['similarity_score'] >= threshold).astype(int)

    tp = ((predictions == 1) & (df_label['human_binary'] == 1)).sum()
    fp = ((predictions == 1) & (df_label['human_binary'] == 0)).sum()
    fn = ((predictions == 0) & (df_label['human_binary'] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    f1_scores.append(f1)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Optimal threshold: τ* = {best_threshold:.2f}")
print(f"F1 score at optimal threshold: {best_f1:.3f}")

# ============================================================================
# PART 2: MODEL EVALUATION WITH CORRECT METHOD
# ============================================================================
print("\n" + "="*80)
print("PART 2: MODEL EVALUATION (N=4,217 per model)")
print("="*80)

# Model configurations with colors
models = [
    ('similarity_gpt-4.1', 'extracted_gpt_4.1', 'gpt-4.1', '#1E88E5'),
    ('similarity_claude-sonnet-4-2025', 'extracted_claude-sonnet-4', 'claude-sonnet-4', '#D32F2F'),
    ('similarity_Qwen3-235B-A22B-fp8-', 'extracted_Qwen3-235B-A22B-fp8-t', 'Qwen3-235B-A22B-FP8', '#388E3C'),
    ('similarity_moonshotaiKimi-K2-In', 'extracted_moonshotaiKimi-K2-Ins', 'kimi-k2', '#F57C00'),
    ('similarity_deepseek-aiDeepSeek-', 'extracted_deepseek-aiDeepSeek-V', 'deepseek-v3.1', '#7B1FA2'),
    ('similarity_gemini-2.5-flash', 'extracted_gemini-2.5-flash', 'gemini-2.5-flash', '#455A64')
]

# Store all results
all_results = {}
model_data = {}
model_colors = {}

for sim_sheet, ext_sheet, model_name, color in models:
    print(f"\n{model_name}:")
    print("-" * 40)

    model_colors[model_name] = color

    # CRITICAL FIX: Do NOT merge by base_prompt
    # Load directly from similarity sheet which has all evaluations
    df_sim = pd.read_excel(xl, sim_sheet)
    df_ext = pd.read_excel(xl, ext_sheet)

    # Use similarity sheet as the main source (has all 4,217 rows)
    # Get confidence from extraction sheet by row index
    if 'extraction_confidence' in df_ext.columns:
        confidence = pd.to_numeric(df_ext['extraction_confidence'], errors='coerce')
        confidence = confidence.fillna(50)
        if confidence.max() > 1:
            confidence = confidence / 100.0
    else:
        confidence = pd.Series([0.5] * len(df_sim))

    # Ensure we have the right length
    if len(confidence) != len(df_sim):
        print(f"  WARNING: Length mismatch. Using default confidence.")
        confidence = pd.Series([0.5] * len(df_sim))

    # Store for cross-model analysis
    model_data[model_name] = {
        'similarity_scores': df_sim['similarity_score'].values,
        'confidence': confidence.values,
        'source': df_sim['source'].values if 'source' in df_sim.columns else None
    }

    # Calculate accuracy
    predictions = (df_sim['similarity_score'] >= best_threshold).astype(int)
    accuracy = predictions.mean()

    # Bootstrap for confidence interval (10,000 iterations)
    n_bootstrap = 10000
    accuracies = []
    np.random.seed(42)

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(predictions), len(predictions), replace=True)
        boot_acc = predictions.iloc[indices].mean()
        accuracies.append(boot_acc)

    ci_lower = np.percentile(accuracies, 2.5)
    ci_upper = np.percentile(accuracies, 97.5)

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
            bin_acc = predictions[mask].mean()
            bin_weight = mask.sum() / len(predictions)
            ece += bin_weight * abs(bin_conf - bin_acc)

    # Brier score
    brier = np.mean((confidence - predictions)**2)

    # Wrong at high confidence
    wrong_80 = ((predictions == 0) & (confidence >= 0.80)).sum()
    total_80 = (confidence >= 0.80).sum()
    wrong_at_80 = (wrong_80 / total_80 * 100) if total_80 > 0 else 0

    wrong_90 = ((predictions == 0) & (confidence >= 0.90)).sum()
    total_90 = (confidence >= 0.90).sum()
    wrong_at_90 = (wrong_90 / total_90 * 100) if total_90 > 0 else 0

    wrong_95 = ((predictions == 0) & (confidence >= 0.95)).sum()
    total_95 = (confidence >= 0.95).sum()
    wrong_at_95 = (wrong_95 / total_95 * 100) if total_95 > 0 else 0

    # AURC calculation
    sorted_indices = np.argsort(-confidence)
    sorted_preds = predictions.iloc[sorted_indices]
    sorted_conf = confidence.iloc[sorted_indices]

    n_samples = len(predictions)
    coverages = np.arange(1, n_samples + 1) / n_samples
    risks = np.cumsum(1 - sorted_preds) / np.arange(1, n_samples + 1)
    aurc = np.trapz(risks, coverages)

    # Store results
    results = {
        'accuracy': accuracy,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ece': ece,
        'brier': brier,
        'wrong_at_80': wrong_at_80,
        'wrong_at_90': wrong_at_90,
        'wrong_at_95': wrong_at_95,
        'aurc': aurc,
        'n_samples': len(df_sim),
        'n_high_conf_80': total_80,
        'n_high_conf_90': total_90,
        'n_high_conf_95': total_95,
        'predictions': predictions,
        'confidence': confidence
    }

    all_results[model_name] = results

    # Print results
    print(f"  N = {len(df_sim):,}")
    print(f"  Accuracy: {accuracy:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print(f"  ECE: {ece:.3f}")
    print(f"  Brier: {brier:.3f}")
    print(f"  Wrong@0.80: {wrong_at_80:.1f}% ({wrong_80}/{total_80})")
    print(f"  Wrong@0.90: {wrong_at_90:.1f}% ({wrong_90}/{total_90})")
    print(f"  Wrong@0.95: {wrong_at_95:.1f}% ({wrong_95}/{total_95})")
    print(f"  AURC: {aurc:.3f}")

# ============================================================================
# PART 3: DATASET-WISE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 3: DATASET-WISE ANALYSIS")
print("="*80)

datasets = ['CoSafe', 'SafeMTData_1K', 'SafeMTData_Attack600', 'MHJ_local']
dataset_results = {}

for model_name in all_results.keys():
    if model_data[model_name]['source'] is not None:
        print(f"\n{model_name}:")
        print("-" * 40)

        scores = model_data[model_name]['similarity_scores']
        sources = model_data[model_name]['source']

        model_dataset_acc = {}
        for dataset in datasets:
            mask = sources == dataset
            if mask.sum() > 0:
                dataset_acc = (scores[mask] >= best_threshold).mean()
                model_dataset_acc[dataset] = dataset_acc
                print(f"  {dataset:25s}: {dataset_acc:.3f} (N={mask.sum():,})")

        dataset_results[model_name] = model_dataset_acc

# ============================================================================
# PART 4: GENERATE FIGURES
# ============================================================================
print("\n" + "="*80)
print("PART 4: GENERATING FIGURES")
print("="*80)

# Create visualizations directory
import os
os.makedirs('visualizations', exist_ok=True)

# -----------------------------------------------------------------------------
# Figure 1: Overall Accuracy Bar Chart
# -----------------------------------------------------------------------------
print("\nGenerating: fig_overall_accuracy.pdf")
sorted_models = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

fig, ax = plt.subplots(figsize=(10, 6))
y_positions = range(len(sorted_models))

for i, (model_name, results) in enumerate(sorted_models):
    # Main bar
    ax.barh(i, results['accuracy'], color=model_colors[model_name], alpha=0.7, height=0.6)

    # Error bars for CI
    ax.errorbar(results['accuracy'], i,
                xerr=[[results['accuracy'] - results['ci_lower']],
                      [results['ci_upper'] - results['accuracy']]],
                fmt='none', color='black', capsize=5, capthick=2, alpha=0.8)

    # Add accuracy value to the right of CI bar
    ax.text(results['ci_upper'] + 0.015, i, f"{results['accuracy']:.3f}",
            va='center', ha='left', fontsize=11, fontweight='bold')

ax.set_yticks(y_positions)
ax.set_yticklabels([m[0] for m in sorted_models], fontsize=11)
ax.set_xlabel('Extraction Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Overall Extraction Accuracy Across Models', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0.3, 0.7)

ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.5, -0.7, 'Random (0.5)', ha='center', fontsize=9, color='red', style='italic')

# Info box
info_text = f"N=4,217 per model\\n95% Bootstrap CI\\n(10,000 iterations)\\nτ* = {best_threshold:.2f}"
props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='black')
ax.text(0.68, 0.15, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('visualizations/fig_overall_accuracy.pdf', dpi=300, bbox_inches='tight')
plt.savefig('visualizations/fig_overall_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# Figure 2: Dataset Performance Heatmap
# -----------------------------------------------------------------------------
print("Generating: fig_dataset_heatmap.pdf")
if dataset_results:
    # Create matrix for heatmap
    models_list = list(dataset_results.keys())
    datasets_list = ['MHJ_local', 'SafeMTData_1K', 'SafeMTData_Attack600', 'CoSafe']

    matrix = []
    for model in models_list:
        row = []
        for dataset in datasets_list:
            if dataset in dataset_results[model]:
                row.append(dataset_results[model][dataset])
            else:
                row.append(0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(datasets_list)))
    ax.set_yticks(np.arange(len(models_list)))
    ax.set_xticklabels(datasets_list, rotation=45, ha='right')
    ax.set_yticklabels(models_list)

    # Add values
    for i in range(len(models_list)):
        for j in range(len(datasets_list)):
            text = ax.text(j, i, f'{matrix[i][j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Model Performance Across Datasets', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Extraction Accuracy', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig('visualizations/fig_dataset_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('visualizations/fig_dataset_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Figure 3: Confidence Distribution
# -----------------------------------------------------------------------------
print("Generating: fig_confidence_distribution.pdf")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (model_name, results) in enumerate(sorted_models[:6]):
    ax = axes[idx]

    conf = results['confidence']
    preds = results['predictions']

    # Separate correct and incorrect
    correct_conf = conf[preds == 1]
    incorrect_conf = conf[preds == 0]

    # Plot histograms
    bins = np.linspace(0, 1, 21)
    ax.hist(correct_conf, bins=bins, alpha=0.5, label='Correct', color='green', density=True)
    ax.hist(incorrect_conf, bins=bins, alpha=0.5, label='Incorrect', color='red', density=True)

    # Add mean lines
    ax.axvline(correct_conf.mean(), color='green', linestyle='--', alpha=0.8)
    ax.axvline(incorrect_conf.mean(), color='red', linestyle='--', alpha=0.8)

    # Stats text
    stats_text = f"ACC: {results['accuracy']:.3f}\\nECE: {results['ece']:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(model_name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Confidence Distribution: Correct vs Incorrect Predictions',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('visualizations/fig_confidence_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('visualizations/fig_confidence_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# Figure 4: Accuracy-Calibration Tradeoff
# -----------------------------------------------------------------------------
print("Generating: fig_accuracy_calibration_tradeoff.pdf")
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each model
for model_name, results in all_results.items():
    acc = results['accuracy']
    ece = results['ece']

    ax.scatter(acc, ece, s=200, color=model_colors[model_name],
               alpha=0.7, edgecolors='black', linewidth=2)

    # Add label with offset to avoid overlap
    if model_name == 'claude-sonnet-4':
        offset = (0.005, -0.015)
    elif model_name == 'kimi-k2':
        offset = (0.005, 0.015)
    elif model_name == 'deepseek-v3.1':
        offset = (-0.005, -0.015)
    elif model_name == 'gemini-2.5-flash':
        offset = (-0.005, 0.015)
    elif model_name == 'gpt-4.1':
        offset = (0.008, 0)
    else:
        offset = (-0.008, 0)

    ax.annotate(model_name, (acc, ece),
                xytext=(acc + offset[0], ece + offset[1]),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

# Add ideal region (high acc, low ECE)
ideal_patch = plt.Rectangle((0.45, 0.35), 0.15, 0.25,
                           color='green', alpha=0.1, label='Ideal Region')
ax.add_patch(ideal_patch)

ax.set_xlabel('Extraction Accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy-Calibration Trade-off', fontsize=14, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3)
ax.set_xlim(0.35, 0.6)
ax.set_ylim(0.35, 0.65)

# Add legend
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('visualizations/fig_accuracy_calibration_tradeoff.pdf', dpi=300, bbox_inches='tight')
plt.savefig('visualizations/fig_accuracy_calibration_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# Figure 5: Performance Spread Analysis
# -----------------------------------------------------------------------------
print("Generating: fig_performance_spread.pdf")
if dataset_results:
    fig, ax = plt.subplots(figsize=(10, 6))

    spread_data = []
    for model_name in dataset_results.keys():
        accs = list(dataset_results[model_name].values())
        if accs:
            spread = max(accs) - min(accs)
            mean_acc = np.mean(accs)
            spread_data.append({
                'model': model_name,
                'spread': spread,
                'mean': mean_acc,
                'min': min(accs),
                'max': max(accs)
            })

    # Sort by spread
    spread_data.sort(key=lambda x: x['spread'])

    x_pos = np.arange(len(spread_data))

    # Plot bars for spread
    bars = ax.bar(x_pos, [d['spread'] for d in spread_data],
                   color=[model_colors[d['model']] for d in spread_data], alpha=0.7)

    # Add error bars showing min-max range
    for i, d in enumerate(spread_data):
        ax.plot([i, i], [d['min'], d['max']], 'k-', linewidth=2)
        ax.plot(i, d['mean'], 'ko', markersize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([d['model'] for d in spread_data], rotation=45, ha='right')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Performance Consistency Across Datasets', fontsize=14, fontweight='bold', pad=20)

    # Add legend
    ax.plot([], [], 'ko-', label='Mean with Range')
    ax.bar([], [], alpha=0.7, label='Spread (Max-Min)')
    ax.legend()

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/fig_performance_spread.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('visualizations/fig_performance_spread.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Figure 6: Threshold Optimization Curve
# -----------------------------------------------------------------------------
print("Generating: fig_threshold_optimization.pdf")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(thresholds, f1_scores, linewidth=2, color='blue')
ax.scatter([best_threshold], [best_f1], color='red', s=200, zorder=5)
ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.5)
ax.axhline(y=best_f1, color='red', linestyle='--', alpha=0.5)

ax.text(best_threshold + 0.02, best_f1 - 0.02,
        f'τ* = {best_threshold:.2f}\\nF1 = {best_f1:.3f}',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Threshold Optimization on Human-Labeled Calibration Set',
             fontsize=14, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('visualizations/fig_threshold_optimization.pdf', dpi=300, bbox_inches='tight')
plt.savefig('visualizations/fig_threshold_optimization.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[OK] All figures generated successfully!")
print("Figures saved in: visualizations/")

# ============================================================================
# PART 5: SAVE ALL RESULTS
# ============================================================================
print("\n" + "="*80)
print("PART 5: SAVING RESULTS")
print("="*80)

# Create summary dictionary (remove non-serializable data)
summary = {
    'analysis_timestamp': datetime.now().isoformat(),
    'optimal_threshold': best_threshold,
    'f1_at_threshold': best_f1,
    'n_labeled_samples': len(df_label),
    'n_samples_per_model': 4217,
    'n_bootstrap_iterations': 10000,
    'models': {}
}

# Clean up results for JSON serialization
for model_name, results in all_results.items():
    summary['models'][model_name] = {
        'accuracy': float(results['accuracy']),
        'ci_lower': float(results['ci_lower']),
        'ci_upper': float(results['ci_upper']),
        'ece': float(results['ece']),
        'brier': float(results['brier']),
        'wrong_at_80': float(results['wrong_at_80']),
        'wrong_at_90': float(results['wrong_at_90']),
        'wrong_at_95': float(results['wrong_at_95']),
        'aurc': float(results['aurc']),
        'n_samples': int(results['n_samples']),
        'n_high_conf_80': int(results['n_high_conf_80']),
        'n_high_conf_90': int(results['n_high_conf_90']),
        'n_high_conf_95': int(results['n_high_conf_95'])
    }

summary['dataset_analysis'] = dataset_results

# Save to JSON
output_file = 'complete_analysis_results_FIXED.json'
with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"Results saved to: {output_file}")

# ============================================================================
# PART 6: LATEX TABLE GENERATION
# ============================================================================
print("\n" + "="*80)
print("PART 6: LATEX TABLES FOR PAPER")
print("="*80)

print("\n% Main results table")
print("\\begin{table}[t]")
print("\\centering")
print("\\caption{Complete model evaluation results on OBJEX dataset}")
print("\\begin{tabular}{lcccccc}")
print("\\toprule")
print("Model & Acc & 95\\% CI & ECE & Brier & Wrong@0.9 & AURC \\\\")
print("\\midrule")

for model_name, results in sorted_models:
    acc = results['accuracy']
    ci_l = results['ci_lower']
    ci_u = results['ci_upper']
    ece = results['ece']
    brier = results['brier']
    wrong90 = results['wrong_at_90']
    aurc = results['aurc']

    # Bold the best values
    if model_name == sorted_models[0][0]:
        print(f"\\texttt{{{model_name}}} & \\textbf{{{acc:.3f}}} & [{ci_l:.3f}, {ci_u:.3f}] & "
              f"{ece:.3f} & {brier:.3f} & {wrong90:.1f}\\% & {aurc:.3f} \\\\")
    else:
        print(f"\\texttt{{{model_name}}} & {acc:.3f} & [{ci_l:.3f}, {ci_u:.3f}] & "
              f"{ece:.3f} & {brier:.3f} & {wrong90:.1f}\\% & {aurc:.3f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ============================================================================
# PART 7: FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n1. Optimal threshold: τ* = {best_threshold:.2f} (F1 = {best_f1:.3f})")
print(f"2. Best performing model: {sorted_models[0][0]} (Acc = {sorted_models[0][1]['accuracy']:.3f})")
print(f"3. Worst performing model: {sorted_models[-1][0]} (Acc = {sorted_models[-1][1]['accuracy']:.3f})")
print(f"4. All models evaluated on N = 4,217 samples each")

print("\n" + "="*80)
print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

print("\n[OK] Analysis completed successfully!")
print("Results saved to: complete_analysis_results_FIXED.json")
print("Figures saved to: visualizations/")