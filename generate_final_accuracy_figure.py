import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
excel_path = 'E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx'
xl = pd.ExcelFile(excel_path)

# Define threshold
threshold = 0.66

# Model configurations with colors
models = [
    ('similarity_gpt-4.1', 'gpt-4.1', '#1E88E5'),
    ('similarity_claude-sonnet-4-2025', 'claude-sonnet-4', '#D32F2F'),
    ('similarity_Qwen3-235B-A22B-fp8-', 'Qwen3-235B-A22B-FP8', '#388E3C'),
    ('similarity_moonshotaiKimi-K2-In', 'kimi-k2', '#F57C00'),
    ('similarity_deepseek-aiDeepSeek-', 'deepseek-v3.1', '#7B1FA2'),
    ('similarity_gemini-2.5-flash', 'gemini-2.5-flash', '#455A64')
]

results = []

for sim_sheet, model_name, color in models:
    # Load similarity scores - DO NOT deduplicate!
    # Each row is a separate jailbreak attempt even if base_prompt is the same
    df = pd.read_excel(xl, sim_sheet, usecols=['similarity_score'])

    # Calculate accuracy - each row counts separately
    accuracy = (df['similarity_score'] >= threshold).mean()

    # Bootstrap for CI
    n_bootstrap = 10000
    np.random.seed(42)
    bootstrap_accs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(df), len(df), replace=True)
        boot_acc = (df.iloc[indices]['similarity_score'] >= threshold).mean()
        bootstrap_accs.append(boot_acc)

    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)

    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'Color': color
    })

# Sort by accuracy
results.sort(key=lambda x: x['Accuracy'], reverse=True)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
y_positions = range(len(results))
for i, r in enumerate(results):
    # Main bar
    ax.barh(i, r['Accuracy'], color=r['Color'], alpha=0.7, height=0.6)

    # Error bars for CI
    ax.errorbar(r['Accuracy'], i, xerr=[[r['Accuracy'] - r['CI_Lower']],
                                         [r['CI_Upper'] - r['Accuracy']]],
                fmt='none', color='black', capsize=5, capthick=2, alpha=0.8)

    # Add accuracy value to the right of CI bar
    ax.text(r['CI_Upper'] + 0.015, i, f"{r['Accuracy']:.3f}",
            va='center', ha='left', fontsize=11, fontweight='bold')

# Customize plot
ax.set_yticks(y_positions)
ax.set_yticklabels([r['Model'] for r in results], fontsize=11)
ax.set_xlabel('Extraction Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Overall Extraction Accuracy Across Models', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0.3, 0.7)

# Add grid
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add vertical line at 0.5
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.5, -0.7, 'Random (0.5)', ha='center', fontsize=9, color='red', style='italic')

# Add info box
info_text = f"N=4,217 per model\n95% Bootstrap CI\n(10,000 iterations)\nτ* = {threshold}"
props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='black')
ax.text(0.68, 0.15, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()

# Save figure
plt.savefig('fig_overall_accuracy.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('fig_overall_accuracy.png', dpi=300, bbox_inches='tight')

print("Figure saved as fig_overall_accuracy.pdf and fig_overall_accuracy.png")