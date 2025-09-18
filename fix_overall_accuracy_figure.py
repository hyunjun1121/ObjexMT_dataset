import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Generating Overall Accuracy Figure with CORRECT Sample Size")
print("="*70)

# Load data
excel_path = 'E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx'
xl = pd.ExcelFile(excel_path)

# Define threshold
threshold = 0.66

# Model configurations
models = [
    ('extracted_gpt_4.1', 'similarity_gpt-4.1', 'gpt-4.1', '#1E88E5'),
    ('extracted_claude-sonnet-4', 'similarity_claude-sonnet-4-2025', 'claude-sonnet-4', '#D32F2F'),
    ('extracted_Qwen3-235B-A22B-fp8-t', 'similarity_Qwen3-235B-A22B-fp8-', 'Qwen3-235B-A22B-FP8', '#388E3C'),
    ('extracted_moonshotaiKimi-K2-Ins', 'similarity_moonshotaiKimi-K2-In', 'kimi-k2', '#F57C00'),
    ('extracted_deepseek-aiDeepSeek-V', 'similarity_deepseek-aiDeepSeek-', 'deepseek-v3.1', '#7B1FA2'),
    ('extracted_gemini-2.5-flash', 'similarity_gemini-2.5-flash', 'gemini-2.5-flash', '#455A64')
]

results = []

for ext_sheet, sim_sheet, model_name, color in models:
    # Load data - just use similarity sheet which has everything
    df = pd.read_excel(xl, sim_sheet, usecols=['base_prompt', 'similarity_score'])

    # Calculate accuracy
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
        'Color': color,
        'N': len(df)  # This should be 4217
    })

# Sort by accuracy
results.sort(key=lambda x: x['Accuracy'], reverse=True)

# Print actual sample sizes
print("\nActual sample sizes per model:")
print("-"*40)
for r in results:
    print(f"{r['Model']:25s}: N = {r['N']:,}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
y_positions = range(len(results))
bars = []
for i, r in enumerate(results):
    bar = ax.barh(i, r['Accuracy'], color=r['Color'], alpha=0.7, height=0.6)
    bars.append(bar)

    # Error bars for CI
    ax.errorbar(r['Accuracy'], i, xerr=[[r['Accuracy'] - r['CI_Lower']],
                                         [r['CI_Upper'] - r['Accuracy']]],
                fmt='none', color='black', capsize=5, capthick=2, alpha=0.8)

    # Add accuracy value to the right of CI bar
    ci_right = r['CI_Upper']
    ax.text(ci_right + 0.015, i, f"{r['Accuracy']:.3f}",
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

# Add vertical line at threshold
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.5, -0.7, 'Random (0.5)', ha='center', fontsize=9, color='red', style='italic')

# Add info box with CORRECT sample size
# Get the actual N from the first model (they should all be similar)
actual_n = results[0]['N']
info_text = f"N={actual_n:,} per model\n95% Bootstrap CI\n(10,000 iterations)\nτ* = {threshold}"
props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='black')
ax.text(0.68, 0.15, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()

# Save figure - save as PNG first if PDF fails
output_path_png = 'fig_overall_accuracy.png'
output_path_pdf = 'fig_overall_accuracy.pdf'

plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path_png}")

try:
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"PDF also saved to: {output_path_pdf}")
except:
    print("Could not save PDF version")

plt.show()
print("\n[OK] Figure generated successfully with CORRECT sample size!")