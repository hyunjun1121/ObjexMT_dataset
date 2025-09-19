import matplotlib.pyplot as plt
import numpy as np

# Test data - same as what we see in debug output
models = ['GPT-4.1', 'Claude-Sonnet-4', 'Qwen3-235B', 'Kimi-K2', 'DeepSeek-V3.1', 'Gemini-2.5']
datasets = ['SafeMTData_Attack600', 'SafeMTData_1K', 'MHJ', 'CoSafe']
colors = ['#1E88E5', '#D32F2F', '#388E3C', '#F57C00', '#7B1FA2', '#455A64']

# Data from debug output
data = {
    'GPT-4.1': [0.162, 0.502, 0.816, 0.309],
    'Claude-Sonnet-4': [0.292, 0.635, 0.853, 0.272],
    'Qwen3-235B': [0.200, 0.495, 0.717, 0.306],
    'Kimi-K2': [0.328, 0.635, 0.857, 0.222],
    'DeepSeek-V3.1': [0.320, 0.624, 0.831, 0.247],
    'Gemini-2.5': [0.215, 0.571, 0.814, 0.247]
}

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(datasets))
width = 0.12
offsets = np.linspace(-(len(models)-1)*width/2, (len(models)-1)*width/2, len(models))

for idx, (model, color) in enumerate(zip(models, colors)):
    values = data[model]
    bars = ax.bar(x + offsets[idx], values, width, label=model, alpha=0.8, color=color)
    print(f"Drew {model}: {values}")

ax.set_xlabel('Dataset')
ax.set_ylabel('Extraction Accuracy')
ax.set_title('Model Performance Across Datasets - TEST')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='upper right', ncol=2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1)
ax.set_xlim(-0.5, len(datasets) - 0.5)

plt.tight_layout()
plt.savefig('test_per_dataset.png', dpi=150)
print("\nSaved test_per_dataset.png")

# Create heatmap
fig2, ax2 = plt.subplots(figsize=(10, 7))

matrix = np.array([data[model] for model in models])
im = ax2.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

ax2.set_xticks(np.arange(len(datasets)))
ax2.set_yticks(np.arange(len(models)))
ax2.set_xticklabels(datasets)
ax2.set_yticklabels(models)

# Add text annotations
for i in range(len(models)):
    for j in range(len(datasets)):
        text = ax2.text(j, i, f'{matrix[i, j]:.3f}',
                       ha="center", va="center", color="black", fontweight='bold')

ax2.set_title('Model Performance Heatmap - TEST')
ax2.set_xlabel('Dataset')
ax2.set_ylabel('Model')

cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Extraction Accuracy', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('test_heatmap.png', dpi=150)
print("Saved test_heatmap.png")

print("\nTest complete!")