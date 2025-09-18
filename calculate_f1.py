import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Load Excel file
excel_path = 'E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx'
xl_file = pd.ExcelFile(excel_path)

# Load Labeling sheet
df_label = pd.read_excel(xl_file, 'Labeling')

print("="*60)
print("F1 Score Calculation for Human-aligned Thresholding")
print("="*60)

# Convert human labels to binary
label_mapping = {
    'Exact match': 1,
    'High similarity': 1,
    'Moderate similarity': 0,
    'Low similarity': 0
}

df_label['human_binary'] = df_label['human_label'].map(label_mapping)

print(f"\nTotal samples: {len(df_label)}")
print(f"Positive samples (Exact + High): {df_label['human_binary'].sum()}")
print(f"Negative samples (Moderate + Low): {(1-df_label['human_binary']).sum()}")

# Find optimal threshold by F1 score
thresholds = np.arange(0.0, 1.01, 0.01)
best_f1 = 0
best_threshold = 0
best_metrics = {}

print("\nSearching for optimal threshold...")

for threshold in thresholds:
    predictions = (df_label['similarity_score'] >= threshold).astype(int)
    f1 = f1_score(df_label['human_binary'], predictions)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_metrics = {
            'f1': f1,
            'accuracy': accuracy_score(df_label['human_binary'], predictions),
            'precision': precision_score(df_label['human_binary'], predictions),
            'recall': recall_score(df_label['human_binary'], predictions)
        }

print(f"\n{'='*40}")
print(f"OPTIMAL THRESHOLD: {best_threshold:.2f}")
print(f"{'='*40}")
print(f"F1 Score:  {best_metrics['f1']:.4f}")
print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall:    {best_metrics['recall']:.4f}")
print(f"{'='*40}")

# Show F1 scores at different thresholds
print("\nF1 Scores at key thresholds:")
key_thresholds = [0.5, 0.55, 0.58, 0.6, 0.65, 0.7, 0.75, 0.8]
for t in key_thresholds:
    predictions = (df_label['similarity_score'] >= t).astype(int)
    f1 = f1_score(df_label['human_binary'], predictions)
    acc = accuracy_score(df_label['human_binary'], predictions)
    print(f"  τ={t:.2f}: F1={f1:.4f}, Acc={acc:.4f}")