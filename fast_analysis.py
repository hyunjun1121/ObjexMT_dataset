import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("OBJEX Dataset Fast Analysis")
print("="*60)

# Load data
excel_path = 'E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx'
xl_file = pd.ExcelFile(excel_path)

# 1. Human-aligned thresholding
print("\n1. Human-aligned Thresholding")
print("-"*40)
df_label = pd.read_excel(xl_file, 'Labeling')

label_mapping = {
    'Exact match': 1,
    'High similarity': 1,
    'Moderate similarity': 0,
    'Low similarity': 0
}

df_label['human_binary'] = df_label['human_label'].map(label_mapping)

# Find optimal threshold
thresholds = np.arange(0.0, 1.01, 0.01)
best_f1 = 0
best_threshold = 0

for threshold in thresholds:
    predictions = (df_label['similarity_score'] >= threshold).astype(int)
    f1 = f1_score(df_label['human_binary'], predictions)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold:.2f}")
print(f"Best F1 score: {best_f1:.4f}")

# 2. Model performance
print("\n2. Model Performance Analysis")
print("-"*40)

models = {
    'gpt_4.1': 'GPT-4.1',
    'claude-sonnet-4': 'Claude Sonnet 4',
    'Qwen3-235B-A22B-fp8': 'Qwen3-235B',
    'moonshotaiKimi-K2-In': 'Moonshot Kimi',
    'deepseek-aiDeepSeek': 'DeepSeek',
    'gemini-2.5-flash': 'Gemini 2.5'
}

results = []

for model_key, model_name in models.items():
    # Find matching sheets
    extract_sheet = None
    sim_sheet = None

    for sheet in xl_file.sheet_names:
        if sheet.startswith('extracted_') and model_key in sheet:
            extract_sheet = sheet
        if sheet.startswith('similarity_') and model_key in sheet:
            sim_sheet = sheet

    if extract_sheet and sim_sheet:
        # Load data
        df_extract = pd.read_excel(xl_file, extract_sheet)
        df_sim = pd.read_excel(xl_file, sim_sheet)

        # Merge data
        merged = pd.merge(
            df_extract[['base_prompt', 'extraction_confidence']],
            df_sim[['base_prompt', 'similarity_score']],
            on='base_prompt',
            how='inner'
        )

        # Calculate accuracy
        predictions = (merged['similarity_score'] >= best_threshold).astype(int)
        accuracy = predictions.mean()

        # Calculate confidence metrics
        confidence = pd.to_numeric(merged['extraction_confidence'], errors='coerce')
        confidence = confidence.fillna(50) / 100.0

        # ECE calculation (simplified)
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
        wrong_90 = ((predictions == 0) & (confidence >= 0.9)).mean()

        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Samples': len(merged),
            'ECE': ece,
            'Brier': brier,
            'Wrong@0.9': wrong_90
        })

        print(f"{model_name:20s}: Acc={accuracy:.4f}, N={len(merged):4d}, ECE={ece:.4f}")

# 3. Create results DataFrame
print("\n3. Summary Results")
print("-"*40)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\nTop Models by Accuracy:")
print(results_df[['Model', 'Accuracy', 'Samples']].head())

print("\nBest Calibrated Models (lowest ECE):")
print(results_df.nsmallest(3, 'ECE')[['Model', 'ECE', 'Brier']])

# 4. Dataset-wise performance
print("\n4. Dataset Performance")
print("-"*40)

datasets = ['SafeMTData_Attack600', 'SafeMTData_1K', 'MHJ', 'CoSafe']

for model_key, model_name in list(models.items())[:3]:  # Top 3 models only
    extract_sheet = None
    sim_sheet = None

    for sheet in xl_file.sheet_names:
        if sheet.startswith('extracted_') and model_key in sheet:
            extract_sheet = sheet
        if sheet.startswith('similarity_') and model_key in sheet:
            sim_sheet = sheet

    if extract_sheet and sim_sheet:
        df_extract = pd.read_excel(xl_file, extract_sheet)
        df_sim = pd.read_excel(xl_file, sim_sheet)

        merged = pd.merge(
            df_extract[['source', 'base_prompt']],
            df_sim[['base_prompt', 'similarity_score']],
            on='base_prompt',
            how='inner'
        )

        print(f"\n{model_name}:")
        for dataset in datasets:
            dataset_data = merged[merged['source'] == dataset]
            if len(dataset_data) > 0:
                acc = (dataset_data['similarity_score'] >= best_threshold).mean()
                print(f"  {dataset:20s}: {acc:.4f} (n={len(dataset_data)})")

# 5. Save results
print("\n5. Saving Results")
print("-"*40)

# Save to Excel
with pd.ExcelWriter('fast_analysis_results.xlsx', engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Model_Performance', index=False)

    # Add threshold info
    threshold_df = pd.DataFrame([{
        'Optimal_Threshold': best_threshold,
        'F1_Score': best_f1,
        'Calibration_Samples': len(df_label)
    }])
    threshold_df.to_excel(writer, sheet_name='Threshold', index=False)

print("[OK] Results saved to fast_analysis_results.xlsx")

# Generate simple LaTeX table
print("\n6. LaTeX Table")
print("-"*40)

latex_table = "\\begin{tabular}{lcc}\n"
latex_table += "\\toprule\n"
latex_table += "Model & Accuracy & ECE \\\\\n"
latex_table += "\\midrule\n"

for _, row in results_df.head(6).iterrows():
    latex_table += f"{row['Model']} & {row['Accuracy']:.3f} & {row['ECE']:.3f} \\\\\n"

latex_table += "\\bottomrule\n"
latex_table += "\\end{tabular}"

with open('fast_latex_table.tex', 'w') as f:
    f.write(latex_table)

print("[OK] LaTeX table saved to fast_latex_table.tex")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

print(f"\nKey Results:")
print(f"  * Optimal Threshold: {best_threshold:.2f}")
print(f"  * Best Model: {results_df.iloc[0]['Model']}")
print(f"  * Best Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"  * Total Models Analyzed: {len(results_df)}")
print(f"  * Total Samples: {results_df['Samples'].sum():,}")