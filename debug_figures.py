import pandas as pd
import numpy as np

excel_path = 'OBJEX_dataset_labeling.xlsx'
xl = pd.ExcelFile(excel_path)

# Model configurations - should be EXACTLY the same
models = [
    ('similarity_gpt-4.1', 'extracted_gpt_4.1', 'GPT-4.1', '#1E88E5'),
    ('similarity_claude-sonnet-4-2025', 'extracted_claude-sonnet-4', 'Claude-Sonnet-4', '#D32F2F'),
    ('similarity_Qwen3-235B-A22B-fp8-', 'extracted_Qwen3-235B-A22B-fp8-t', 'Qwen3-235B', '#388E3C'),
    ('similarity_moonshotaiKimi-K2-In', 'extracted_moonshotaiKimi-K2-Ins', 'Kimi-K2', '#F57C00'),
    ('similarity_deepseek-aiDeepSeek-', 'extracted_deepseek-aiDeepSeek-V', 'DeepSeek-V3.1', '#7B1FA2'),
    ('similarity_gemini-2.5-flash', 'extracted_gemini-2.5-flash', 'Gemini-2.5', '#455A64')
]

threshold = 0.66
model_results = []
dataset_results = []

print("Building model_results...")
for sim_sheet, ext_sheet, model_name, color in models:
    # Just add basic model info
    model_results.append({
        'Model': model_name,
        'Color': color,
        'Accuracy': 0.5  # dummy value
    })
    print(f"  Added to model_results: {model_name}")

print("\nBuilding dataset_results...")
for sim_sheet, ext_sheet, model_name, color in models:
    df_s = pd.read_excel(xl, sim_sheet)
    df_e = pd.read_excel(xl, ext_sheet)

    similarity_scores = df_s['similarity_score'].values
    sources = df_e['source'].values if 'source' in df_e.columns else None

    if sources is not None:
        dataset_accs = {}
        for dataset in ['SafeMTData_Attack600', 'SafeMTData_1K', 'MHJ_local', 'CoSafe']:
            dataset_mask = sources == dataset
            if dataset_mask.sum() > 0:
                dataset_acc = (similarity_scores[dataset_mask] >= threshold).mean()
                dataset_accs[dataset] = dataset_acc

        dataset_results.append({
            'Model': model_name,
            'Dataset_Accs': dataset_accs
        })
        print(f"  Added to dataset_results: {model_name}")

print("\n" + "="*60)
print("Checking model name consistency:")
print("="*60)

print("\nModels in model_results:")
for mr in model_results:
    print(f"  - '{mr['Model']}'")

print("\nModels in dataset_results:")
for dr in dataset_results:
    print(f"  - '{dr['Model']}'")

print("\n" + "="*60)
print("Testing color lookup:")
print("="*60)

for dr in dataset_results:
    model = dr['Model']
    try:
        color = next(m['Color'] for m in model_results if m['Model'] == model)
        print(f"  {model}: Found color {color}")
    except StopIteration:
        print(f"  {model}: COLOR NOT FOUND - Name mismatch!")

print("\n[OK] Debug complete")