"""
Fix similarity sheets - remove CoSafe rows that were missed
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# File paths
input_file = 'OBJEX_dataset_labeling.xlsx'
output_file = 'OBJEX_dataset_labeling_fixed.xlsx'

print("Fixing similarity sheets to remove CoSafe...")

# Load Excel file
xl = pd.ExcelFile(input_file)

# Process all sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet_name in xl.sheet_names:
        print(f"Processing: {sheet_name}")

        df = pd.read_excel(xl, sheet_name)
        original_count = len(df)

        # Check if this is a similarity sheet with 4217 rows
        if 'similarity' in sheet_name and len(df) == 4217:
            print(f"  -> Similarity sheet with 4217 rows detected")

            # Need to get source information
            # Try to find corresponding extraction sheet
            if sheet_name == 'similarity_gpt-4.1':
                ext_sheet = 'extracted_gpt_4.1'
            elif sheet_name == 'similarity_claude-sonnet-4-2025':
                ext_sheet = 'extracted_claude-sonnet-4'
            elif sheet_name == 'similarity_Qwen3-235B-A22B-fp8-':
                ext_sheet = 'extracted_Qwen3-235B-A22B-fp8-t'
            elif sheet_name == 'similarity_moonshotaiKimi-K2-In':
                ext_sheet = 'extracted_moonshotaiKimi-K2-Ins'
            elif sheet_name == 'similarity_deepseek-aiDeepSeek-':
                ext_sheet = 'extracted_deepseek-aiDeepSeek-V'
            elif sheet_name == 'similarity_gemini-2.5-flash':
                ext_sheet = 'extracted_gemini-2.5-flash'
            else:
                ext_sheet = None

            if ext_sheet:
                # Get the extraction sheet to identify which rows to keep
                df_ext = pd.read_excel(xl, ext_sheet)

                # Since extraction sheets already have CoSafe removed (2817 rows)
                # We need to keep only the first 2817 rows from similarity sheet
                # (assuming they're in the same order)
                if len(df_ext) == 2817:
                    df_filtered = df.iloc[:2817].copy()

                    # Copy source information if available
                    if 'source' in df_ext.columns and 'source' not in df_filtered.columns:
                        df_filtered['source'] = df_ext['source'].values

                    print(f"  -> Reduced from {original_count} to {len(df_filtered)} rows")
                    df = df_filtered

        # Write to new file
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\nSaved to: {output_file}")

# Verify the fix
print("\nVerification:")
xl_new = pd.ExcelFile(output_file)

for sheet in ['similarity_deepseek-aiDeepSeek-', 'extracted_deepseek-aiDeepSeek-V']:
    df = pd.read_excel(xl_new, sheet)
    print(f"{sheet}: {len(df)} rows")