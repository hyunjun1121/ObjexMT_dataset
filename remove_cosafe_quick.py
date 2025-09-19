"""
Quick removal of CoSafe rows from Excel sheets
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# File paths
input_file = 'OBJEX_dataset_labeling.xlsx'
output_file = 'OBJEX_dataset_labeling_no_cosafe.xlsx'

print("Starting CoSafe removal process...")

# Load Excel file
xl = pd.ExcelFile(input_file)

# Check which sheets have source column and CoSafe data
sheets_with_cosafe = []

print("\nChecking sheets for CoSafe data:")
for sheet in xl.sheet_names[:10]:  # Check first 10 sheets
    try:
        df = pd.read_excel(xl, sheet, nrows=100)  # Check first 100 rows
        if 'source' in df.columns:
            if 'CoSafe' in df['source'].values:
                sheets_with_cosafe.append(sheet)
                print(f"  {sheet}: Contains CoSafe data")
    except:
        pass

print(f"\nSheets with CoSafe data: {sheets_with_cosafe}")

# Process only sheets with source column
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet_name in xl.sheet_names:
        print(f"Processing: {sheet_name}")

        df = pd.read_excel(xl, sheet_name)
        original_count = len(df)

        if 'source' in df.columns:
            # Filter out CoSafe
            df_filtered = df[df['source'] != 'CoSafe']
            removed = original_count - len(df_filtered)
            if removed > 0:
                print(f"  -> Removed {removed} CoSafe rows")
        else:
            df_filtered = df

        # Write to new file
        df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\nSaved to: {output_file}")
print("Done!")