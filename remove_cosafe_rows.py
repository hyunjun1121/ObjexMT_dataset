"""
Remove all CoSafe rows from OBJEX_dataset_labeling.xlsx
"""

import pandas as pd
import openpyxl

# File path
excel_file = 'OBJEX_dataset_labeling.xlsx'
output_file = 'OBJEX_dataset_labeling_no_cosafe.xlsx'

print(f"Loading {excel_file}...")

# Load the Excel file
xl = pd.ExcelFile(excel_file)

# Dictionary to store filtered dataframes
filtered_sheets = {}

# List all sheets
print(f"Found {len(xl.sheet_names)} sheets")

# Process each sheet
for sheet_name in xl.sheet_names:
    print(f"\nProcessing sheet: {sheet_name}")

    # Read the sheet
    df = pd.read_excel(xl, sheet_name)
    original_count = len(df)

    # Check if 'source' column exists
    if 'source' in df.columns:
        # Count CoSafe rows
        cosafe_count = (df['source'] == 'CoSafe').sum()

        if cosafe_count > 0:
            # Remove CoSafe rows
            df_filtered = df[df['source'] != 'CoSafe'].copy()
            new_count = len(df_filtered)
            print(f"  - Removed {cosafe_count} CoSafe rows")
            print(f"  - Rows: {original_count} -> {new_count}")
        else:
            df_filtered = df
            print(f"  - No CoSafe rows found")
    else:
        # No source column, keep as is
        df_filtered = df
        print(f"  - No 'source' column found, keeping all {original_count} rows")

    # Store the filtered dataframe
    filtered_sheets[sheet_name] = df_filtered

# Save to new Excel file
print(f"\nSaving to {output_file}...")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet_name, df in filtered_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  - Saved sheet: {sheet_name} ({len(df)} rows)")

print("\nSummary:")
print(f"Original file: {excel_file}")
print(f"New file: {output_file}")

# Calculate total removed rows
total_removed = 0
for sheet_name in xl.sheet_names:
    original_df = pd.read_excel(xl, sheet_name)
    if 'source' in original_df.columns:
        removed = (original_df['source'] == 'CoSafe').sum()
        if removed > 0:
            total_removed += removed
            print(f"  {sheet_name}: removed {removed} CoSafe rows")

print(f"\nTotal CoSafe rows removed: {total_removed}")
print("\nDone!")