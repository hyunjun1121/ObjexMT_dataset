import pandas as pd

print("="*60)
print("Checking Labeling Sheet Source Distribution")
print("="*60)

# Load data
excel_path = 'E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx'
xl = pd.ExcelFile(excel_path)

# Load Labeling sheet
df_label = pd.read_excel(xl, 'Labeling')

print(f"\nTotal samples in Labeling sheet: {len(df_label)}")
print("\nSource distribution:")
print("-"*40)

# Count by source
source_counts = df_label['source'].value_counts().sort_index()

for source, count in source_counts.items():
    percentage = (count / len(df_label)) * 100
    print(f"{source:25s}: {count:3d} ({percentage:5.1f}%)")

print("-"*40)
print(f"{'Total':25s}: {source_counts.sum():3d} (100.0%)")

# Check if it's stratified sampling (should be 75 each)
print("\n" + "="*60)
print("Analysis:")
print("-"*40)

unique_sources = df_label['source'].unique()
print(f"Number of unique sources: {len(unique_sources)}")
print(f"Expected per source (stratified): {300 // 4} = 75")

# Check human label distribution per source
print("\n" + "="*60)
print("Human Label Distribution by Source:")
print("="*60)

label_mapping = {
    'Exact match': 'Exact',
    'High similarity': 'High',
    'Moderate similarity': 'Moderate',
    'Low similarity': 'Low'
}

for source in source_counts.index:
    print(f"\n{source}:")
    source_data = df_label[df_label['source'] == source]
    label_dist = source_data['human_label'].value_counts()

    for label, count in label_dist.items():
        short_label = label_mapping.get(label, label)
        percentage = (count / len(source_data)) * 100
        print(f"  {short_label:10s}: {count:2d} ({percentage:5.1f}%)")

    # Calculate positive rate (Exact + High)
    positive = source_data['human_label'].isin(['Exact match', 'High similarity']).sum()
    positive_rate = (positive / len(source_data)) * 100
    print(f"  => Positive rate (Exact+High): {positive}/{len(source_data)} ({positive_rate:.1f}%)")

# Overall statistics
print("\n" + "="*60)
print("Overall Human Label Distribution:")
print("="*60)

overall_dist = df_label['human_label'].value_counts()
for label, count in overall_dist.items():
    percentage = (count / len(df_label)) * 100
    print(f"{label:20s}: {count:3d} ({percentage:5.1f}%)")

# Binary distribution
positive_total = df_label['human_label'].isin(['Exact match', 'High similarity']).sum()
negative_total = df_label['human_label'].isin(['Moderate similarity', 'Low similarity']).sum()

print(f"\nBinary distribution:")
print(f"  Positive (Exact+High):     {positive_total:3d} ({positive_total/len(df_label)*100:5.1f}%)")
print(f"  Negative (Moderate+Low):   {negative_total:3d} ({negative_total/len(df_label)*100:5.1f}%)")