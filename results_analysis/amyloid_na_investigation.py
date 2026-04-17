import pandas as pd

# Load your training set
df = pd.read_csv('results/tau_raw_Gothenburg_CNN3D_MetaTemporal_2split80-20_stratify-site,MetaTemporal_raw-metatemporal-test_20260409_160005/train-test-split/train-test-split_training-set.csv')

# Find rows where amyloid_status is NA (NaN or empty string)
na_rows = df[df['amyloid_status'].isna() | (df['amyloid_status'] == '')]


# Print all such rows
print(na_rows)

# Save all NA rows to a CSV for further inspection
na_rows.to_csv('amyloid_status_NA_rows.csv', index=False)

# Optionally, see how many and from which sites/diagnoses
print(f"\nTotal NAs: {len(na_rows)}")
print("\nBy site:")
print(na_rows['site'].value_counts())
print("\nBy dx:")
print(na_rows['dx'].value_counts())