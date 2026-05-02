import pandas as pd
import re
import matplotlib.pyplot as plt

df = pd.read_csv("./raw_data/OxCGRT_AUS_latest.csv")

drop_cols = []

for col in df.columns:
    if col in ['CountryName', 'CountryCode', 'RegionCode', 'Jurisdiction','M1_Wildcard']:
        drop_cols.append(col)
    elif col.endswith('_Flag'):
        drop_cols.append(col)
    elif col.startswith('V'):
        drop_cols.append(col)
    elif col.endswith('_ForDisplay'):
        drop_cols.append(col)

print(f"the number of columns to be dropped: {len(drop_cols)}")
print(f"columns to be dropped: {drop_cols}\n")

df = df.drop(columns=drop_cols)

print(f"remaining columns: {df.shape[1]}\n")
print("=" * 50)
print("remaining column names:")
print("=" * 50)
for i, col in enumerate(df.columns):
    print(f"{i+1:3d}. {col}")

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
print(df.dtypes)

df.to_csv("./cleaned_data/aus_oxcgrt_before_cleaned.csv", index=False)

df = pd.read_csv("./cleaned_data/aus_oxcgrt_before_cleaned.csv")
drop_cols = [ col for col in df.columns if col in [
     'StringencyIndex_Average',
     'GovernmentResponseIndex_Average',
     'ContainmentHealthIndex_Average',
     'EconomicSupportIndex',
     'MajorityVaccinated'
 ]]
df = df.drop(columns=drop_cols)
print(f"remaining columns: {df.shape[1]}\n")
print("=" * 50)
print("remaining column names:")
print("=" * 50)
for i, col in enumerate(df.columns):
    print(f"{i+1:3d}. {col}")

missing_rate = df.isnull().mean() * 100
print(missing_rate.sort_values(ascending=False))

non_zero = missing_rate[missing_rate > 0].sort_values(ascending=False)
zero_count = (missing_rate == 0).sum()
other = pd.Series({'Others': 0}, )
fig, ax = plt.subplots(figsize=(12, 6))
non_zero.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title(f'Missing Value Rates ({zero_count} columns with no missing values)')
ax.set_xlabel('Column Name')
ax.set_ylabel('Missing Rate (%)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('./data_figures/aus_oxcgrt_missing_rate.png', dpi=150)
plt.show()

df = df.drop(columns=missing_rate[missing_rate > 80].index.tolist())
df = df.dropna()
df.to_csv("./cleaned_data/aus_oxcgrt_cleaned.csv", index=False)