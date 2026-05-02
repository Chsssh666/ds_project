# %% Load libraries
import pandas as pd
import matplotlib.pyplot as plt

# %% Load the dataset
df = pd.read_csv("./raw_data/yougov_australia.csv",
                 na_values=[" ", "__NA__"] , keep_default_na=True)

df = df.drop(columns=['RecordNo','weight'])
df['endtime'] = pd.to_datetime(df['endtime'].str.split().str[0], format='%d/%m/%Y')

# %% Analyze missing values
missing_rate = df.isnull().mean() * 100
missing_rate_sorted = missing_rate.sort_values(ascending=False)

missing_rate_sorted.to_csv('./data_figures/missing_rate_yougov.csv')

print(missing_rate_sorted[missing_rate_sorted > 0].to_string())

results = []

for thresh in range(0, 101, 5):
    cols_to_drop = missing_rate[missing_rate > thresh].index.tolist()
    df_temp = df.drop(columns=cols_to_drop).dropna()
    results.append({
        'threshold': thresh,
        'n_cols': df_temp.shape[1],
        'n_rows': df_temp.shape[0]
    })

results_df = pd.DataFrame(results)
print(results_df.to_string())

# %% threshold of missing values analysis plot
fig, ax1 = plt.subplots(figsize=(10, 6))

ax2 = ax1.twinx()

ax1.plot(results_df['threshold'], results_df['n_rows'], color='steelblue', marker='o', label='number of rows')
ax2.plot(results_df['threshold'], results_df['n_cols'], color='orange', marker='s', label='number of columns')

ax1.set_xlabel('threshold (%)')
ax1.set_ylabel('rows', color='steelblue')
ax2.set_ylabel('columns', color='orange')

ax1.axvline(x=20, color='red', linestyle='--', label='suggested threshold (20%)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.title('threshold vs rows/columns')
plt.tight_layout()
plt.savefig('./data_figures/yougov_threshold_analysis.png', dpi=150)
plt.show()

# %% Drop columns with more than 21% missing values (PHQ4_2 20.026749)
df = df.drop(columns=missing_rate[missing_rate > 21].index.tolist())

# %% Consent Problem
sdate = "2021-02-19"
edate = "2021-10-18"
mask = (df["endtime"] >= sdate) & (df["endtime"] <= edate)

for i in range(1, 5):
    df.loc[mask, f"PHQ4_{i}"] = df.loc[mask, f"PHQ4_{i}"].fillna("N/A")

for i in range(1, 14):
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask, f"d1_health_{i}"].fillna("N/A")

for i in [98, 99]:
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask, f"d1_health_{i}"].fillna("N/A")

# %% Drop rows with any remaining missing values
df = df.dropna()
print(f"rows: {df.shape[0]}")
print(f"columns names: {df.columns.tolist()}")

# %% Map frequency strings to numeric values
frequency_dict = {"Always": 5, "Frequently": 4,
                  "Sometimes": 3, "Rarely": 2, "Not at all": 1}
for col in df.columns:
    if col.startswith("i12_health_"):
        df[col] = df[col].map(frequency_dict)

# %% Construct protective behaviour scale and binary target variable
# Using all i12_health_ columns
protective_behaviour_cols = [col for col in df.columns if col.startswith("i12_")]

df["protective_behaviour_scale"] = df[protective_behaviour_cols].median(axis=1)
df["protective_behaviour_binary"] = df["protective_behaviour_scale"].apply(
    lambda x: "Yes" if x >= 4 else "No"
)
df = df.drop(columns=protective_behaviour_cols)
df = df.drop(columns=['protective_behaviour_scale'])

# %% Consolidate comorbidities (d1_health_*) into a single categorical variable
d1_cols = [col for col in df.columns if col.startswith("d1_")]

df["d1_comorbidities"] = "Yes"
df.loc[df["d1_health_99"] == "Yes", "d1_comorbidities"] = "No"
df.loc[df["d1_health_99"] == "N/A", "d1_comorbidities"] = "NA"
df.loc[df["d1_health_98"] == "Yes", "d1_comorbidities"] = "Prefer_not_to_say"

df = df.drop(columns=d1_cols)

print(df["protective_behaviour_binary"].value_counts())
print(df["d1_comorbidities"].value_counts())

# %% household number
df['household_size'] = df['household_size'].replace({'8 or more': 8})
df['household_size'] = pd.to_numeric(df['household_size'], errors='coerce')
df = df.dropna(subset=['household_size'])
df['household_size'] = df['household_size'].astype(int)

# %% Save the cleaned dataset
df.to_csv("./cleaned_data/aus_yougov_cleaned.csv", index=False)
