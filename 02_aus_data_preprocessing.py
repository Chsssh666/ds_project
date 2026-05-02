# %% Imports
import pandas as pd
import matplotlib.pyplot as plt

# %% Load dataset
yougov_df = pd.read_csv("./cleaned_data/aus_yougov_cleaned.csv")
oxcgrt_df = pd.read_csv("./cleaned_data/aus_oxcgrt_cleaned.csv")

# %% Merge datasets on 'state' and 'endtime'
df = yougov_df.merge(oxcgrt_df, 
                             left_on=['endtime', 'state'], 
                             right_on=['Date', 'RegionName'], 
                             how='left')

df = df.drop(columns=['Date', 'RegionName','endtime'])

#####################################
######## data preprocessing  ########
#####################################

# %% transform qweek
df['qweek'] = df['qweek'].str.extract(r'(\d+)').astype(int)
print(df['qweek'].describe())

# %% i2_health 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Full distribution
axes[0].hist(df['i2_health'].dropna(), bins=50, color='steelblue')
axes[0].set_title('i2_health - Full distribution')
axes[0].set_xlabel('Number of contacts')

# Capped at 50 to see the main distribution
axes[1].hist(df['i2_health'].clip(upper=50).dropna(), bins=50, color='steelblue')
axes[1].set_title('i2_health - Capped at 50')
axes[1].set_xlabel('Number of contacts')

plt.tight_layout()
plt.show()

# %% i9_health
print(df['i9_health'].value_counts())
df = pd.get_dummies(df, columns=['i9_health'], prefix='i9_health', drop_first=False)
print(df[['i9_health_Yes', 'i9_health_No', 'i9_health_Not sure']].sum())
# %% i11_health
print(df['i11_health'].value_counts())
willing_dict = {
    "Very unwilling": 5,
    "Somewhat unwilling": 4,
    "Neither willing nor unwilling": 3,
    "Somewhat willing": 2,
    "Very willing": 1,
    "Not sure": None
}

df['i11_health'] = df['i11_health'].map(willing_dict)
print(df['i11_health'].value_counts(dropna=False))

# the number of "Not sure" responses is low (only 2%)
# this is a ordinal variable, so we can drop these 'Not sure' rows
df = df.dropna(subset=['i11_health'])

# %% Gender
print(df['gender'].value_counts())
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# %% state
df = pd.get_dummies(df, columns=['state'], prefix='state', drop_first=False)
print(df[[col for col in df.columns if col.startswith('state_')]].sum())

# %% employment_status
print(df['employment_status'].value_counts())
df = pd.get_dummies(df, columns=['employment_status'], prefix='employment_status', drop_first=False)
print(df[[col for col in df.columns if col.startswith('employment_status_')]].sum())

# %% WCRex2
print(df['WCRex2'].value_counts())
confidence_dict = {
    "No confidence at all": 1,
    "Not very much confidence": 2,
    "A fair amount of confidence": 3,
    "A lot of confidence": 4,
    "Don't know": None
}

df['WCRex2'] = df['WCRex2'].map(confidence_dict)
df = df.dropna(subset=['WCRex2'])
print(df['WCRex2'].value_counts(dropna=False))

# %% PHQ4_*
phq4_dict = {
    "Not at all": 4,
    "Several days": 3,
    "More than half the days": 2,
    "Nearly every day": 1,
    "Prefer not to say": None,
    "N/A": "N/A"
}

for i in range(1, 5):
    df[f'PHQ4_{i}'] = df[f'PHQ4_{i}'].map(phq4_dict)

print(df[['PHQ4_1', 'PHQ4_2', 'PHQ4_3', 'PHQ4_4']].value_counts(dropna=False))

for i in range(1, 5):
    df = df.dropna(subset=[f'PHQ4_{i}'])
print(f"remaining rows: {df.shape[0]}")

# %% WCRex1
print(df['WCRex1'].value_counts())
wcrex1_dict = {
    "Very badly": 1,
    "Somewhat badly": 2,
    "Somewhat well": 3,
    "Very well": 4,
    "Don't know": None
}

df['WCRex1'] = df['WCRex1'].map(wcrex1_dict)
df = df.dropna(subset=['WCRex1'])
print(f"remaining rows: {df.shape[0]}")

# %% r1_*
print(df['r1_1'].value_counts())
print(df['r1_2'].value_counts())
for col in ['r1_1', 'r1_2']:
    df[col] = df[col].replace({"7 - Agree": 7, "1 – Disagree": 1})
    df[col] = df[col].astype(int)

print(df[['r1_1', 'r1_2']].describe())

# %% d1_comorbidities
print(df['d1_comorbidities'].value_counts())
df = pd.get_dummies(df, columns=['d1_comorbidities'], prefix='d1_comorbidities', drop_first=False)
print(df[[col for col in df.columns if col.startswith('d1_comorbidities_')]].sum())

# %% protective_behaviour_binary
df['protective_behaviour_binary'] = df['protective_behaviour_binary'].map({'Yes': 1, 'No': 0})
print(df['protective_behaviour_binary'].value_counts())

# %% Save data
df.to_csv("./cleaned_data/aus_preprocessed.csv", index=False)
