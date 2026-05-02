# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %% Load dataset
df = pd.read_csv("./cleaned_data/aus_preprocessed.csv")

# %% Vaccination distribution
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(df['PopulationVaccinated'], bins=50, color='steelblue', edgecolor='white', linewidth=0.5)

ax.set_xlabel('Population Vaccinated (%)', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title('Distribution of Population Vaccinated', fontsize=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig('./data_figures/vac_dist.png', dpi=300, bbox_inches='tight')
plt.show()

# %% Vaccination groups test
bins = [0, 0.001, 70, 100]
labels = ['No vaccination', 'Early vaccination', 'High vaccination']

df['vac_group'] = pd.cut(df['PopulationVaccinated'], 
                          bins=[-0.001, 0.001, 70, 100], 
                          labels=labels)

print(df['vac_group'].value_counts().sort_index())

# %% Simple split
df['vac_group'] = (df['PopulationVaccinated'] > 0).astype(int)
print(df['vac_group'].value_counts())

df_train, df_test = train_test_split(df,
                                      test_size=0.2,
                                      random_state=42,
                                      stratify=df['vac_group'])

print(f"Train: {df_train.shape}")
print(f"Test: {df_test.shape}")
print(f"\nTrain vac_group:\n{df_train['vac_group'].value_counts()}")
print(f"\nTest vac_group:\n{df_test['vac_group'].value_counts()}")

# %% xy split
label_encoder = LabelEncoder()

response_col = 'protective_behaviour_binary'
feature_cols = df.columns.drop(['protective_behaviour_binary', 'vac_group', 'PopulationVaccinated'])

# Group 0: No vaccination
X_train_0 = df_train[df_train['vac_group'] == 0][feature_cols]
X_test_0 = df_test[df_test['vac_group'] == 0][feature_cols]
y_train_0 = df_train[df_train['vac_group'] == 0][response_col]
y_test_0 = df_test[df_test['vac_group'] == 0][response_col]

# Group 1: Vaccination period
X_train_1 = df_train[df_train['vac_group'] == 1][feature_cols]
X_test_1 = df_test[df_test['vac_group'] == 1][feature_cols]
y_train_1 = df_train[df_train['vac_group'] == 1][response_col]
y_test_1 = df_test[df_test['vac_group'] == 1][response_col]

print(f"Group 0 - Train: {X_train_0.shape}, Test: {X_test_0.shape}")
print(f"Group 1 - Train: {X_train_1.shape}, Test: {X_test_1.shape}")

# %% Save splits
X_train_0.to_csv('./data/aus_X_train_0.csv', index=False)
X_test_0.to_csv('./data/aus_X_test_0.csv', index=False)
y_train_0.to_csv('./data/aus_y_train_0.csv', index=False)
y_test_0.to_csv('./data/aus_y_test_0.csv', index=False)

X_train_1.to_csv('./data/aus_X_train_1.csv', index=False)
X_test_1.to_csv('./data/aus_X_test_1.csv', index=False)
y_train_1.to_csv('./data/aus_y_train_1.csv', index=False)
y_test_1.to_csv('./data/aus_y_test_1.csv', index=False)

# %%
