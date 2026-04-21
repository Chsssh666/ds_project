import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

cleaned_df = pd.read_csv(
    "./data/yougov_australia_preprocessed.csv", keep_default_na=False)

mandate_period_label = cleaned_df.loc[:, "within_mandate_period"]

df_train, df_test = train_test_split(cleaned_df,
                                     test_size=0.2,
                                     random_state=42,
                                     stratify=mandate_period_label)

df_train.to_csv("./data/df_train_aus.csv", index=False)
df_test.to_csv("./data/df_test_aus.csv", index=False)

# %% Model 1: Predicting protective behaviour

label_encoder = LabelEncoder()

response_col = ["protective_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "protective_behaviour_nomask_scale",
                                        "endtime"])

X_train_model_1 = df_train[feature_cols]
X_test_model_1 = df_test[feature_cols]

y_train_model_1 = label_encoder.fit_transform(
    df_train[response_col].values.ravel())
y_test_model_1 = label_encoder.fit_transform(
    df_test[response_col].values.ravel())

X_train_model_1.to_csv("./data/X_train_model_1_aus.csv", index=False)
X_test_model_1.to_csv("./data/X_test_model_1_aus.csv", index=False)
pd.DataFrame({'y_train': y_train_model_1}).to_csv(
    "./data/y_train_model_1_aus.csv", index=False)
pd.DataFrame({'y_test': y_test_model_1}).to_csv(
    "./data/y_test_model_1_aus.csv", index=False)

# %% Model 1a: Predicting protective behaviour in early time

mandate_starter = "2022-01-01"

response_col = ["protective_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "protective_behaviour_nomask_scale",
                                        "endtime",
                                        "within_mandate_period"])

logic_subsetter_train = (df_train["endtime"] < mandate_starter) & (
    df_train["within_mandate_period"] == 0)
logic_subsetter_test = (df_test["endtime"] < mandate_starter) & (
    df_test["within_mandate_period"] == 0)

X_train_model_1a = df_train.loc[logic_subsetter_train, feature_cols]
X_test_model_1a = df_test.loc[logic_subsetter_test, feature_cols]

y_train_model_1a = label_encoder.fit_transform(
    df_train.loc[logic_subsetter_train, response_col].values.ravel())
y_test_model_1a = label_encoder.fit_transform(
    df_test.loc[logic_subsetter_test, response_col].values.ravel())

X_train_model_1a.to_csv("./data/X_train_model_1a_aus.csv", index=False)
X_test_model_1a.to_csv("./data/X_test_model_1a_aus.csv", index=False)
pd.DataFrame({'y_train': y_train_model_1a}).to_csv(
    "./data/y_train_model_1a_aus.csv", index=False)
pd.DataFrame({'y_test': y_test_model_1a}).to_csv(
    "./data/y_test_model_1a_aus.csv", index=False)

# %% Model 1b: Predicting protective behaviour in mandate periods

response_col = ["protective_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "protective_behaviour_nomask_scale",
                                        "endtime",
                                        "within_mandate_period"])

logic_subsetter_train = df_train["within_mandate_period"] == 1
logic_subsetter_test = df_test["within_mandate_period"] == 1

X_train_model_1b = df_train.loc[logic_subsetter_train, feature_cols]
X_test_model_1b = df_test.loc[logic_subsetter_test, feature_cols]

y_train_model_1b = label_encoder.fit_transform(
    df_train.loc[logic_subsetter_train, response_col].values.ravel())
y_test_model_1b = label_encoder.fit_transform(
    df_test.loc[logic_subsetter_test, response_col].values.ravel())

X_train_model_1b.to_csv("./data/X_train_model_1b_aus.csv", index=False)
X_test_model_1b.to_csv("./data/X_test_model_1b_aus.csv", index=False)
pd.DataFrame({'y_train': y_train_model_1b}).to_csv(
    "./data/y_train_model_1b_aus.csv", index=False)
pd.DataFrame({'y_test': y_test_model_1b}).to_csv(
    "./data/y_test_model_1b_aus.csv", index=False)
