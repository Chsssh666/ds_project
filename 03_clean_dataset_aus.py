
import pandas as pd
from datetime import datetime

def convert_datetime(dt):
    date = dt.split()[0]
    return datetime.strptime(date, "%d/%m/%Y")


def household_convert(size_str):
    for i in range(1, 8):
        if size_str == str(i):
            return i
        elif size_str == "8 or more":
            return 8
        elif size_str == "Prefer not to say" or size_str == "Don't know":
            return None

df = pd.read_csv("./raw_data/yougov_australia.csv",
                 na_values=[" ", "__NA__"], keep_default_na=True)

df["endtime"] = df["endtime"].apply(convert_datetime)
thresh_value = 10781

missing_value_df = pd.read_csv('./data/missing_value_counts_aus.csv')
columns_to_drop = missing_value_df.loc[missing_value_df['Missing Value Count']
                                       > thresh_value, 'Variable Name'].tolist()

df.drop(columns=columns_to_drop, inplace=True)

sdate = "2021-02-10"
edate = "2021-10-18"
mask = (df["endtime"] <= edate) & (df["endtime"] >= sdate)

for i in range(1, 5):
    df.loc[mask, f"PHQ4_{i}"] = df.loc[mask, f"PHQ4_{i}"].fillna("N/A")
for i in range(1, 14):
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask,
                                            f"d1_health_{i}"].fillna("N/A")
for i in range(98, 100):
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask,
                                            f"d1_health_{i}"].fillna("N/A")

df.dropna(inplace=True)

for i in range(1, 3):
    df[f"r1_{i}"] = df[f"r1_{i}"].replace(
        {"7 - Agree": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2, "1 – Disagree": 1})

frequency_dict = {"Always": 5, "Frequently": 4,
                  "Sometimes": 3, "Rarely": 2, "Not at all": 1}
for column in df.columns:
    if column.startswith("i12_health_"):
        df[column] = df[column].map(frequency_dict)


df["face_mask_behaviour_scale"] = df[["i12_health_1",
                                      "i12_health_22", "i12_health_23", "i12_health_25"]].median(axis=1)
df["face_mask_behaviour_binary"] = df["face_mask_behaviour_scale"].apply(
    lambda x: "Yes" if x >= 4 else "No")

protective_behaviour_cols = [col for col in df if col.startswith("i12_")]

df["protective_behaviour_scale"] = df[protective_behaviour_cols].median(axis=1)
df["protective_behaviour_binary"] = df["protective_behaviour_scale"].apply(
    lambda x: "Yes" if x >= 4 else "No")

protective_behaviour_nomask_cols = [col for col in protective_behaviour_cols if not col in ["i12_health_1",
                                                                                            "i12_health_22", "i12_health_23", "i12_health_25"]]
df["protective_behaviour_nomask_scale"] = df[protective_behaviour_nomask_cols].median(
    axis=1)

d1_cols = [col for col in df if col.startswith("d1_")]

df["d1_comorbidities"] = "Yes"
df.loc[df["d1_health_99"] == "Yes", "d1_comorbidities"] = "No"
df.loc[df["d1_health_99"] == "N/A", "d1_comorbidities"] = "NA"
df.loc[df["d1_health_98"] == "Yes", "d1_comorbidities"] = "Prefer_not_to_say"

df = df.drop(d1_cols, axis=1)


start_date = df['endtime'].min()
end_date = df['endtime'].max()

df['week_number'] = ((df['endtime'] - start_date).dt.days // 14) + 1

df["household_size"] = df["household_size"].apply(household_convert)
df.dropna(inplace=True)

df = df.drop(["qweek", "weight",] + protective_behaviour_cols, axis=1)

df.to_csv("./data/yougov_australia_cleaned.csv", index=False)
