import pandas as pd

def mandates_convert(row):
    endtime = pd.to_datetime(row['endtime'], format='%Y-%m-%d')
    state = row['state']

    if states_date[state][0] <= endtime:
        return 1
    else:
        return 0

cleaned_df = pd.read_csv("./data/yougov_australia_cleaned.csv", keep_default_na=False)

mandate_df = pd.read_csv("./data/mandate_start_dates_aus.csv")
states_date = {}
for state, date in zip(mandate_df["RegionName"], mandate_df["Date"]):
    states_date.update({state: [date]})


for state, date_range in states_date.items():
    states_date[state] = [pd.to_datetime(
        date, format='%Y-%m-%d') for date in date_range]

cleaned_df['within_mandate_period'] = cleaned_df.apply(
    mandates_convert, axis=1)

convert_into_dummy_cols = ['state', 'gender', 'i9_health', 'employment_status', 'i11_health',
                           'WCRex1', 'WCRex2', 'PHQ4_1', 'PHQ4_2', 'PHQ4_3', 'PHQ4_4',
                           "d1_comorbidities"
                           # 'd1_health_1', 'd1_health_2', 'd1_health_3', 'd1_health_4', 'd1_health_5',
                           # 'd1_health_6', 'd1_health_7', 'd1_health_8', 'd1_health_9', 'd1_health_10',
                           # 'd1_health_11', 'd1_health_12', 'd1_health_13', 'd1_health_98', 'd1_health_99'
                           ]

for col in convert_into_dummy_cols:
    dummy = pd.get_dummies(cleaned_df[col], prefix=col, drop_first=True, dtype=int)
    cleaned_df = pd.concat([cleaned_df, dummy], axis=1)
    cleaned_df = cleaned_df.drop(col, axis=1)

cleaned_df.to_csv("./data/yougov_australia_preprocessed.csv", index=False)
