import pandas as pd

df = pd.read_csv("./raw_data/yougov_australia.csv",
                 na_values=[" ", "__NA__"], keep_default_na=True, low_memory=False)

missing_value_counts = {}
for col in df:

    missing_count = (df[col].isna()).sum()

    missing_value_counts[col] = missing_count

missing_value_df = pd.DataFrame(list(missing_value_counts.items()), columns=[
                                'Variable Name', 'Missing Value Count'])
missing_value_df = missing_value_df.sort_values(
    by=['Missing Value Count', 'Variable Name'])


missing_value_df.to_csv('./data/missing_value_counts_aus.csv', index=False)