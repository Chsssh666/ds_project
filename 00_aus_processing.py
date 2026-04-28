import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


RAW_DIR = Path("./raw_data")
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
EDA_DIR = Path("./eda_figures")
EDA_DIR.mkdir(exist_ok=True)


def convert_datetime(value):
    date_part = str(value).split()[0]
    return datetime.strptime(date_part, "%d/%m/%Y")


def household_convert(value):
    if value in [str(i) for i in range(1, 8)]:
        return int(value)
    if value == "8 or more":
        return 8
    if value in ["Prefer not to say", "Don't know"]:
        return None
    return None

def save_bar_plot(series, title, xlabel, ylabel, filename, rotation=45):
    """Save a simple bar chart for a pandas Series."""
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=rotation)
    fig.tight_layout()
    fig.savefig(EDA_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_eda_plots(df):
    """Generate EDA figures from the cleaned, non-dummified dataset."""
    EDA_DIR.mkdir(exist_ok=True)

    if "state" in df.columns:
        state_counts = df["state"].value_counts().sort_values(ascending=False)
        save_bar_plot(
            state_counts,
            "Number of Responses by State",
            "State",
            "Count",
            "01_responses_by_state.png",
        )

    if "endtime" in df.columns:
        date_series = pd.to_datetime(df["endtime"], errors="coerce")
        responses_by_month = date_series.dt.to_period("M").value_counts().sort_index()
        responses_by_month.index = responses_by_month.index.astype(str)
        save_bar_plot(
            responses_by_month,
            "Number of Responses by Month",
            "Month",
            "Count",
            "02_responses_by_month.png",
            rotation=90,
        )

    for col, filename, title in [
        (
            "face_mask_behaviour_binary",
            "03_face_mask_behaviour_binary.png",
            "Face Mask Behaviour Binary Distribution",
        ),
        (
            "protective_behaviour_binary",
            "04_protective_behaviour_binary.png",
            "Protective Behaviour Binary Distribution",
        ),
    ]:
        if col in df.columns:
            save_bar_plot(
                df[col].value_counts(dropna=False),
                title,
                col,
                "Count",
                filename,
                rotation=0,
            )

    scale_cols = [
        "face_mask_behaviour_scale",
        "protective_behaviour_scale",
        "protective_behaviour_nomask_scale",
    ]

    for idx, col in enumerate([c for c in scale_cols if c in df.columns], start=5):
        fig, ax = plt.subplots(figsize=(8, 5))
        df[col].dropna().plot(kind="hist", bins=10, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(
            EDA_DIR / f"{idx:02d}_{col}_histogram.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    if {"state", "within_mandate_period"}.issubset(df.columns):
        mandate_state = pd.crosstab(df["state"], df["within_mandate_period"])

        fig, ax = plt.subplots(figsize=(10, 6))
        mandate_state.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Responses by State and Mandate Period")
        ax.set_xlabel("State")
        ax.set_ylabel("Count")
        ax.legend(title="within_mandate_period")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(
            EDA_DIR / "08_responses_by_state_and_mandate_period.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    if "household_size" in df.columns:
        household_counts = df["household_size"].value_counts().sort_index()
        save_bar_plot(
            household_counts,
            "Household Size Distribution",
            "Household Size",
            "Count",
            "09_household_size_distribution.png",
            rotation=0,
        )

    numeric_df = df.select_dtypes(include="number")
    keep_cols = [
        col
        for col in numeric_df.columns
        if col in [
            "age",
            "household_size",
            "r1_1",
            "r1_2",
            "face_mask_behaviour_scale",
            "protective_behaviour_scale",
            "protective_behaviour_nomask_scale",
            "within_mandate_period",
            "week_number",
        ]
    ]

    if len(keep_cols) >= 2:
        corr = numeric_df[keep_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(corr, aspect="auto")
        ax.set_title("Correlation Matrix for Selected Numeric Variables")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            EDA_DIR / "10_selected_numeric_correlation_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

def create_mask_mandate_dates():
    df = pd.read_csv(RAW_DIR / "OxCGRT_AUS_latest.csv")

    cols = ["RegionName", "RegionCode", "Date", "H6M_Facial Coverings"]
    df.index = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.loc[:, cols]

    rolling = (
        df.loc[:, ["RegionName", "H6M_Facial Coverings"]]
        .groupby("RegionName")
        .rolling(window=14)
        .mean()
    )

    mandates = (
        rolling[rolling["H6M_Facial Coverings"] >= 3]
        .groupby("RegionName")
        .head(1)
    )

    mandates.to_csv(DATA_DIR / "mandate_start_dates_aus.csv", index=True)


def create_missing_value_table():
    df = pd.read_csv(
        RAW_DIR / "yougov_australia.csv",
        na_values=[" ", "__NA__"],
        keep_default_na=True,
        low_memory=False,
    )

    missing_df = pd.DataFrame({
        "Variable Name": df.columns,
        "Missing Value Count": df.isna().sum().values,
    })

    missing_df = missing_df.sort_values(
        by=["Missing Value Count", "Variable Name"]
    )

    missing_df.to_csv(DATA_DIR / "missing_value_counts_aus.csv", index=False)


def clean_yougov_dataset():
    df = pd.read_csv(
        RAW_DIR / "yougov_australia.csv",
        na_values=[" ", "__NA__"],
        keep_default_na=True,
    )

    df["endtime"] = df["endtime"].apply(convert_datetime)

    missing_df = pd.read_csv(DATA_DIR / "missing_value_counts_aus.csv")
    columns_to_drop = missing_df.loc[
        missing_df["Missing Value Count"] > 10781,
        "Variable Name"
    ].tolist()

    df.drop(columns=columns_to_drop, inplace=True)

    date_mask = (df["endtime"] <= "2021-10-18") & (df["endtime"] >= "2021-02-10")

    for i in range(1, 5):
        df.loc[date_mask, f"PHQ4_{i}"] = df.loc[date_mask, f"PHQ4_{i}"].fillna("N/A")

    for i in list(range(1, 14)) + list(range(98, 100)):
        df.loc[date_mask, f"d1_health_{i}"] = df.loc[
            date_mask, f"d1_health_{i}"
        ].fillna("N/A")

    df.dropna(inplace=True)

    agreement_map = {
        "7 - Agree": 7,
        "6": 6,
        "5": 5,
        "4": 4,
        "3": 3,
        "2": 2,
        "1 – Disagree": 1,
    }

    for i in range(1, 3):
        df[f"r1_{i}"] = df[f"r1_{i}"].replace(agreement_map)

    frequency_map = {
        "Always": 5,
        "Frequently": 4,
        "Sometimes": 3,
        "Rarely": 2,
        "Not at all": 1,
    }

    for col in df.columns:
        if col.startswith("i12_health_"):
            df[col] = df[col].map(frequency_map)

    mask_cols = [
        "i12_health_1",
        "i12_health_22",
        "i12_health_23",
        "i12_health_25",
    ]

    df["face_mask_behaviour_scale"] = df[mask_cols].median(axis=1)
    df["face_mask_behaviour_binary"] = df["face_mask_behaviour_scale"].apply(
        lambda x: "Yes" if x >= 4 else "No"
    )

    protective_cols = [col for col in df.columns if col.startswith("i12_")]

    df["protective_behaviour_scale"] = df[protective_cols].median(axis=1)
    df["protective_behaviour_binary"] = df["protective_behaviour_scale"].apply(
        lambda x: "Yes" if x >= 4 else "No"
    )

    protective_nomask_cols = [
        col for col in protective_cols if col not in mask_cols
    ]

    df["protective_behaviour_nomask_scale"] = df[
        protective_nomask_cols
    ].median(axis=1)

    d1_cols = [col for col in df.columns if col.startswith("d1_")]

    df["d1_comorbidities"] = "Yes"
    df.loc[df["d1_health_99"] == "Yes", "d1_comorbidities"] = "No"
    df.loc[df["d1_health_99"] == "N/A", "d1_comorbidities"] = "NA"
    df.loc[df["d1_health_98"] == "Yes", "d1_comorbidities"] = "Prefer_not_to_say"

    df.drop(columns=d1_cols, inplace=True)

    start_date = df["endtime"].min()
    df["week_number"] = ((df["endtime"] - start_date).dt.days // 14) + 1

    df["household_size"] = df["household_size"].apply(household_convert)
    df.dropna(inplace=True)

    df.drop(columns=["qweek", "weight"] + protective_cols, inplace=True)

    df.to_csv(DATA_DIR / "yougov_australia_cleaned.csv", index=False)


def add_mandate_and_dummy_variables():
    df = pd.read_csv(
        DATA_DIR / "yougov_australia_cleaned.csv",
        keep_default_na=False,
    )

    mandate_df = pd.read_csv(DATA_DIR / "mandate_start_dates_aus.csv")

    state_dates = {
        state: pd.to_datetime(date, format="%Y-%m-%d")
        for state, date in zip(mandate_df["RegionName"], mandate_df["Date"])
    }

    df["within_mandate_period"] = df.apply(
        lambda row: int(
            state_dates[row["state"]] <= pd.to_datetime(row["endtime"], format="%Y-%m-%d")
        ),
        axis=1,
    )

    generate_eda_plots(df)

    dummy_cols = [
        "state",
        "gender",
        "i9_health",
        "employment_status",
        "i11_health",
        "WCRex1",
        "WCRex2",
        "PHQ4_1",
        "PHQ4_2",
        "PHQ4_3",
        "PHQ4_4",
        "d1_comorbidities",
    ]

    for col in dummy_cols:
        dummy = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        df = pd.concat([df, dummy], axis=1)
        df.drop(columns=col, inplace=True)

    df.to_csv(DATA_DIR / "yougov_australia_preprocessed.csv", index=False)


def save_model_dataset(
    df_train,
    df_test,
    model_name,
    target_col,
    drop_cols,
    train_filter=None,
    test_filter=None,
):
    encoder = LabelEncoder()

    feature_cols = df_train.columns.drop(drop_cols)

    if train_filter is None:
        train_filter = pd.Series(True, index=df_train.index)
    if test_filter is None:
        test_filter = pd.Series(True, index=df_test.index)

    X_train = df_train.loc[train_filter, feature_cols]
    X_test = df_test.loc[test_filter, feature_cols]

    y_train = encoder.fit_transform(
        df_train.loc[train_filter, [target_col]].values.ravel()
    )
    y_test = encoder.fit_transform(
        df_test.loc[test_filter, [target_col]].values.ravel()
    )

    X_train.to_csv(DATA_DIR / f"X_train_model_{model_name}_aus.csv", index=False)
    X_test.to_csv(DATA_DIR / f"X_test_model_{model_name}_aus.csv", index=False)

    pd.DataFrame({"y_train": y_train}).to_csv(
        DATA_DIR / f"y_train_model_{model_name}_aus.csv",
        index=False,
    )

    pd.DataFrame({"y_test": y_test}).to_csv(
        DATA_DIR / f"y_test_model_{model_name}_aus.csv",
        index=False,
    )


def split_and_create_model_files():
    df = pd.read_csv(
        DATA_DIR / "yougov_australia_preprocessed.csv",
        keep_default_na=False,
    )

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["within_mandate_period"],
    )

    df_train.to_csv(DATA_DIR / "df_train_aus.csv", index=False)
    df_test.to_csv(DATA_DIR / "df_test_aus.csv", index=False)

    base_drop_cols = [
        "RecordNo",
        "face_mask_behaviour_scale",
        "protective_behaviour_scale",
        "face_mask_behaviour_binary",
        "protective_behaviour_binary",
        "protective_behaviour_nomask_scale",
        "endtime",
    ]

    save_model_dataset(
        df_train=df_train,
        df_test=df_test,
        model_name="1",
        target_col="protective_behaviour_binary",
        drop_cols=base_drop_cols,
    )

    mandate_starter = "2022-01-01"

    early_train_filter = (
        (df_train["endtime"] < mandate_starter)
        & (df_train["within_mandate_period"] == 0)
    )

    early_test_filter = (
        (df_test["endtime"] < mandate_starter)
        & (df_test["within_mandate_period"] == 0)
    )

    save_model_dataset(
        df_train=df_train,
        df_test=df_test,
        model_name="1a",
        target_col="protective_behaviour_binary",
        drop_cols=base_drop_cols + ["within_mandate_period"],
        train_filter=early_train_filter,
        test_filter=early_test_filter,
    )

    mandate_train_filter = df_train["within_mandate_period"] == 1
    mandate_test_filter = df_test["within_mandate_period"] == 1

    save_model_dataset(
        df_train=df_train,
        df_test=df_test,
        model_name="1b",
        target_col="protective_behaviour_binary",
        drop_cols=base_drop_cols + ["within_mandate_period"],
        train_filter=mandate_train_filter,
        test_filter=mandate_test_filter,
    )


def main():
    create_mask_mandate_dates()
    create_missing_value_table()
    clean_yougov_dataset()
    add_mandate_and_dummy_variables()
    split_and_create_model_files()


if __name__ == "__main__":
    main()