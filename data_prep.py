"""Data loading, preprocessing, feature engineering, train-test split, and data validation.

Mirrors the Stage 2 steps in notebooks/Data_Preparation.ipynb so that the modelling
notebook, training scripts and API all use the same, leakage-safe preparation.

    load_raw            -> data loading ('?' -> NaN)
    clean               -> cleaning, deduplication, invalid-record removal, target
    engineer_features   -> ICD-9 groups, age midpoint, service utilisation, med changes
    build_preprocessor  -> ColumnTransformer (fit on the training split only)
    split_data          -> stratified 70/15/15 train/validation/test split
    validate_*          -> data validation at each stage (raises DataValidationError)
    prepare_splits      -> runs everything end to end
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"

TARGET = "readmitted_30d"
ID_COLUMNS = ["encounter_id", "patient_nbr"]
VALID_LABELS = {"<30", ">30", "NO"}

# discharge_disposition_id codes for patients who died or went to hospice (cannot be readmitted).
EXPIRED_HOSPICE_IDS = {11, 13, 14, 19, 20, 21}

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_midpoint", "service_utilisation", "num_med_changes",
]
CODED_CATEGORICAL_FEATURES = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
]
SPECIALTY_FEATURE = "medical_specialty"
SPECIALTY_TOP_K = 10

REQUIRED_RAW_COLUMNS = [
    *ID_COLUMNS, "race", "gender", "age", "weight", "admission_type_id",
    "discharge_disposition_id", "admission_source_id", "time_in_hospital", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed", "readmitted",
]
# Representative expectations from the capstone brief (~9% positive after cleaning).
EXPECTED_POSITIVE_RATE = (0.08, 0.10)
MAX_SPLIT_RATE_GAP = 0.005


class DataValidationError(ValueError):
    """Raised when the data breaks an expectation the pipeline relies on."""


def load_raw(path=RAW_DATA_PATH):
    """Load the raw CSV: '?' and blanks become NaN; the literal lab category 'None' is kept."""
    return pd.read_csv(path, na_values=["?", ""], keep_default_na=False, low_memory=False)


def clean(data):
    """Stage 2.1: invalid values, weight/constant columns, first encounter per patient,
    expired/hospice removal, and the binary target."""
    df = data.copy()
    df.loc[df["gender"].eq("Unknown/Invalid"), "gender"] = np.nan
    df = df.drop(columns=["weight"])
    constant = [
        c for c in df.columns
        if c not in {*ID_COLUMNS, "readmitted"} and df[c].nunique(dropna=False) <= 1
    ]
    df = df.drop(columns=constant)

    # No encounter date exists, so "first" means first in file order.
    df = df.drop_duplicates(subset="patient_nbr", keep="first")
    df = df.loc[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_IDS)].reset_index(drop=True)

    df[TARGET] = (df["readmitted"] == "<30").astype(int)
    return df.drop(columns=["readmitted"])


def icd9_group(code):
    """Map an ICD-9 code to a clinical category (Strack et al., 2014 grouping)."""
    if pd.isna(code):
        return "Missing"
    code = str(code).strip()
    if code.startswith(("V", "E")):
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return "Other"
    if int(value) == 250:
        return "Diabetes"
    if 390 <= value < 460 or int(value) == 785:
        return "Circulatory"
    if 460 <= value < 520 or int(value) == 786:
        return "Respiratory"
    if 520 <= value < 580 or int(value) == 787:
        return "Digestive"
    if 580 <= value < 630 or int(value) == 788:
        return "Genitourinary"
    if 710 <= value < 740:
        return "Musculoskeletal"
    if 800 <= value < 1000:
        return "Injury"
    if 140 <= value < 240:
        return "Neoplasms"
    return "Other"


def engineer_features(df):
    """Stage 2.2: row-wise features only (no statistics learned from the data, so no leakage)."""
    df = df.copy()
    for column in ["diag_1", "diag_2", "diag_3"]:
        df[f"{column}_group"] = df[column].map(icd9_group)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # "[70-80)" -> 75
    bounds = df["age"].str.extract(r"\[(\d+)-(\d+)\)").astype(float)
    df["age_midpoint"] = bounds.mean(axis=1)
    df = df.drop(columns=["age"])

    df["service_utilisation"] = df[["number_outpatient", "number_emergency", "number_inpatient"]].sum(axis=1)
    medications = [c for c in MEDICATION_COLUMNS if c in df.columns]
    df["num_med_changes"] = df[medications].isin(["Up", "Down"]).sum(axis=1)

    for column in CODED_CATEGORICAL_FEATURES:
        df[column] = df[column].astype(str)
    # Plain object dtype keeps sklearn imputers and MLflow schema inference predictable.
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    df[text_columns] = df[text_columns].astype(object)
    return df


def feature_columns(df):
    """Split model inputs into numeric, general categorical, and specialty features."""
    excluded = {*ID_COLUMNS, TARGET, *NUMERIC_FEATURES, SPECIALTY_FEATURE}
    categorical = [c for c in df.columns if c not in excluded]
    return list(NUMERIC_FEATURES), categorical, [SPECIALTY_FEATURE]


def build_preprocessor(numeric, categorical, specialty):
    """Stage 2.3: one ColumnTransformer, fitted on the training split only (inside the model Pipeline)."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    # Top-k specialty grouping is learned on the training data; rarer specialties share one column.
    specialty_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=SPECIALTY_TOP_K + 1)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
        ("specialty", specialty_pipe, specialty),
    ])


def split_data(df, random_state=RANDOM_STATE):
    """Stage 2.4: stratified 70/15/15 train/validation/test split, created before any fitting."""
    train, temp = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=random_state)
    validation, test = train_test_split(temp, test_size=0.50, stratify=temp[TARGET], random_state=random_state)

    splits = {"train": train, "validation": validation, "test": test}
    return {name: part.reset_index(drop=True) for name, part in splits.items()}


def validate_raw(data):
    """Schema and label checks on the raw CSV, before any cleaning."""
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in data.columns]
    if missing:
        raise DataValidationError(f"Raw data is missing required columns: {missing}")
    if data.empty:
        raise DataValidationError("Raw data has no rows")
    if data["encounter_id"].duplicated().any():
        raise DataValidationError("Duplicate encounter_id values in raw data")
    text = data.select_dtypes(include=["object", "string"])
    if text.apply(lambda s: s.str.strip().eq("?")).any().any():
        raise DataValidationError("'?' tokens were not converted to NaN on load")
    unexpected = set(data["readmitted"].dropna().unique()) - VALID_LABELS
    if data["readmitted"].isna().any() or unexpected:
        raise DataValidationError(f"Invalid readmission labels: {unexpected or 'missing values'}")
    return data


def validate_clean(df):
    """Checks after cleaning: one row per patient, no invalid records, a binary ~9% target."""
    if "weight" in df.columns:
        raise DataValidationError("'weight' should have been dropped")
    if not df["patient_nbr"].is_unique:
        raise DataValidationError("More than one encounter per patient remains")
    if df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_IDS).any():
        raise DataValidationError("Expired/hospice discharges remain")
    if set(df[TARGET].unique()) != {0, 1}:
        raise DataValidationError(f"Target '{TARGET}' is not binary")
    low, high = EXPECTED_POSITIVE_RATE
    rate = df[TARGET].mean()
    if not low <= rate <= high:
        raise DataValidationError(f"Positive rate {rate:.2%} is outside the expected {low:.0%}-{high:.0%}")
    return df


def validate_splits(splits):
    """Checks after splitting: no shared patients and the class balance preserved in every split."""
    names = list(splits)
    patients = {name: set(splits[name]["patient_nbr"]) for name in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if patients[a] & patients[b]:
                raise DataValidationError(f"Patient IDs overlap between {a} and {b}")
    rates = {name: splits[name][TARGET].mean() for name in names}
    if max(rates.values()) - min(rates.values()) > MAX_SPLIT_RATE_GAP:
        raise DataValidationError(f"Class balance differs across splits: {rates}")
    return splits


def to_xy(df):
    """Separate model inputs from the target; identifiers are never predictors."""
    return df.drop(columns=[*ID_COLUMNS, TARGET]), df[TARGET]


def prepare_splits(path=RAW_DATA_PATH, random_state=RANDOM_STATE):
    """Run the full Stage 2 preparation and return model-ready splits and feature lists."""
    raw = validate_raw(load_raw(path))
    df = engineer_features(validate_clean(clean(raw)))
    splits = validate_splits(split_data(df, random_state))
    X_train, y_train = to_xy(splits["train"])
    X_val, y_val = to_xy(splits["validation"])
    X_test, y_test = to_xy(splits["test"])
    numeric, categorical, specialty = feature_columns(X_train)
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "numeric": numeric, "categorical": categorical, "specialty": specialty,
    }
