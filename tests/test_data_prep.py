import pandas as pd
import pytest

import data_prep as dp


def test_raw_load_converts_question_marks():
    raw = dp.load_raw()
    assert raw.shape == (101_766, 50)
    assert raw["weight"].isna().mean() > 0.95
    assert "None" in set(raw["A1Cresult"].unique())  # literal lab category preserved


def test_cleaned_data_meets_expectations():
    df = dp.validate_clean(dp.clean(dp.validate_raw(dp.load_raw())))
    assert len(df) == 69_973
    assert df["patient_nbr"].is_unique
    assert not df["discharge_disposition_id"].isin(dp.EXPIRED_HOSPICE_IDS).any()
    assert {"weight", "examide", "citoglipton", "readmitted"}.isdisjoint(df.columns)
    assert 0.08 <= df[dp.TARGET].mean() <= 0.10


def test_splits_are_stratified_and_patient_disjoint(splits):
    sizes = [len(splits["y_train"]), len(splits["y_val"]), len(splits["y_test"])]
    assert sizes == [48_981, 10_496, 10_496]
    rates = [splits[k].mean() for k in ("y_train", "y_val", "y_test")]
    assert max(rates) - min(rates) < 0.005
    assert not set(dp.ID_COLUMNS + [dp.TARGET]) & set(splits["X_train"].columns)


@pytest.mark.parametrize(
    "code, group",
    [("250.83", "Diabetes"), ("428", "Circulatory"), ("785", "Circulatory"), ("486", "Respiratory"),
     ("V57", "Other"), ("E909", "Other"), (None, "Missing"), ("996", "Injury"), ("174", "Neoplasms")],
)
def test_icd9_grouping(code, group):
    assert dp.icd9_group(code) == group


def test_engineered_features():
    row = pd.DataFrame([{
        "age": "[70-80)", "diag_1": "428", "diag_2": None, "diag_3": "250.01",
        "number_outpatient": 1, "number_emergency": 2, "number_inpatient": 3,
        "insulin": "Up", "metformin": "Down", "glipizide": "Steady",
        "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
    }])
    out = dp.engineer_features(row).iloc[0]
    assert out["age_midpoint"] == 75
    assert out["service_utilisation"] == 6
    assert out["num_med_changes"] == 2
    assert (out["diag_1_group"], out["diag_2_group"], out["diag_3_group"]) == ("Circulatory", "Missing", "Diabetes")


def test_preprocessor_fits_on_train_and_ignores_unknown_categories(splits):
    pre = dp.build_preprocessor(splits["numeric"], splits["categorical"], splits["specialty"]).fit(splits["X_train"])
    unseen = splits["X_val"].head(5).copy()
    unseen["race"] = "UnseenCategory"
    assert pre.transform(unseen).shape[0] == 5


def test_validation_rejects_bad_data():
    raw = dp.load_raw()
    with pytest.raises(dp.DataValidationError):
        dp.validate_raw(raw.drop(columns=["age"]))
    with pytest.raises(dp.DataValidationError):
        dp.validate_raw(raw.assign(readmitted="maybe"))
    cleaned = dp.clean(raw)
    with pytest.raises(dp.DataValidationError):
        dp.validate_clean(pd.concat([cleaned, cleaned.head(1)]))
