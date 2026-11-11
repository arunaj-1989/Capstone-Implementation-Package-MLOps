import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import data_prep as dp


@pytest.fixture(scope="session")
def splits():
    return dp.prepare_splits()


@pytest.fixture(scope="session")
def raw_encounter():
    """A valid /predict payload built from a real cleaned encounter."""
    row = dp.clean(dp.load_raw()).iloc[0]
    payload = {}
    for column, value in row.items():
        if column in dp.ID_COLUMNS or column == dp.TARGET:
            continue
        if hasattr(value, "item"):
            value = value.item()
        payload[column] = None if isinstance(value, float) and value != value else value
    return payload
