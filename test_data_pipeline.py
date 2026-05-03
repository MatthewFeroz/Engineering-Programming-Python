import os

import pandas as pd
import pytest

from data_pipeline import DataPipeline


# Test file name must start with test_
def test_load_students_file():
    test_file = "test_students.csv"

    data = pd.DataFrame({
        "Study Hours": [3, 5],
        " Sleep Hours ": [7, 8],
        "GPA": [3.5, 3.8]
    })
    data.to_csv(test_file, index=False)

    pipeline = DataPipeline(test_file)
    loaded_data = pipeline.load()

    assert list(loaded_data.columns) == ["study_hours", "sleep_hours", "gpa"]
    assert loaded_data.shape[0] == 2
    os.remove(test_file)


def test_clean_students_file():
    test_file = "test_students_missing.csv"

    data = pd.DataFrame({
        "Study Hours": [3, 5, 2],
        "Sleep Hours": [7, None, 6],
        "GPA": [3.5, 3.8, 3.1]
    })
    data.to_csv(test_file, index=False)

    pipeline = DataPipeline(test_file)
    clean_data = pipeline.clean()

    assert clean_data.shape[0] == 2
    assert clean_data.isnull().sum().sum() == 0
    os.remove(test_file)


def test_missing_students_file():
    pipeline = DataPipeline("missing_students.csv")

    with pytest.raises(FileNotFoundError):
        pipeline.load()
