import os

import pandas as pd
import pytest

from data_pipeline import DataPipeline, FEATURE_COLUMNS


def test_load_students_file():
    """Test loading because the pipeline must read CSV data and normalize column names before analysis."""
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
    """Test cleaning because later analysis and modeling should not receive rows with missing values."""
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
    """Test missing files because the project should give a clear error when the dataset is not installed."""
    pipeline = DataPipeline("missing_students.csv")

    with pytest.raises(FileNotFoundError):
        pipeline.load()


def test_get_features_and_target():
    """Test feature selection because the model should use focused inputs and keep GPA as a separate target."""
    test_file = "test_students_features.csv"
    data = pd.DataFrame({
        "Stress": [3, 7],
        "Anxiety": [2, 6],
        "Depression": [1, 5],
        "Motivation": [8, 4],
        "Concentration": [7, 3],
        "Time Management": [8, 4],
        "Self Discipline": [9, 3],
        "Procrastination Score": [2, 8],
        "Financial Stress": [3, 7],
        "Sleep Hours": [7, 5],
        "Late Night Frequency": [1, 4],
        "Screen Time": [5, 9],
        "Phone Unlocks Per Day": [40, 120],
        "Previous GPA": [8.2, 5.9],
        "GPA": [1.2, 0.7],
        "Performance Level": ["Low", "Low"],
    })
    data.to_csv(test_file, index=False)

    pipeline = DataPipeline(test_file)
    features, target = pipeline.get_features_and_target()

    assert list(features.columns) == FEATURE_COLUMNS
    assert target.name == "gpa"
    assert "performance_level" not in features.columns
    assert list(target) == [1.2, 0.7]
    os.remove(test_file)


def test_get_features_and_target_missing_column():
    """Test missing required columns because model training should stop when the dataset schema is incomplete."""
    test_file = "test_students_missing_feature.csv"
    data = pd.DataFrame({
        "Stress": [3],
        "GPA": [1.2],
    })
    data.to_csv(test_file, index=False)

    pipeline = DataPipeline(test_file)

    with pytest.raises(ValueError):
        pipeline.get_features_and_target()
    os.remove(test_file)
