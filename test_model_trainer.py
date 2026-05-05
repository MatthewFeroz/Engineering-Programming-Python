import os

import pandas as pd

from data_pipeline import FEATURE_COLUMNS, DataPipeline
from model_trainer import ModelTrainer


def make_model_test_file(test_file):
    """Create a small complete CSV for testing model training."""
    rows = []
    for i in range(10):
        row = {column: i + 1 for column in FEATURE_COLUMNS}
        row["previous_gpa"] = 5 + i * 0.2
        row["gpa"] = 0.5 + i * 0.05
        row["performance_level"] = "Low"
        rows.append(row)
    pd.DataFrame(rows).to_csv(test_file, index=False)


def test_model_trainer_trains_and_predicts():
    """Test training and prediction because the project needs a working GPA model."""
    test_file = "test_model_students.csv"
    make_model_test_file(test_file)

    trainer = ModelTrainer(DataPipeline(test_file))
    model = trainer.train()
    predictions = trainer.predict(trainer.X_test)

    assert model is trainer.model
    assert len(predictions) == len(trainer.X_test)
    assert predictions.name == "predicted_gpa"
    os.remove(test_file)


def test_model_trainer_len_counts_training_rows():
    """Test length because __len__ should report how many rows were used for training."""
    test_file = "test_model_len_students.csv"
    make_model_test_file(test_file)

    trainer = ModelTrainer(DataPipeline(test_file))

    assert len(trainer) == 8
    os.remove(test_file)
