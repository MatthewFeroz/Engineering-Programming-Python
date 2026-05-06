"""Train a simple GPA prediction model."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_pipeline import DataPipeline
from utils import evaluate_model

class ModelTrainer:
    """Trains a linear regression model using data from a DataPipeline."""

    def __init__(self, pipeline: DataPipeline):
        """Store the pipeline and create the regression model."""
        self.pipeline = pipeline
        self.model = LinearRegression()
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

    def prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the focused features and GPA target into training and test data."""
        features, target = self.pipeline.get_features_and_target()
        split_data = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=67,
        )
        self.X_train, self.X_test, self.y_train, self.y_test = split_data
        return tuple(split_data)

    def train(self) -> LinearRegression:
        """Train the linear regression model and return it."""
        if self.X_train is None or self.y_train is None:
            self.prepare_data()
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    def evaluate(self) -> dict:
        """Score the model on the test set and print the results.
 
        Metrics:
            MAE  - average GPA prediction error
            RMSE - average GPA error but punishes large mistakes
            R2   - fraction of GPA variation explained by model
 
        Returns:
            Dict with keys "mae", "rmse", "r2".
            Also prints a coefficient table with t-stats and p-values.
        """
        if self.X_train is None:
            self.train()
 
        coefficient_table, metrics = evaluate_model(self.model, self.X_test, self.y_test)
 
        print("Coefficient table:")
        print(coefficient_table.to_string(index=False))
        print()
        print("Model evaluation on test set:")
        print(f"  MAE  (avg GPA error):        {metrics['mae']:.3f}")
        print(f"  RMSE (penalises big misses): {metrics['rmse']:.3f}")
        print(f"  R2   (1.0 = perfect):        {metrics['r2']:.3f}")
 
        return metrics

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict GPA scores for prepared feature rows."""
        if self.X_train is None:
            self.train()
        predictions = self.model.predict(features)
        return pd.Series(predictions, index=features.index, name="predicted_gpa")

    def __len__(self) -> int:
        """Return the number of training rows."""
        if self.X_train is None:
            self.prepare_data()
        return len(self.X_train)

    def __str__(self) -> str:
        """Return a short summary of the trainer."""
        row_count = len(self)
        return f"ModelTrainer(model=LinearRegression, training_rows={row_count})"




if __name__ == "__main__":
    trainer = ModelTrainer(DataPipeline())
    trainer.train()
    trainer.evaluate()
    print(trainer)

