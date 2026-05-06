"""Train a simple GPA prediction model."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_pipeline import DataPipeline


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
        return split_data

    def train(self) -> LinearRegression:
        """Train the linear regression model and return it."""
        if self.X_train is None or self.y_train is None:
            self.prepare_data()
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    def evaluate(self) -> dict:
        """Score the model on the test set and print the results.
 
        Metrics:
            MAE  - average GPA prediction error (lower is better).
            RMSE - like MAE but punishes large mistakes more.
            R2   - value from 0 to 1 describing the fraction of GPA variation the model explains
                   (1.0 = perfect, 0.0 = no better than guessing the mean).
 
        Returns:
            Dict with keys "mae", "rmse", "r2".
        """
        if self.X_train is None:
            self.train()
 
        predictions = self.model.predict(self.X_test)
        errors      = predictions - self.y_test
 
        mae  = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((self.y_test - self.y_test.mean()) ** 2))
        r2   = 1 - ss_res / ss_tot
 
        print("Model evaluation on test set:")
        print(f"  MAE  (avg GPA error):        {mae:.3f}")
        print(f"  RMSE (penalises big misses): {rmse:.3f}")
        print(f"  R2   (1.0 = perfect):        {r2:.3f}")
 
        return {"mae": mae, "rmse": rmse, "r2": r2}

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

