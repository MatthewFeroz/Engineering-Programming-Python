"""Data pipeline: load and clean the student-habits CSV."""

from __future__ import annotations

import pandas as pd

DATA_URL = "https://www.kaggle.com/datasets/sharmajicoder/college-students-habits-and-performance"

# These columns match the project's narrowed focus on mental health,
# discipline, sleep, screen time, and prior academic performance.
FEATURE_COLUMNS = [
    "stress",
    "anxiety",
    "depression",
    "motivation",
    "concentration",
    "time_management",
    "self_discipline",
    "procrastination_score",
    "financial_stress",
    "sleep_hours",
    "late_night_frequency",
    "screen_time",
    "phone_unlocks_per_day",
    "previous_gpa",
]
TARGET_COLUMN = "gpa"


class DataPipeline:
    """Loads and cleans the student-habits dataset."""

    def __init__(self, csv_path: str = "data/students.csv"):
        """Store the CSV path."""
        self.csv_path = csv_path
        self.df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Read the CSV and lowercase its column names.

        Raises:
            FileNotFoundError: If ``csv_path`` does not exist.
        """
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}. "
                f"Download from {DATA_URL} and place the CSV at {self.csv_path}."
            ) from e
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        self.df = df
        return df

    def clean(self) -> pd.DataFrame:
        """Drop rows with any missing values. Loads the CSV first if needed."""
        if self.df is None:
            self.load()
        self.df = self.df.dropna().copy()
        return self.df

    def get_features_and_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return focused input features and the GPA target."""
        if self.df is None:
            self.clean()

        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing_columns = [
            column for column in required_columns if column not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Keep inputs and target separate so model training cannot leak GPA
        # into the feature matrix.
        features = self.df.loc[:, FEATURE_COLUMNS].copy()
        target = self.df.loc[:, TARGET_COLUMN].copy()
        return features, target


if __name__ == "__main__":
    pipe = DataPipeline()
    X, y = pipe.get_features_and_target()
    print(f"rows={len(X)}, features={list(X.columns)}, target={y.name}")
