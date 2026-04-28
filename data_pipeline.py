"""Data pipeline: load and clean the student-habits CSV."""

from __future__ import annotations

import pandas as pd

DATA_URL = "https://www.kaggle.com/datasets/sharmajicoder/college-students-habits-and-performance"


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


if __name__ == "__main__":
    pipe = DataPipeline()
    pipe.clean()
    print(f"rows={len(pipe.df)}, cols={list(pipe.df.columns)}")
