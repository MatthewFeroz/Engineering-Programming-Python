"""Data pipeline: load, clean, and prepare student-habits data for modeling.

Provides the :class:`DataPipeline` class which handles the full journey from a
raw CSV to model-ready feature / target arrays.  The pipeline enforces a fixed
schema of mental-health, discipline, sleep, and digital-behavior columns that
the downstream ``ModelTrainer`` depends on.

Typical usage::

    pipe = DataPipeline("data/students.csv")
    features, target = pipe.get_features_and_target()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import find_missing_columns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_URL: str = (
    "https://www.kaggle.com/datasets/sharmajicoder/"
    "college-students-habits-and-performance"
)

# The 13 modeling inputs capture mental health, discipline, sleep, and
# digital-behavior signals alongside prior academic performance.
#
# ``screen_time`` is intentionally excluded: it is near-perfectly collinear
# with ``late_night_frequency`` (the two combined produce +/-85 cancelling OLS
# coefficients, making out-of-distribution predictions explode).
# ``phone_unlocks_per_day`` already captures phone-use intensity while
# ``late_night_frequency`` captures a distinct timing/sleep signal, so
# dropping ``screen_time`` removes the redundant axis without losing unique
# information.
FEATURE_COLUMNS: list[str] = [
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
    "phone_unlocks_per_day",
    "previous_gpa",
]

TARGET_COLUMN: str = "gpa"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DataPipeline:
    """Loads, cleans, and prepares the student-habits dataset for modeling.

    The three public methods form a sequential pipeline::

        load  ->  clean  ->  get_features_and_target

    Each step lazily triggers all preceding steps that have not yet run, so
    callers can jump straight to :meth:`get_features_and_target` for
    convenience or invoke each step individually for finer control.
    """

    def __init__(self, csv_path: str | Path = "data/students.csv") -> None:
        """Store the path to the CSV file.

        Parameters
        ----------
        csv_path:
            Filesystem path (relative or absolute) to the raw student-habits
            CSV.  Defaults to ``data/students.csv``.
        """
        self.csv_path = csv_path
        self.df: pd.DataFrame | None = None

    # -- Step 1: Load --------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Read the CSV and normalize column names to lowercase snake_case.

        Raises
        ------
        FileNotFoundError
            If ``csv_path`` does not point to an existing file.
        """
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}. "
                f"Download from {DATA_URL} and place the CSV at {self.csv_path}."
            ) from e

        # Normalize headers: strip surrounding whitespace, lowercase, and
        # replace inner spaces with underscores so downstream code can use
        # simple dot-access and consistent column references.
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        self.df = df
        return df

    # -- Step 2: Clean -------------------------------------------------------

    def clean(self) -> pd.DataFrame:
        """Drop rows with any missing values.

        Automatically calls :meth:`load` first if the data has not yet been
        loaded.
        """
        if self.df is None:
            self.load()

        self.df = self.df.dropna().copy()
        return self.df

    # -- Step 3: Feature / target extraction ---------------------------------

    def get_features_and_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return the focused feature matrix and GPA target series.

        Automatically calls :meth:`clean` (and transitively :meth:`load`) if
        the data has not been prepared yet.

        Raises
        ------
        ValueError
            If the loaded dataset is missing any of the required columns
            defined in :data:`FEATURE_COLUMNS` or :data:`TARGET_COLUMN`.
        """
        if self.df is None:
            self.clean()

        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing_columns = find_missing_columns(self.df.columns, required_columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Separate inputs from target to prevent GPA from leaking into the
        # feature matrix during model training.
        features = self.df.loc[:, FEATURE_COLUMNS].copy()
        target = self.df.loc[:, TARGET_COLUMN].copy()
        return features, target

    # -- Dunder helpers ------------------------------------------------------

    def __repr__(self) -> str:
        row_info = f", rows={len(self.df)}" if self.df is not None else ""
        return f"DataPipeline(csv_path='{self.csv_path}'{row_info})"


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipe = DataPipeline()
    features, target = pipe.get_features_and_target()
    print(f"rows={len(features)}, features={list(features.columns)}, target={target.name}")
