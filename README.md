# Mental Health and Academic Performance Predictor

This project analyzes how college students' mental health, discipline, sleep, and digital-behavior habits relate to academic performance. Using a public Kaggle student-habits dataset, it studies factors such as stress, anxiety, motivation, concentration, procrastination, sleep hours, late-night frequency, and phone use, then trains a standardized linear regression model (`StandardScaler` + `LinearRegression`) that predicts a student's GPA and identifies the strongest predictors.

Course: AAI / CPE / EE 551 - Engineering Programming (Python), Stevens Institute of Technology, Spring 2026.

## Team Members


| Name              | Email                                               | Stevens ID |
| ----------------- | --------------------------------------------------- | ---------- |
| Matthew Feroz     | [mferoz@stevens.edu](mailto:mferoz@stevens.edu)     | 10454830   |
| Kinga Kurcaba     | [kkurcaba@stevens.edu](mailto:kkurcaba@stevens.edu) | 20023860   |
| Lilian Wierzbicki | [lwierzbi@stevens.edu](mailto:lwierzbi@stevens.edu) | 20010440   |


### Main Contributions

- **Matthew Feroz** - Created the data pipeline (`DataPipeline`), the model trainer (`ModelTrainer`), and supplemental helper functions in `utils.py`.
- **Kinga Kurcaba** - Implemented the pytest suite (`test_data_pipeline.py`, `test_model_trainer.py`), wrote `data_summary.py`, contributed to the data pipeline, and added project setup documentation.
- **Lilian Wierzbicki** - Built the exploratory data analysis module (`data_eda.py`), implemented the regression model integration, and authored the main notebook (`main.ipynb`).

All members participated in design discussions, code review, and final testing.

## Problem & Approach

**Problem.** Can self-reported mental-health, sleep, focus, and digital-behavior signals predict a student's GPA, and which of those signals matter most?

**Approach.** Load and clean a Kaggle student-habits dataset (`DataPipeline`), narrow to 13 modeling inputs (12 habit/well-being features plus `previous_gpa`), run EDA on correlations with GPA (`data_eda.py`), train a standardized linear regression on an 80/20 split (`ModelTrainer` — `StandardScaler` followed by `LinearRegression`), and evaluate with MAE, RMSE, R², and per-feature t-statistics. Standardizing the inputs puts the fitted coefficients on a comparable scale. `screen_time` is intentionally excluded from the modeling features because in the source data it is near-perfectly collinear with `late_night_frequency`. Including both produced ±85 cancelling OLS coefficients that exploded predictions on out-of-distribution inputs. A custom student profile produced `Predicted GPA = -74.331` before the fix. After dropping `screen_time`, every coefficient is bounded in `[-0.20, +0.20]` and predictions stay inside the dataset's GPA range. The walkthrough lives in `main.ipynb`. Reusable logic sits in the `.py` modules.

**Reproducibility.** The train/test split uses `random_state=67`, so re-runs produce the same splits and metrics.

## Dependencies / Libraries

- Python 3.12, 3.13, or 3.14
- pandas
- numpy
- scikit-learn
- matplotlib
- pytest
- jupyter

All dependencies are listed in `requirements.txt`.

## File / Module Structure

```text
Engineering-Programming-Python/
├── main.ipynb              # Main program: full walkthrough (load, EDA, train, evaluate, predict)
├── data_pipeline.py        # DataPipeline class: CSV load, clean, feature/target split
├── model_trainer.py        # ModelTrainer class: composes DataPipeline, trains StandardScaler + LinearRegression
├── data_eda.py             # run_eda(): correlation analysis and matplotlib charts
├── data_summary.py         # print_data_summary(): quick dataset overview
├── utils.py                # evaluate_model() and find_missing_columns() helpers
├── test_data_pipeline.py   # Pytest cases for DataPipeline (load, clean, errors, features)
├── test_model_trainer.py   # Pytest cases for ModelTrainer (train, predict, __len__)
├── requirements.txt        # Python dependencies
├── data/
│   └── students.csv        # Kaggle dataset (downloaded by the user, not committed)
└── README.md
```

## Data Source

The dataset is from Kaggle:

College Students Habits & Performance - [https://www.kaggle.com/datasets/sharmajicoder/college-students-habits-and-performance](https://www.kaggle.com/datasets/sharmajicoder/college-students-habits-and-performance)

Download the CSV file, rename it to `students.csv`, and place it in the `data/` folder:

```text
data/students.csv
```

The dataset includes `previous_gpa` on a 0-10 scale and `gpa` as the final target score. In the downloaded CSV, `gpa` ranges from about 0 to 2.01, so this project uses it as the dataset's provided academic performance score instead of rescaling it.

## Current Focus

The project focuses on these mental-health, discipline, sleep, and digital-behavior features:

```text
stress
anxiety
depression
motivation
concentration
time_management
self_discipline
procrastination_score
financial_stress
sleep_hours
late_night_frequency
phone_unlocks_per_day
previous_gpa
```

The target column is:

```text
gpa
```

The `performance_level` column is not used for modeling because every row in the downloaded file is labeled `Low`.

## Findings

**Model performance (test set, 20% holdout, `random_state=67`):**

| Metric | Value |
| ------ | ----- |
| R²     | 0.880 |
| MAE    | 0.081 |
| RMSE   | 0.102 |

The model is typically off by under a tenth of a GPA point on the project's `[0.000, 2.009]` GPA scale.

**Top predictors (ranked by absolute standardized coefficient):**

| Rank | Feature                | Coef    | Direction |
| ---- | ---------------------- | ------- | --------- |
| 1    | `previous_gpa`         | +0.1933 | up GPA    |
| 2    | `stress`               | -0.1121 | down GPA  |
| 3    | `motivation`           | +0.0951 | up GPA    |
| 4    | `phone_unlocks_per_day`| -0.0667 | down GPA  |
| 5    | `late_night_frequency` | +0.0612 | up GPA    |

`previous_gpa` is the dominant signal. On top of it, `stress`, `procrastination_score`, and `phone_unlocks_per_day` are statistically significant down-pressures on GPA. `sleep_hours`, `motivation`, and `concentration` are significant up-pressures. `anxiety`, `depression`, and `financial_stress` were not significant after controlling for the others.

**Sanity check on a custom student input** (above-average profile, `previous_gpa=7.5`):

```text
Predicted GPA : 0.917
Dataset range : 0.000 - 2.009
```

The prediction sits between the dataset mean (0.832) and the upper bound, which is the expected, sensible result.

**Limitations.**

- Self-reported features carry rater bias. The linear model also misses interactions between features.
- `time_management` and `self_discipline` returned identical coefficients, indicating perfect collinearity between those two columns in the source data — one could be dropped with no information loss.
- `gpa` spans only ~0–2 in the source CSV and every row is labeled `Low`, so the model learns to rank within a narrow band rather than across a full 0-4 GPA range.
- A regularized model (Ridge / Lasso) would handle any residual feature redundancy more robustly than plain OLS.

**Takeaway.** Habit and well-being signals carry significant, directionally sensible predictive power on top of prior GPA — supporting the case for using them in early-warning systems rather than waiting for grades to slip.

## How to Run

1. Clone the repository and create a virtual environment using Python 3.12, 3.13, or 3.14.
2. Install the required packages:
  ```bash
   pip install -r requirements.txt
  ```
3. Download the dataset from Kaggle (link above), rename the CSV to `students.csv`, and place it at `data/students.csv`.
4. Launch Jupyter and run the main notebook end-to-end:
  ```bash
   jupyter notebook main.ipynb
  ```
   Then choose **Cell → Run All** in the Jupyter UI.
5. Run the full test suite:
  ```bash
   pytest -v
  ```

