# Mental Health and Academic Performance Predictor

This project analyzes how college students' mental health, discipline, sleep, and screen-time habits relate to academic performance. Using a public Kaggle student-habits dataset, it studies factors such as stress, anxiety, motivation, concentration, procrastination, sleep hours, screen time, and phone use, then trains a linear regression model that predicts a student's GPA and identifies the strongest predictors.

Course: AAI / CPE / EE 551 - Engineering Programming (Python), Stevens Institute of Technology, Spring 2026.

## Team Members

| Name | Email | Stevens ID |
| --- | --- | --- |
| Matthew Feroz | mferoz@stevens.edu | 10454830 |
| Kinga Kurcaba | kkurcaba@stevens.edu | 20023860 |
| Lilian Wierzbicki | lwierzbi@stevens.edu | 20010440 |

### Main Contributions

- **Matthew Feroz** - Created the data pipeline (`DataPipeline`), the model trainer (`ModelTrainer`), and supplemental helper functions in `utils.py`.
- **Kinga Kurcaba** - Implemented the pytest suite (`test_data_pipeline.py`, `test_model_trainer.py`), wrote `data_summary.py`, contributed to the data pipeline, and added project setup documentation.
- **Lilian Wierzbicki** - Built the exploratory data analysis module (`data_eda.py`), implemented the regression model integration, and authored the main notebook (`main.ipynb`).

All members participated in design discussions, code review, and final testing.

## Project Description

### Overview

College students' academic outcomes are shaped by mental health, sleep, focus, self-discipline, and digital behavior. This project loads and cleans a public Kaggle student-habits dataset, runs exploratory data analysis on the relationship between those factors and GPA, trains a linear regression model on a focused set of mental-health, discipline, sleep, screen-time, and prior-GPA features, evaluates it with coefficient statistics and standard regression metrics (MAE, RMSE, R²), and predicts a GPA score for a custom student profile.

The walkthrough lives in `main.ipynb`. Reusable logic is split across small `.py` modules so the notebook stays focused on narration and results.

### Dependencies / Libraries

- Python 3.12, 3.13, or 3.14
- pandas
- numpy
- scikit-learn
- matplotlib
- pytest
- jupyter

All dependencies are listed in `requirements.txt`.

### File / Module Structure

```text
Engineering-Programming-Python/
├── main.ipynb              # Main program: full walkthrough (load, EDA, train, evaluate, predict)
├── data_pipeline.py        # DataPipeline class: CSV load, clean, feature/target split
├── model_trainer.py        # ModelTrainer class: composes DataPipeline, trains LinearRegression
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

College Students Habits & Performance - https://www.kaggle.com/datasets/sharmajicoder/college-students-habits-and-performance

Download the CSV file, rename it to `students.csv`, and place it in the `data/` folder:

```text
data/students.csv
```

The dataset includes `previous_gpa` on a 0-10 scale and `gpa` as the final target score. In the downloaded CSV, `gpa` ranges from about 0 to 2.01, so this project uses it as the dataset's provided academic performance score instead of rescaling it.

## Current Focus

The project focuses on these mental-health, discipline, sleep, and screen-time features:

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
screen_time
phone_unlocks_per_day
previous_gpa
```

The target column is:

```text
gpa
```

The `performance_level` column is not used for modeling because every row in the downloaded file is labeled `Low`.

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
