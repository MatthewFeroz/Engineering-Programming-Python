# Student Academic Performance Predictor

A data analysis and machine learning project that identifies the lifestyle and behavioral factors most strongly correlated with college student academic success, and provides a predictive model for estimating GPA based on user inputs.

## Project Overview

College students balance coursework, jobs, social life, sleep, and physical health within a limited number of hours per day. This project uses data analysis and machine learning to answer:

- Which lifestyle factors most strongly predict academic performance?
- Which features carry the most (and least) predictive weight?
- Are there nonlinear interaction effects where two factors compound to hurt performance more than expected?

The final deliverable is a trained predictive model that lets a user input their own lifestyle factors and receive a predicted GPA.

## Data Source

[College Students Habits & Performance (Kaggle)](https://www.kaggle.com/datasets/sharmajicoder/college-students-habits-and-performance) — download `students.csv` and place it at `data/students.csv`. The dataset contains ~500 rows of self-reported daily-activity minutes and a previous-semester SGPA target.

## Approach

1. **Data Pipeline** — Load a publicly available student-performance dataset, clean it, standardize numeric features, and split into train/test sets.
2. **Exploratory Data Analysis (EDA)** — Visualize distributions, correlations, and feature relationships.
3. **Modeling** — Train and compare several models:
   - Linear/Regularized Regression (scikit-learn)
   - Random Forest (scikit-learn) for feature-importance analysis
   - Neural Network (PyTorch) for capturing nonlinear interactions
4. **Evaluation** — Compare models using MAE, RMSE, and R².
5. **Prediction Interface** — Accept user input and return a predicted GPA.

## Dependencies

- Python 3.12+
- pandas
- numpy
- scikit-learn
- pytorch
- matplotlib
- pytest
- jupyter

Install with:

```bash
pip install -r requirements.txt
```

## File Structure

```text
.
├── main.ipynb              # Main notebook walking through the full pipeline
├── data_pipeline.py        # DataPipeline class: load, clean, split, batch generator
├── model_trainer.py        # ModelTrainer class: training, evaluation, prediction
├── utils.py                # Helper functions (evaluate_model, predict_gpa)
├── tests/
│   └── test_pipeline.py    # Pytest cases
├── data/
│   └── students.csv        # Dataset
├── requirements.txt
└── README.md
```

## Implementation Highlights

### Part 1 Requirements

| Requirement | Implementation |
|---|---|
| Two classes (composition) | `DataPipeline` (load/clean/split) and `ModelTrainer` (holds a `DataPipeline`) |
| Two functions | `evaluate_model()` returns MAE/RMSE/R²; `predict_gpa()` returns GPA estimate |
| Advanced libraries | pandas, scikit-learn, PyTorch |
| Exception handling | `FileNotFoundError` on CSV load; `ValueError` on invalid user input |
| Pytest | Pipeline output-shape test; prediction-range test |
| Data I/O | CSV read via pandas |
| Loops & conditionals | `for` in training epochs/EDA; `if` in validation/model selection |
| Mutable / immutable | `list`, `dict` / `str`, `tuple` |
| Operator overloads | `__str__`, `__len__`, `__eq__` on `ModelTrainer` |

### Part 2 Requirements

| Component | Where it Appears |
|---|---|
| `enumerate` / `lambda` | Training-loop progress logging; `lambda` inside pandas `apply` |
| List comprehension | `numeric_cols = [c for c in df.columns if df[c].dtype != 'object']` |
| Generator function | `DataPipeline.batch_generator(batch_size)` yields training batches |
| `__name__` guard | `if __name__ == '__main__':` in `data_pipeline.py` and `model_trainer.py` |