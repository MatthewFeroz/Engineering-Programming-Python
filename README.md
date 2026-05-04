# Student Academic Performance Predictor

This project looks at college student habits and academic performance. The goal is to clean the student data, study which habits are related to GPA, and later build a model that can predict GPA from student information.

## Data Source

The dataset is from Kaggle:

College Students Habits & Performance

https://www.kaggle.com/datasets/sharmajicoder/college-students-habits-and-performance

Download the CSV file, rename it to `students.csv`, and place it in the `data` folder:

```text
data/students.csv
```

## Current Files

```text
data_pipeline.py        loads and cleans the student CSV file
test_data_pipeline.py   tests the data pipeline
requirements.txt        lists the packages used in the project
data/                   folder where students.csv should go
```

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the data pipeline:

```bash
python data_pipeline.py
```

Run the tests:

```bash
pytest -v test_data_pipeline.py
```
