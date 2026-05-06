# Mental Health and Academic Performance Predictor

This project analyzes how college students' mental health, discipline, sleep, and screen-time habits relate to academic performance. Using a public Kaggle student habits dataset, the project studies factors such as stress, anxiety, motivation, concentration, procrastination, sleep hours, screen time, and phone use. The final program will train a machine learning model that predicts a student's academic performance score and identifies the strongest predictors.

# Team Members

Matthew Feroz  
Main contributions:  
Kinga Kurcaba  
Main contributions:  
Lilian Wierzbicki (lwierzbi@stevens.edu 20010440)  
Main contributions: Created the EDA, some of regression model, and the main notebook.   

## Data Source

The dataset is from Kaggle:

College Students Habits & Performance

https://www.kaggle.com/datasets/sharmajicoder/college-students-habits-and-performance

Download the CSV file, rename it to `students.csv`, and place it in the `data` folder:

```text
data/students.csv
```

The dataset includes `previous_gpa` on a 0-10 scale and `gpa` as the final target score. In the downloaded CSV, `gpa` ranges from about 0 to 2.01, so this project treats it as the dataset's provided academic performance score instead of rescaling it.

## Current Focus

The project focuses on these mental health, discipline, and screen-time features:

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
