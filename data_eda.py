"""EDA (Exploratory Data Analysis): explores which student habits are most
related to GPA, and provides visuals."""


import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from data_pipeline import DataPipeline


def run_eda(X, y):
    """
    Print feature correlations with GPA and display charts.
    """
    # Combine into one numeric DataFrame
    df = pd.concat([X, y], axis=1).select_dtypes(include="number")
    correlations = df.corr()["gpa"].drop("gpa").sort_values(key=abs, ascending=False)
 
    # Print correlations
    print("Correlation with GPA (strongest first):\n")
    for feature, value in correlations.items():
        print(f"  {feature:<35} {value:+.3f}")
    # Bar chart of correlations
    plt.figure(figsize=(10, 5))
    correlations.plot(kind="bar")
    plt.title("Feature Correlations with GPA")
    plt.xlabel("Feature")
    plt.ylabel("Correlation")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
 
    # GPA distribution
    plt.figure()
    df["gpa"].plot(kind="hist", bins=30, edgecolor="black")
    plt.title("GPA Distribution")
    plt.xlabel("GPA")
    plt.ylabel("Number of students")
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    pipeline = DataPipeline()
    X, y = pipeline.get_features_and_target()
    run_eda(X, y)