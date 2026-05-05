"""EDA (Exploratory Data Analysis): explores which student habits are most
related to GPA, and provides visuals."""


import matplotlib.pyplot as plt
import pandas as pd

from data_pipeline import DataPipeline


# Load Data
pipeline = DataPipeline()
X, y = pipeline.get_features_and_target()

# Keep only numeric columns, turn X, y into 1 dataframe.
df = pd.concat([X, y], axis=1)
df = df.select_dtypes(include="number")


# Prints correlations between variables and GPA in descending order

print("Correlation with GPA (strongest first):\n")
correlations = df.corr()["gpa"].drop("gpa").sort_values(key=abs, ascending=False)
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
plt.savefig("correlations.png")
print("\nSaved correlations.png")


# Graph of GPA Distributions
plt.figure()
df["gpa"].plot(kind="hist", bins=30, edgecolor="black")
plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.ylabel("Number of students")
plt.tight_layout()
plt.savefig("gpa_distribution.png")
print("Saved gpa_distribution.png")


