"""Print a basic summary of the student dataset."""

from data_pipeline import DataPipeline


def print_data_summary():
    """Load the data and print basic information about it."""
    pipeline = DataPipeline()
    data = pipeline.clean()

    print("Student Data Summary")
    print("--------------------")
    print("Rows:", data.shape[0])
    print("Columns:", data.shape[1])

    print("\nColumn names:")
    for column in data.columns:
        print(column)

    print("\nMissing values:")
    print(data.isnull().sum())

    if "gpa" in data.columns:
        print("\nGPA statistics:")
        print("Lowest GPA:", data["gpa"].min())
        print("Highest GPA:", data["gpa"].max())
        print("Average GPA:", data["gpa"].mean())


if __name__ == "__main__":
    print_data_summary()
