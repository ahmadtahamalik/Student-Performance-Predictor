import pandas as pd

# Load the student performance dataset , sep =";" is used to separate for this specific dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

# Display the first 5 rows
print(data.head())

# Display the dataset dimensions
print("\nDataset shape:")
print(data.shape)

# Display all the column names
print("\nColumn names:")
print(data.columns.tolist())

# Display basic information about the dataset
print("\nDataset Information:")
print(data.info())

# Check for missing values 
print("\nMissing values:")
print(data.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(data.duplicated().sum())

# Display the data types of each coloumn 
print("\nData types:")
print(data.dtypes)

# Display basic statistical information
print("\nStatistical information: ")
print(data.describe())

# Define our target variable 
target = "G3"

print("\nTarget variable:")
print(target)
