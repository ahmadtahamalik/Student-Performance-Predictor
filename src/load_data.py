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

# Display each column with its position 
print("\nFeatures in the dataset:")

for number, coloumn in enumerate(data.columns, start= 1):
    print(f"{number}.{coloumn}")

# Separate features(X) and target(Y)

X = data.drop(columns=["G3"])
Y = data["G3"]

print("\nFeatures (X) shape:")
print(X.shape)

print("\nTarget (Y) shape:")
print(Y.shape)

print("\nTarget variable:")
print(Y.name)

# Analyze the final grade (G3)

print("\nG3 statistics:")
print(data["G3"].describe())

print("\nG3 value counts:")
print(data["G3"].value_counts().sort_index())


# Check correlations with the final grade

print("\nCorrealation with G3:")
print("G1:", data["G1"].corr(data["G3"]))
print("G2:", data["G2"].corr(data["G3"]))


# Find correlation between  numerical features and G3
print("\nNumerical feature correlations with G3:")
numeric_data = data.select_dtypes(include="number")
correlation = numeric_data.corr()["G3"].sort_values(ascending=False)
print(correlation)

# Remove G3 and Compare
correlation = data.select_dtypes(include="number").corr()["G3"]
correlation_without_G3 = correlation.drop("G3")
print(correlation_without_G3.sort_values(ascending=False))

# Graph the plot of the correlations (G2 VS G3)
import matplotlib.pyplot as plt

plt.scatter(data["G2"], data["G3"])

plt.xlabel("G2")
plt.ylabel("G3")
plt.title("G2 vs G3")

plt.show()

# Graph the plot of the correlation (G1 VS G3)
import matplotlib.pyplot as plt 


plt.scatter(data["G1"], data["G3"])

plt.xlabel("G1")
plt.ylabel("G3")
plt.title("G1 VS G3")

plt.show()

# Compare the other feature using abs
correlation_without_G3.abs().sort_values(ascending=False) 

# Compare non numerical features with G3 
# data.groupby("sex")["G3"].mean()
# print(data.groupby("sex")["G3"].mean())

# Plot a bar chart for the data 
# data.groupby("sex")["G3"].mean().plot(kind="bar")
# plt.xlabel("Sex")
# plt.ylabel("Average G3")
# plt.title("Average G3 by Sex")
# plt.show()

# Compare non numerical features all at once

categorical_features = data.select_dtypes(include="str").columns

for feature in categorical_features:
    print("\n",feature)
    print(data.groupby(feature)["G3"].mean())

print(data["higher"].value_counts())
for feature in categorical_features:
    print(f"\n--- {feature} ---")
    print( data.groupby(feature)["G3"].agg(["count", "mean"]).sort_values("mean", ascending=False))

# Make a bar chart 
data.groupby("higher")["G3"].mean().plot(kind="bar")

plt.xlabel("Higher Education")
plt.ylabel("Average G3")
plt.title("Average G3 by Higher Education Intention")

plt.show()

# Now plot MJob data 

data.groupby("Mjob")["G3"].mean().plot(kind="bar")

plt.xlabel("Mother's Job")
plt.ylabel("Average G3")
plt.title("Average G3 by Mother's Job")

plt.show()

# Now plt FJob (Father Job) data
data.groupby("Fjob")["G3"].mean().plot(kind="bar")

plt.xlabel("Father's Job")
plt.ylabel("Average G3")
plt.title("Average G3 by Father's Job")

plt.show()

# Do students who chose the school for different reasons have noticeably different average G3 scores?
data.groupby("reason")["G3"].mean().plot(kind="bar")

plt.xlabel("Reason for Choosing School")
plt.ylabel("Average G3")
plt.title("Average G3 by Reason for Choosing School")

plt.show()

# If it is, guardian becomes another candidate for further testing.
data.groupby("guardian")["G3"].mean().plot(kind="bar")

plt.xlabel("Guardian")
plt.ylabel("Average G3")
plt.title("Average G3 by Guardian")

plt.show()

# Do students with and without internet access have noticeably different average G3 scores?

data.groupby("internet")["G3"].mean().plot(kind="bar")

plt.xlabel("Internet Access")
plt.ylabel("Average G3")
plt.title("Average G3 by Internet Access")

plt.show()

# Do students in the two romantic categories have noticeably different average G3 scores?

data.groupby("romantic")["G3"].mean().plot(kind="bar")

plt.xlabel("Romantic Relationship")
plt.ylabel("Average G3")
plt.title("Average G3 by Romantic Relationship")

plt.show()

# Do students who receive extra educational support have a different average G3 than students who don't?
data.groupby("schoolsup")["G3"].mean().plot(kind="bar")

plt.xlabel("Extra Educational Support")
plt.ylabel("Average G3")
plt.title("Average G3 by School Support")

plt.show()

# Do students who take paid extra classes have a different average G3 from those who don't?
data.groupby("paid")["G3"].mean().plot(kind="bar")

plt.xlabel("Extra Paid Classes")
plt.ylabel("Average G3")
plt.title("Average G3 by Extra Paid Classes")

plt.show()