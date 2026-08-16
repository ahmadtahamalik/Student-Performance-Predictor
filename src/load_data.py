import pandas as pd
import numpy as np

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

# Define X and Y 

X = data.drop("G3", axis=1)
Y = data["G3"]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# We want the model to make predictions for students whose G3 it hasn't seen during training.

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape", X_test.shape)
print("Y_train shape", Y_train.shape)
print("Y_test shape", Y_test.shape)

# Separate numerical and categorical features to prepare them for ML preprocessing

categorical_features = X.select_dtypes(include="str").columns.tolist()
numerical_features = X.select_dtypes(include="number").columns.tolist()

print("Categorical columns:")
print(categorical_features)

print("\nNumerical columns:")
print(numerical_features)

# Creating a preprocessing pipeline to encode categorical features while keeping numerical features

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Create preprocessing pipeline for numerical and categorical features

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", StandardScaler(), numerical_features)
    ]
)

# Fit preprocessing on training data and transform both training and test data

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("X_train_processed:", X_train_processed.shape)
print("X_test_processed:", X_test_processed.shape)

# Get the names of all the features before preprocessing
feature_names = preprocessor.get_feature_names_out()

print("Number of processed feature:", len(feature_names))
print("\nProcessed feature names:")
print(feature_names)

# Check the processed training and test data 

print("Processed training data shape:",X_train_processed.shape)
print("Processed test data shape:", X_test_processed.shape)

print("\n Data type of processed training data:")
print(type(X_train_processed))

print("\nContains NaN values:")
print(np.isnan(X_train_processed.toarray()).any() if hasattr(X_train_processed, "toarray") else np.isnan(X_train_processed).any())

# Final Check before model training 

print("Final training data shape:", X_train_processed.shape)
print("Final test data shape:", X_test_processed.shape)

print("\nTraining target shape:",Y_train.shape)
print("Test target shape:", Y_test.shape)

print("\n Number of Processed features:",len(feature_names))

print("\nFirst 10 processed feature names:")
print(feature_names[:10])
