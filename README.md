# 🎓 Student Performance Predictor

A machine-learning application that predicts a student's final academic performance using demographic, academic, family, social, and lifestyle information.

The project covers the complete machine-learning workflow, from data exploration and preprocessing to model training, evaluation, and deployment through an interactive Streamlit application.

---

## 📌 Project Overview

The objective of this project was to investigate whether information about a student's academic background, family environment, school situation, and lifestyle can be used to estimate their final academic performance.

The project follows a complete machine-learning workflow:

- Data collection
- Data cleaning and exploration
- Exploratory data analysis
- Feature preprocessing
- Model training
- Model evaluation
- Model selection
- Model saving
- Application development
- Interactive prediction
- Deployment preparation

The final application allows users to enter student information and receive an estimated final grade.

---

## 🎯 Project Objectives

The main objectives of the project are:

- Understand and explore student-performance data
- Identify patterns and relationships between student characteristics and academic performance
- Analyse numerical and categorical features
- Prepare data for machine learning
- Train and compare machine-learning models
- Predict a student's final academic grade
- Determine whether the predicted grade indicates a pass or fail
- Build an interactive user interface using Streamlit
- Create a reproducible and well-documented machine-learning project

---

## 🤖 Machine-Learning Approach

### 📈 Regression

The main prediction task is **regression**.

The model estimates the student's final grade:

**G3 — Final Grade**

The predicted grade is presented to the user through the Streamlit application.

The application also provides a simple interpretation:

- **G3 ≥ 10 → PASS**
- **G3 < 10 → FAIL**

This threshold follows the grading scale used in the original dataset.

---

## 🧠 Machine-Learning Model

The final Streamlit application uses a:

**Random Forest Regressor**

The trained model is stored in the `models/` directory together with the preprocessing pipeline.

The preprocessing pipeline ensures that the input provided through the application is transformed in the same way as the training data before being passed to the model.

---

## 📊 Model Performance

During model development, different regression approaches were evaluated.

The final model achieved approximately:

- **MAE:** 1.64
- **RMSE:** 2.37
- **R²:** 0.73

These metrics indicate that the model is able to capture a substantial portion of the variation in final student grades.

However, the predictions should be interpreted as estimates rather than guaranteed outcomes.

---

## 📚 Dataset

This project uses the **Student Performance dataset** from the UCI Machine Learning Repository.

The dataset contains information about students':

- Academic performance
- Demographic characteristics
- Family background
- School information
- Social activities
- Lifestyle
- Study habits

The main target variable is:

**G3 — Final Grade**

The original dataset was collected from two Portuguese secondary schools.

Dataset source:

https://archive.ics.uci.edu/dataset/320/student+performance

---

## ⚠️ Dataset Limitations

The model was trained using data from the original UCI Student Performance dataset.

Therefore, the model has learned patterns from the students represented in that dataset.

Predictions may not generalize equally well to:

- Students from other schools
- Students from other countries
- Different education systems
- Different grading systems
- Populations with different demographic or social characteristics

The application should therefore be considered an **educational machine-learning project and prediction tool**, rather than a definitive assessment of a student's future academic performance.

---

## 🖥️ Streamlit Application

The project includes an interactive web application built with **Streamlit**.

Users can enter information such as:

### Student Information

- School
- Sex
- Age
- Address
- Family size
- Parent status
- Mother's education
- Father's education
- Mother's occupation
- Father's occupation
- Reason for choosing school
- Guardian

### Academic & Support Information

- Travel time
- Study time
- Previous failures
- School support
- Family support
- Extra paid classes
- Extra-curricular activities
- Nursery attendance
- Desire for higher education
- Internet access
- Romantic relationship
- Family relationship quality

### Lifestyle Information

- Free time
- Going out
- Weekday alcohol consumption
- Weekend alcohol consumption
- Health status
- Absences
- First-period grade (G1)

After entering the information, the application generates an estimated final G3 grade.

---

## 🔎 Understanding Dataset Codes

Some variables in the original UCI dataset are represented using numerical or abbreviated categorical values.

For example:

### Mother's / Father's Education

- `0` — No education
- `1` — Primary education
- `2` — 5th–9th grade
- `3` — Secondary education
- `4` — Higher education

### Address

- `U` — Urban
- `R` — Rural

### Family Size

- `GT3` — Greater than 3 people
- `LE3` — 3 or fewer people

### Parent Status

- `T` — Living together
- `A` — Living apart

These values are retained because they correspond to the original dataset representation required by the trained preprocessing pipeline and model.

The Streamlit application provides explanations for these values to make the interface easier to understand.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Joblib
- Git
- GitHub

---

## 📁 Project Structure

```text
student-performance-predictor/
│
├── data/
│   └── Dataset files
│
├── images/
│   └── Charts and application screenshots
│
├── models/
│   ├── preprocessor.pkl
│   └── random_forest_model.pkl
│
├── notebooks/
│   └── Machine-learning experiments and analysis
│
├── src/
│   ├── load_data.py
│   ├── predict.py
│   └── text.txt
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md


▶️ How to Run the Application

To run the Student Performance Predictor locally, follow these steps:

1. Clone the repository:
git clone YOUR_GITHUB_REPOSITORY_URL
cd student-performance-predictor.

2. Install the required dependencies:
pip install -r requirements.txt

3. Start the Streamlit application:
streamlit run app.py

The application will open in your web browser. If it does not open automatically, Streamlit will provide a local URL such as:

http://localhost:8501

📌 Important

The application should be launched using:

streamlit run app.py

rather than:

python app.py

because the application is built using Streamlit.

Make sure the saved machine-learning files are also present in the models/ directory:

models/
├── preprocessor.pkl
└── random_forest_model.pkl

That's all the user needs to know to run your project locally.