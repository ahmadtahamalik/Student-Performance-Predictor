import streamlit as st
import pandas as pd
import joblib

# Load Model and Preprocessing Pipeline
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/random_forest_model.pkl")


# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# Header
st.title("🎓 Student Performance Predictor")

st.markdown(
    """
    Predict a student's **final G3 grade** using academic, demographic,
    family, and lifestyle information.

    This application uses a **Random Forest machine learning model**
    trained on student performance data.
    """
)

st.divider()

# Student Information
st.header("👤 Student Information")

col1, col2, col3 = st.columns(3)

# Column 1
with col1:

    school = st.selectbox(
        "School",
        ["GP", "MS"],
        format_func=lambda x: {
            "GP": "GP — Gabriel Pereira",
            "MS": "MS — Mousinho da Silveira"
        }[x]
    )

    sex = st.selectbox(
        "Sex",
        ["F", "M"]
    )

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17
    )

    address = st.selectbox(
        "Address",
        ["U", "R"],
        format_func=lambda x: {
            "U": "U — Urban",
            "R": "R — Rural"
        }[x]
    )

    famsize = st.selectbox(
        "Family Size",
        ["GT3", "LE3"],
        format_func=lambda x: {
            "GT3": "GT3 — Greater than 3 people",
            "LE3": "LE3 — 3 or fewer people"
        }[x]
    )

    Pstatus = st.selectbox(
        "Parent Status",
        ["T", "A"],
        format_func=lambda x: {
            "T": "T — Living together",
            "A": "A — Living apart"
        }[x]
    )

    Medu = st.selectbox(
        "Mother's Education",
        [0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "0 — No education",
            1: "1 — Primary education",
            2: "2 — 5th–9th grade",
            3: "3 — Secondary education",
            4: "4 — Higher education"
        }[x]
    )

    Fedu = st.selectbox(
        "Father's Education",
        [0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "0 — No education",
            1: "1 — Primary education",
            2: "2 — 5th–9th grade",
            3: "3 — Secondary education",
            4: "4 — Higher education"
        }[x]
    )

# Column 2
with col2:

    Mjob = st.selectbox(
        "Mother's Job",
        ["teacher", "health", "services", "at_home", "other"]
    )

    Fjob = st.selectbox(
        "Father's Job",
        ["teacher", "health", "services", "at_home", "other"]
    )

    reason = st.selectbox(
        "Reason for Choosing School",
        ["course", "home", "reputation", "other"]
    )

    guardian = st.selectbox(
        "Guardian",
        ["mother", "father", "other"]
    )

    traveltime = st.number_input(
        "Travel Time",
        min_value=1,
        max_value=4,
        value=2
    )

    studytime = st.number_input(
        "Study Time",
        min_value=1,
        max_value=4,
        value=2
    )

    failures = st.number_input(
        "Past Failures",
        min_value=0,
        max_value=3,
        value=0
    )

    schoolsup = st.selectbox(
        "School Support",
        ["yes", "no"]
    )


# Column 3
with col3:

    famsup = st.selectbox(
        "Family Support",
        ["yes", "no"]
    )

    paid = st.selectbox(
        "Extra Paid Classes",
        ["yes", "no"]
    )

    activities = st.selectbox(
        "Extra-curricular Activities",
        ["yes", "no"]
    )

    nursery = st.selectbox(
        "Attended Nursery",
        ["yes", "no"]
    )

    higher = st.selectbox(
        "Wants Higher Education",
        ["yes", "no"]
    )

    internet = st.selectbox(
        "Internet Access",
        ["yes", "no"]
    )

    romantic = st.selectbox(
        "Romantic Relationship",
        ["yes", "no"]
    )

    famrel = st.number_input(
        "Family Relationship Quality",
        min_value=1,
        max_value=5,
        value=4
    )

# Lifestyle & Academic Information
st.divider()

st.header("📚 Lifestyle & Academic Information")

col1, col2, col3 = st.columns(3)

# Lifestyle Column 1
with col1:

    freetime = st.number_input(
        "Free Time",
        min_value=1,
        max_value=5,
        value=3
    )

    goout = st.number_input(
        "Going Out",
        min_value=1,
        max_value=5,
        value=3
    )

    Dalc = st.number_input(
        "Weekday Alcohol Consumption",
        min_value=0,
        max_value=5,
        value=0
    )



# Lifestyle Column 2
with col2:

    Walc = st.number_input(
        "Weekend Alcohol Consumption",
        min_value=0,
        max_value=5,
        value=0
    )

    health = st.number_input(
        "Health Status",
        min_value=1,
        max_value=5,
        value=4
    )

    absences = st.number_input(
        "Absences",
        min_value=0,
        max_value=100,
        value=10
    )



# Academic Column
with col3:

    G1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )



# Predictions
st.divider()

st.header("🤖 Prediction")

if st.button(
    "🔮 Predict Final Grade",
    type="primary",
    use_container_width=True
):

    # Create dataframe from user inputs
    new_student = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1
    }])

    # Preprocess the input
    new_student_processed = preprocessor.transform(
        new_student
    )

    # Generate prediction
    prediction = model.predict(
        new_student_processed
    )

    predicted_g3 = round(
        prediction[0],
        2
    )

    # Display Result
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Final Grade (G3)",
            predicted_g3
        )

    with result_col2:

        if predicted_g3 >= 10:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

    st.info(
        "The predicted G3 grade represents the model's estimate "
        "of the student's final grade based on the information provided."
    )


# Details about the Project
st.divider()

st.subheader("📌 About This Project")

st.markdown(
    """
    **Student Performance Predictor — Machine Learning Project**

    This application predicts a student's final **G3 grade** using
    academic, demographic, family, and lifestyle information.

    ### Dataset Scope & Limitation

    The machine learning model was trained using the **UCI Student
    Performance dataset**, which contains student data from two
    Portuguese schools.

    Therefore, the predictions are estimates based on patterns learned
    from this dataset and may not generalize equally to students from
    other schools, countries, or education systems.

    ### Education Level Reference

    The UCI dataset represents parents' education using numerical values:

    - **0** — No education
    - **1** — Primary education
    - **2** — 5th–9th grade
    - **3** — Secondary education
    - **4** — Higher education

    These numerical values are retained because they are part of the
    original dataset and are required by the trained machine learning model.
    """
)

st.caption(
    "Student Performance Predictor • Machine Learning Project"
)