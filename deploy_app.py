
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Load the trained model and label encoders ---

@st.cache_resource
def load_model(model_path='linear_regression_model.pkl'):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_label_encoders(encoders_path='label_encoders.pkl'):
    with open(encoders_path, 'rb') as file:
        label_encoders = pickle.load(file)
    return label_encoders

model = load_model()
label_encoders = load_label_encoders()

# Get the list of original column names from one of the encoders (e.g., 'Company Name')
# This is a heuristic; ideally, we'd save feature names directly.
# For now, let's assume the order of features is fixed as per X_train in the notebook:
# 'Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'

# Load the dataset (ensure this file is in the same directory as deploy_app.py when deployed)
df = pd.read_csv('Salary_Dataset_DataScienceLovers (1).csv')

# --- Streamlit UI ---
st.set_page_config(page_title='Salary Prediction App', page_icon=':moneybag:')

st.title('Job Salary Prediction')
st.write('Enter the details below to predict the salary.')

# --- Input fields ---
rating = st.slider('Rating', 1.0, 5.0, 3.5, 0.1)
company_name_input = st.text_input('Company Name')
job_title_input = st.text_input('Job Title')
salaries_reported = st.number_input('Salaries Reported', min_value=1, value=1)
location_input = st.text_input('Location')
employment_status_input = st.selectbox('Employment Status', ['Full Time', 'Intern', 'Contract', 'Part Time'])
job_roles_input = st.text_input('Job Roles') # Assuming Job Roles are provided as text initially

# --- Prediction logic ---
if st.button('Predict Salary'):
    # Preprocess inputs
    input_data = pd.DataFrame({
        'Rating': [rating],
        'Company Name': [company_name_input],
        'Job Title': [job_title_input],
        'Salaries Reported': [salaries_reported],
        'Location': [location_input],
        'Employment Status': [employment_status_input],
        'Job Roles': [job_roles_input]
    })

    # Apply Label Encoding using the loaded encoders
    encoded_input = input_data.copy()
    for col in ['Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles']:
        if col in label_encoders:
            try:
                # Use transform for known labels
                encoded_input[col] = label_encoders[col].transform(encoded_input[col])
            except ValueError:
                # Handle unseen labels: assign a default value or treat as unknown
                # For simplicity here, we'll assign a common encoded value (e.g., 0)
                # In a real app, you might want more sophisticated handling or error messages
                st.warning(f"'{input_data[col].iloc[0]}' for {col} is an unseen value. Assigning default value (0).")
                encoded_input[col] = 0
        else:
            st.error(f"Label encoder for '{col}' not found. Cannot preprocess.")
            st.stop()

    # Ensure column order matches training data (important for many models like Linear Regression)
    # The order of X_train columns was: 'Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'
    # Make sure this matches the order of encoded_input columns
    # This assumes 'Salary' was dropped, so these are the 7 features.
    feature_cols = ['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']
    encoded_input = encoded_input[feature_cols]

    # Make prediction
    prediction = model.predict(encoded_input)

    st.success(f'Predicted Salary: ₹{prediction[0]:,.2f}')

st.write('---')
st.markdown("Developed by Your Name/Team")
