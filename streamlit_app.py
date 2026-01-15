import streamlit as st
import pandas as pd


st.title("CSV Uploader and Viewer App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the DataFrame
    st.write("Data Preview:", df)
else:
    st.info("Please upload a CSV file to get started.")

st.title("Select Model")

# Define the options for the dropdown
options = ["Logistic Regression", "Decision Tree Classifier", "K-Nearest Neighbor Classifier","Naive Bayes Classifier - Gaussian", "Ensemble Model - Random Forest","Ensemble Model - XGBoost"]

# Create the dropdown menu using st.selectbox
selected_option = st.selectbox(
    "How would you like to be contacted?", # Label for the dropdown
    options                               # List, tuple, or array of options
)

# Display the selected option
st.write("Selected model :", selected_option)

st.title("Evaluation Metrics")

st.title("Confusion Metrics")