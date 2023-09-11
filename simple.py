import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pycaret
import pycaret.regression
import pycaret.classification
from pycaret.classification import *
from pycaret.regression import *


from pycaret.classification import setup, create_model, models
from pycaret.regression import setup, create_model, models
from pycaret.regression import setup as setup_reg, create_model as create_model_reg, models as models_reg, pull
from pycaret.classification import setup as setup_class, create_model as create_model_class, models as models_class, pull




col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left corner
    button_container = st.empty()

    # Button to be placed at the top-left corner
    if button_container.button("**Contact us**", key="costom_button"):
        
        st.write("khadidja_mek@hotmail.fr")


        
with col2:
    # Add custom CSS style to position the button at the top-left and in the middle
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 50%;
            left: 10px;
            transform: translateY(-50%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left and in the middle
    button_container = st.empty()

    # Button to be placed at the top-left and in the middle
    if button_container.button("**Author**", key="my_custom"):
       st.write('Khadidja Mekiri') 
        

       
        
with col3:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left corner
    button_container = st.empty()
    
    if button_container.button("**Satisfaction**", key="custom"):
        st.selectbox("Rate your satisfaction (1-5)", range(1, 6))
        
with col4:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left corner
    button_container = st.empty()
    if button_container.button("**Datasets**", key="Data"):
        st.write("https://www.kaggle.com/datasets/morriswongch/kaggle-datasets")
        
with col5:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left corner
    button_container = st.empty()
    if button_container.button("**About us**", key="info"):
        st.write("VisualModel360 a complete modeling tool that can do visualisation and prediction")


st.sidebar.header('Help Menu')

# Add a button in the sidebar
show_steps = st.sidebar.button('Show Menu')

if show_steps:
    with st.sidebar:
        st.write(" 🌟Welcome to VisualModel360!🌟")  
        st.write("Here's how to dive right into my powerful data visualization and modeling platform:")
        st.write('Upload Your Data: Simply drag and drop your dataset.')
        st.write('Pick Your Model: Browse through our selection and pick the model that best fits your needs.')
        st.write('Sit Back & Relax: Once you are set up, watch VisualModel360 work its magic.')
        st.write('Happy analyzing and enjoy your data journey with us! 🚀')
# Use raw HTML to change the title color
st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)

st.title("VisualModel360")






setup_success = False

# Load data
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", ["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

   
    st.write("Null Values per Column:")
    st.write(df.isnull().sum())

    # Allow user to drop columns
    columns_to_drop = st.multiselect('Select columns to drop:', df.columns)
    df = df.drop(columns=columns_to_drop)
    st.write("Data Visualizations:")

    # For categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        st.write(f"Bar Chart for {col}:")
        fig = plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=col)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # For numerical columns
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        # Histogram
        st.write(f"Histogram for {col}:")
        fig = plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x=col, kde=True)
        st.pyplot(fig)

        # Boxplot
        st.write(f"Boxplot for {col}:")
        fig = plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=col)
        st.pyplot(fig)

    # Ask user for target column
    target_column = st.selectbox('Select your target column:', df.columns)
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    target_continuous = target_column in numerical_columns

    if target_continuous:
        st.write("Based on the continuous nature of the target column, Regression will be applied.")
        task = 'regression'
    else:
        st.write("Based on the categorical nature of the target column, Classification will be applied.")
        task = 'classification'

    # Preprocessing options based on column types
    cat_columns = df.select_dtypes(include=['object']).columns
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns

    if len(cat_columns) > 0:
        cat_missing = st.selectbox("How do you want to handle missing values in categorical columns?", ["most frequent", "create category"])
        if cat_missing == "most frequent":
            for col in cat_columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            for col in cat_columns:
                df[col].fillna("Missing", inplace=True)

    if len(num_columns) > 0:
        num_missing = st.selectbox("How do you want to handle missing values in numerical columns?", ["mean", "median", "mode"])
        if num_missing == "mean":
            for col in num_columns:
                df[col].fillna(df[col].mean(), inplace=True)
        elif num_missing == "median":
            for col in num_columns:
                df[col].fillna(df[col].median(), inplace=True)
        else:
            for col in num_columns:
                df[col].fillna(df[col].mode()[0], inplace=True)

    

    try:
        if task == 'regression':
            # User chooses a regression model
            models_regression = ['lr', 'lasso', 'ridge', 'en', 'dt', 'rf', 'ada', 'gbr', 'mlp']
            model_choice = st.selectbox("Choose a Regression Model:", models_regression)

            # Setting up PyCaret for regression
            reg_setup = pycaret.regression.setup(data=df, target=target_column, session_id=123)
            setup_success = True

            # Creating the chosen regression model
            model_reg = pycaret.regression.create_model(model_choice, verbose=False)

            # Pulling the report
            report = pycaret.regression.pull()
            st.write("PyCaret Regression Model Report")
            st.write(report)

        else:
            # User chooses a classification model
            models_classification = ['dt', 'rf', 'lr', 'knn', 'nb', 'svm', 'ada', 'gbc', 'mlp']
            model_choice = st.selectbox("Choose a Classification Model:", models_classification)

            # Setting up PyCaret for classification
            class_setup = pycaret.classification.setup(data=df, target=target_column, session_id=123)
            setup_success = True

            # Creating the chosen classification model
            model_class = pycaret.classification.create_model(model_choice, verbose=False)

            # Pulling the report
            report = pycaret.classification.pull()
            st.write("PyCaret Classification Model Report")
            st.write(report)

    except Exception as e:
        st.write(f"Error: {e}")
