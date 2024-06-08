import streamlit as st
import pandas as pd
import joblib # type: ignore
import os

# Get the absolute path to the current script
script_path = os.path.abspath(__file__)

# Set the working directory to the script's directory
os.chdir(os.path.dirname(script_path))

# Specify the path to the model folder relative to the script's directory
model_folder = "model"

# Construct the full path to the model file
model_file_path = os.path.join(model_folder, 'logistic_regression.pkl')
model = joblib.load(model_file_path)
st.image(r'https://media.licdn.com/dms/image/D4E12AQH_H_AW53uc3w/article-cover_image-shrink_600_2000/0/1692542594542?e=2147483647&v=beta&t=TFq4vi4dbR3XpubgjI0PVF6ozNWVdDSDynBj1IAXpx0')
# Define the input fields
def user_input_features():
    gender = st.selectbox('Gender', ('Female', 'Male'))
    SeniorCitizen = st.selectbox('Senior Citizen', ('Yes', 'No'))
    Partner = st.selectbox('Partner', ('Yes', 'No'))
    Dependents = st.selectbox('Dependents', ('Yes', 'No'))
    tenure = st.slider('Tenure (months)', 0, 72, 1)
    PhoneService = st.selectbox('Phone Service', ('Yes', 'No'))
    MultipleLines = st.selectbox('Multiple Lines', ('Yes', 'No'))
    InternetService = st.selectbox('Internet Service', ('No','Yes'))
    Contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
    PaymentMethod = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    MonthlyCharges = st.number_input('Monthly Charges')
    TotalCharges = st.number_input('Total Charges')
    TotalOnlineServices = st.slider('Total Online Services', 0, 6, 1)
    

    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'TotalOnlineServices': TotalOnlineServices
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Streamlit app
st.title('Customer Churn Prediction')

#st.sidebar.header('User Input Features')
input_df = user_input_features()

# Process the inputs and make a prediction
if st.button('Predict'):

    prediction = model.predict(input_df)
    st.subheader('Prediction:')
    st.header(':green[Churn]' if prediction[0] == 1 else ':red[Churn]')
