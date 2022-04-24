#import imp
#from pyexpat import features
import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib
from sklearn.ensemble import RandomForestClassifier

def app():
    st.title('FDP re-employement Scheme Selection')
    st.write(
        """
        #FDP(Forcibly Displaced Persons) Contribution Prediction (Dummy-UNHCR-App)
        This application predicts the **contribution possiblility of the FDPs/acceptance for the scheme**, based on their **language, education and past profession**.
        
        """
    )

    st.sidebar.header('User Input Features')

    st.sidebar.markdown(
        """
        [CSV file Example](https://raw.githubusercontent.com/Muhammadabir/Refugee_App/main/example_accept_final_dataset.csv)
        """
    )

    #Collecting input features into dataframe
    file_upload = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if file_upload is not None:
        #input_df
        df_input = pd.read_csv(file_upload)
    else:
        def input_features():
            #trying to use the forsm
            form1 = st.sidebar.form(key="Options")
            language = form1.selectbox('Select Language:', ("Native", "Native + English", "Native + English + French", "Native + French"))
            education_level = form1.selectbox('Select Latest Education Qualification:', ("No Education", "Primary", "High School", "Undergrad", "Associate Degree", "Vocational Degree", "Masters", "PhD"))
            last_occupation = form1.selectbox('Select Last Occupation',("Not employed","Chefs", "Baker", "Fisher", "Farmer", 
                                                                        "Construction Worker", "Mechanic", "Senior Software Developer", "Junior Software Developer" 
                                                                        "Junior Data Scientist", "Junior AI Specialist", "Junior DevOps Professional", 
                                                                        "Junior Machine Learning Professional", "Junior Cybersecurity Professional", 
                                                                        "Junior Newtwork Engineer", "Junior Cloud Engineer", "Junior IT Professional and Consultant", 
                                                                        "Junior Dentist", "Junior Vet", "Junior Medical Doctor", "Junior Medical Technician", 
                                                                        "Junior Physical Therapist", "Junior Pharmaceutical Professional", 
                                                                        "Junior Nurses and Personal Carers", "Junior Mechanical Engineer", 
                                                                        "Junior Electronic Engineer", "Junior Civil Engineer", 
                                                                        "Junior Electrical Engineer", "Junior Petroleum Engineer", "Junior Accountant", 
                                                                        "Junior Sales Professional", 
                                                                        "Junior Legal Professional", "Junior Marketing Professional", "Junior Management Professional",
                                                                        "Junior Administrative Assistant", "Senior Data Scientist", "Senior AI Specialist", 
                                                                        "Senior DevOps Professional", 
                                                                        "Senior Machine Learning Professional", "Senior Cybersecurity Professional", 
                                                                        "Senior Newtwork Engineer", "Senior Cloud Engineer", "Senior IT Professional and Consultant", 
                                                                        "Senior Dentist", "Senior Vet", "Senior Medical Doctor", "Senior Medical Technician", 
                                                                        "Senior Physical Therapist", "SeniorPharmaceutical Professional", 
                                                                        "Senior Nurses and Personal Carers", "Senior Mechanical Engineer", 
                                                                        "Senior Electronic Engineer", "Senior Civil Engineer", 
                                                                        "Senior Electrical Engineer", "Senior Petroleum Engineer", "Senior Accountant", 
                                                                        "Senior Sales Professional", 
                                                                        "Senior Legal Professional", "Senior Marketing Professional", "Senior Management Professional", 
                                                                        "Senior Administrative Assistant", "Junior University Lecturer", 
                                                                        "Senior Petroleum Engineer-PhD", "Senior University Lecturer", "Senior Cybersecurity Professional-PhD",
                                                                        "Senior Machine Learning Professional-PhD",
                                                                        "Senior Data Scientist-PhD", "Senior Software Developer-PhD",  "Senior AI Specialist-PhD", 
                                                                        "Senior Medical Doctor-PhD", "Senior Dentist-PhD", 
                                                                        "Senior Pharmaceutical Professional-PhD","Senior Electrical Engineer-PhD"))
            data = {
                'language':language,
                'education_level': education_level,
                'last_occupation':last_occupation
            }
        
            #the form button
            form1.form_submit_button('Predict')
        
            features = pd.DataFrame(data, index=[0])
            return features
        df_input = input_features()

    #Combining input features with entire refugee dataset to use it for encoding
    refugee_raw = pd.read_csv('1milrefugee_accept_final_dataset.csv', index_col=0)
    refugee = refugee_raw.drop(columns=['qualified_for_unhcr_mandatory_contribution'])
    df = pd.concat([df_input, refugee], axis=0)

    #Ordinal Encoding/oneHot encoding of the features
    encode_variable = ['language', 'education_level', 'last_occupation']
    for col in encode_variable:
        dummy_data = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy_data], axis=1)
        del df[col]
    #To select only the first row/input data
    df = df[:1]

    #Displaying input features
    st.subheader('Selected Input Feature')

    if file_upload is not None:
        #This gives the actual encoded version of the data
        #st.write(df)
        #This returns the non encoded version of the data
        st.write(df_input)
    else:
        st.write('Upload CSV file? Example input parameters are currently in use')
        st.write(refugee_raw.head(1)) #this gives the non encoded version
        #st.write(df)#this gives the onehot encoded version

    #Using cahce
    #@st.experimental_singleton
    @st.cache(allow_output_mutation=True)
    def model_load(model_name):
        load_rfc = joblib.load(model_name)    
        return (load_rfc)
    #Uploading the model created for contribution range predection
    load_rfc = model_load('model_folder/accept_main_model.pkl')

    #Making prediction using the model
    predict = load_rfc.predict(df)
    probability_pred = load_rfc.predict_proba(df)

    st.subheader('Final Prediction')
    qualified_for_unhcr_mandatory_contribution = np.array(['not accepted', 'accepted'])
    st.write(qualified_for_unhcr_mandatory_contribution[predict])

    st.subheader('Prediction Probability Distribution')
    st.write(probability_pred)