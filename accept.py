#import imp
#from pyexpat import features
import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib
from sklearn.ensemble import RandomForestClassifier

def app():
    st.title('The FDP re-employement for UNHCR contribution Scheme')
    st.write(
        """
        #FDP(Forcibly Displaced Persons) Contribution Prediction (Dummy-UNHCR-App)
        This application predicts the possible **contribution range of the FDPs**, based on their **language, education and past profession**.
        Note: Work experience lenght has not been considered and with finer and granulated detailed data, 
        the prediction will become much more realistic, reliable and accurate.
        Data have been synthesized from multiple government, UNHCR and non governmental bodies along with reasonable speculations to fill in the gaps.#
        Steps have been taken to align the data to the real world facts as much as possible by relying on the mentioned resources and research insights.
        """
    )

    st.sidebar.header('User Input Features')

    st.sidebar.markdown(
        """
        [CSV file Example](https://raw.githubusercontent.com/Muhammadabir/Refugee_App/main/refugee_range_example_dataset.csv)
        """
    )

    #Collecting input features into dataframe
    file_upload = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if file_upload is not None:
        #input_df
        df_input = pd.read_csv(file_upload)
    else:
        def input_features():
            language = st.sidebar.selectbox('Select Language:', ("Native", "Native + English", "Native + English + French", "Native + French"))
            education_level = st.sidebar.selectbox('Select Latest Education Qualification:', ("No Education", "Primary", "High School", "Undergrad", "Associate Degree", "Vocational Degree", "Masters", "PhD"))
            last_occupation = st.sidebar.selectbox('Select Last Occupation',("Not employed","Chefs", "Baker", "Fisher", "Farmer", "Construction Worker", "Mechanic", "University Lecturer", "Software Developer", "Data Scientist", "AI Specialist", "DevOps Professional", 
                              "Machine Learning Professional", "Cybersecurity Professional", 
                              "Newtwork Engineer", "Cloud Engineer", "IT Professional and Constultant", 
                              "Dentist", "Vet", "Medical Doctor", "Medical Technician", 
                              "Physical Therapist", "Pharmaceutical Professional", 
                              "Nurses and Personal Carers", "Mechanical Engineer", "Electronic Engineer", "Civil Engineer", 
                              "Electrical Engineer", "Petroleum Engineer", "Accountant", "Sales Professional", 
                              "Legal Professional", "Marketing Professional", "Managment Professional", "Administrative Assistant"))

            data = {
                'language':language,
                'education_level': education_level,
                'last_occupation':last_occupation
            }
        
            features = pd.DataFrame(data, index=[0])
            return features
        df_input = input_features()

    #Combining input features with entire refugee dataset to use it for encoding
    refugee_raw = pd.read_csv('refugee_acceptance_dataset.csv', index_col=0)
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
        st.write(df)
    else:
        st.write('Upload CSV file? Example input parameters are currently in use')
        st.write(refugee_raw.head(1)) #this gives the non encoded version
        #st.write(df)#this gives the onehot encoded version

    #Uploading the model created for contribution range predection
    load_rfc = joblib.load('model_folder/accept_final_model.pkl')

    #Making prediction using the model
    predict = load_rfc.predict(df)
    probability_pred = load_rfc.predict_proba(df)

    st.subheader('Final Prediction')
    qualified_for_unhcr_mandatory_contribution = np.array(['not accepted', 'accepted'])
    st.write(qualified_for_unhcr_mandatory_contribution[predict])

    st.subheader('Prediction Probability Distribution')
    st.write(probability_pred)