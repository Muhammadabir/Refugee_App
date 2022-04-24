import streamlit as st 
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image

def app():
    header = st.container()
    dataset = st.container()
    
    with header:
        image = Image.open('ref.jpg')
        st.image(image, caption='Refugees used in the UK to fill up the skill shortage of the healthcare sectors (Refugee Council)')

        st.title('Synthetic FDP re-employement data')
        st.text('The project looks at the skills and possible values of those skills that are present in the FDP communities')

    with dataset:
        st.header('Synthetic skills and employement possibilities Dataset')
        st.text('The Dataset have been made from various sources including UNHCR, Talent Beyond Boundaries Categories, Indeed, Glassdoor, Government Websites and many more')
    
        data = pd.read_csv('data/refugee_t_dataset.csv', index_col=0)
        st.write(data.head(25))
     
        st.subheader('Skills within the synthetic sample of FDP population')
    
        last_ocu = pd.DataFrame(data['last_occupation'].value_counts())#.head()
        st.bar_chart(last_ocu)
        with st.expander("Data synthesized from"):
            st.write("Talents Beyond Bounderies [link](https://www.talentbeyondboundaries.org/the-talent-catalog) and from UNHCR [link](https://data2.unhcr.org/en/dataviz/105?sv=0&geo=0)")
        
        chart_df = pd.DataFrame(data['discretionary_income'].value_counts())
        st.area_chart(chart_df)
    
        st.subheader('Education qualifications within the synthetic sample of FDP population')
    
        edu = pd.DataFrame(data['education_level'].value_counts())#.head()
        st.bar_chart(edu)
        with st.expander("Data synthesized from"):
            st.write("Talents Beyond Bounderies [link](https://www.talentbeyondboundaries.org/the-talent-catalog) and from UNHCR [link](https://data2.unhcr.org/en/dataviz/105?sv=0&geo=0)")
        
    
        st.subheader('Language qualifications within the synthetic sample of FDP population')
    
        language = pd.DataFrame(data['language'].value_counts())#.head()
        st.bar_chart(language)
        with st.expander("Data synthesized from"):
            st.write("Talents Beyond Bounderies [link](https://www.talentbeyondboundaries.org/the-talent-catalog), One World Nations Online [link](https://www.nationsonline.org/oneworld/african_languages.htm) and BellaOnline [link](https://www.bellaonline.com/articles/art27809.asp)")
      
        st.subheader('Possible contribution values within the synthetic sample of FDP population')
        contribution_range = pd.DataFrame(data['mandatory_contribution_range'].value_counts())#.head(500)
        st.bar_chart(contribution_range)
    
        st.subheader('Cumulative possible potential opportunities')
        data = pd.read_csv('data/refugee_t_dataset.csv', index_col=0)
        st.write(data.tail(3))
    
        st.subheader('Population meeting the criteria for UNHCR contribution')
        acceptance = pd.DataFrame(data['qualified_for_unhcr_mandatory_contribution'].value_counts())#.head(500)
        st.bar_chart(acceptance)
    
    