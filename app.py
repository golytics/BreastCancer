import base64
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import FastICA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive



st.set_page_config(page_title='Mohamed Gabr - House Price Prediction', page_icon ='logo.png', layout = 'wide', initial_sidebar_state = 'auto')


import os
import base64

# the functions to prepare the image to be a hyperlink
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


# preparing the layout for the top section of the app
# dividing the layout vertically (dividing the first row)
row1_1, row1_2, row1_3 = st.columns((1, 5, 4))

# first row first column
with row1_1:
    gif_html = get_img_with_href('logo.png', 'https://golytics.github.io/')
    st.markdown(gif_html, unsafe_allow_html=True)

with row1_2:
    # st.image('logo.png')
    st.title('Predicting Breast Cancer Using Artificial Intelligence')
    st.markdown("<h2>A Machine Learning POC for a Client</h2>", unsafe_allow_html=True)

# first row second column
with row1_3:
    st.info(
        """
        ##
        This data product has been prepared as a proof of concept of a machine learning model to predict breast cancer. Developing the final model required
        many steps following the CRISP-DM methodology. After building the model we used it to predict the disease in this application. **The model can be changed/
        enhanced for any another population based on its own data.**
        """)






st.write("""
        This app predicts  **Breast Cancer**
        """)

st.subheader('How to use the model?')
'''
You can use the model by modifying the User Input Parameters on the left. The parameters will be passed to the classification
model. You will see the parameters you selected under the **"These are the values you entered"** section. 

1- By clicking "Submit",  the model will run each time you modify the parameters.

2- You will see the prediction result (whether the person has breast cancer or not) under the **'Results'** section below.

'''
#load data
@st.cache
def load_data():
    return pd.read_csv("cleaned_data.csv")

data = load_data()



@st.cache(allow_output_mutation=True) #added 'allow_output_mutation=True' because kept getting 'CachedObjectMutationWarning: Return value of load_model() was mutated between runs.'
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()



st.sidebar.header("""User input features/ parameters: 

Select/ modify/ type the combination of features below to predict the breast cancer
                """)
# input_type = st.sidebar.selectbox('Input Method', ['Move Sliders', 'Enter Values'], index=1)

# if input_type == 'Enter Values': #display text input fields, show user input, submit button

    #number input fields for features

    #format="%.3f rfom https://discuss.streamlit.io/t/st-number-input-formatter-displaying-blank-input-box/1217
BMI = st.sidebar.number_input('BMI (kg/m2)', format="%.4f", step=0.0001)
Glucose = st.sidebar.number_input('Glucose (mg/dL)', format="%.0f")
Insulin = st.sidebar.number_input('Insulin (µU/mL)', format="%.4f", step=0.0001)
HOMA = st.sidebar.number_input('HOMA', format="%.4f",step=0.0001)
Resistin = st.sidebar.number_input('Resistin (ng/mL)', format="%.4f", step=0.0001)

    # st.sidebar.info(
    # '''
    # 💡 **Tip:**
    # Change input to "move sliders" to get a feel for how the model thinks. Play around with the sliders and watch the predictions change.
    # '''
    # )

    # show user input
'''
### These are the values you entered
'''
f"**BMI**: {BMI:.4f} kg/m2"
f"**Glucose**: {Glucose:.0f} mg/dL"
f"**Insulin**: {Insulin:.4f} µU/mL"
f"**HOMA**: {HOMA:.4f}"
f"**Resistin**: {Resistin:.4f} ng/mL"

#button to create new dataframe with input values
if st.button("submit"):
    dataframe = pd.DataFrame(
        {'BMI':BMI,
        'Glucose':Glucose,
        'Insulin':Insulin,
        'HOMA':HOMA,
        'Resistin ':Resistin }, index=[0]
    )

# if input_type == 'Move Sliders': #display slider input fields

# BMI = st.sidebar.slider('BMI (kg/m2)',
#             min_value=10.0,
#             max_value=50.0,
#             value=float(data['BMI'][0]),
#             step=0.01)
#
# Glucose = st.sidebar.slider('Glucose (mg/dL)',
#             min_value=25,
#             max_value=250,
#             value=int(data['Glucose'][0]),
#             step=1)
    # i kept getting an error so i just used int() after about an hour or so of frustration
    # fyi before applying int(), value was <class 'numpy.int64'>
    # the error:
    # StreamlitAPIException: Slider value should either be an int/float or a list/tuple of 0 to 2 ints/floats

    # Insulin = st.sidebar.slider('Insulin (µU/mL)',
    #                     min_value=1.0,
    #                     max_value=75.0,
    #                     value=float(data['Insulin'][0]),
    #                     step=0.01)
    #
    # HOMA = st.sidebar.slider('HOMA',
    #                     min_value=0.25,
    #                     max_value=30.0,
    #                     value=float(data['HOMA'][0]),
    #                     step=0.01)
    #
    # Resistin = st.sidebar.slider('Resistin (ng/mL)',
    #                     min_value=1.0,
    #                     max_value=100.0,
    #                     value=float(data['Resistin'][0]),
    #                     step=0.01)
    # got KeyError: <class ‘numpy.float64’> and had to add float()
    # https://discuss.streamlit.io/t/keyerror-class-numpy-float64/5147

    #slider values to dataframe
    # dataframe = pd.DataFrame(
    #     {'BMI':BMI,
    #     'Glucose':Glucose,
    #     'Insulin':Insulin,
    #     'HOMA':HOMA,
    #     'Resistin ':Resistin }, index=[0]
    # )
    #
    # '''
    # ## Move the Sliders to Update Results ↔
    # '''

#selectbox section ends
###################################################################################################


try:
    '''
    ## Results
    '''

    if model.predict(dataframe)==0:
        html_str = f"""
        <h3 style="color:lightgreen;">NO BREAST CANCER</h3>
        """

        st.markdown(html_str, unsafe_allow_html=True)
        # st.write('Prediction: **NO BREAST CANCER **')
    else:
        html_str = f"""
        <h3 style="color:red;">BREAST CANCER PRESENT</h3>
        """

        st.markdown(html_str, unsafe_allow_html=True)
        # st.write('Prediction: **BREAST CANCER PRESENT**')

    for healthy, cancer in model.predict_proba(dataframe):
        
        healthy = f"{healthy*100:.2f}%"
        cancer = f"{cancer*100:.2f}%"

        st.table(pd.DataFrame({'The person is healthy is':healthy,
                                'The person has breast cancer is':cancer}, index=['Probability that']))

    # if input_type == 'enter values':
    #     st.success('done 👍')
    # else:
    #     pass

    import warnings
    warnings.filterwarnings('ignore')

except:
    st.write('*press submit to compute results*')
    # st.write("if you see this, it finally worked :O")

st.markdown('---')
st.info("""**Note: ** [The data source is]: ** (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra). The following steps have been applied till we reached the model:

        1- Data Acquisition/ Data Collection (reading data, adding headers)

        2- Data Cleaning / Data Wrangling / Data Pre-processing (handling missing values, correcting data fromat/ data standardization 
        or transformation/ data normalization/ data binning/ Preparing Indicator or binary or dummy variables for Regression Analysis/ 
        Saving the dataframe as ".csv" after Data Cleaning & Wrangling)

        3- Exploratory Data Analysis (Analyzing Individual Feature Patterns using Visualizations/ Descriptive statistical Analysis/ 
        Basics of Grouping/ Correlation for continuous numerical variables/ Analysis of Variance-ANOVA for ctaegorical or nominal or 
        ordinal variables/ What are the important variables that will be used in the model?)

        4- Model Development (Single Linear Regression and Multiple Linear Regression Models/ Model Evaluation using Visualization)

        5- Polynomial Regression Using Pipelines (one-dimensional polynomial regession/ multi-dimensional or multivariate polynomial 
        regession/ Pipeline : Simplifying the code and the steps)

        6- Evaluating the model numerically: Measures for in-sample evaluation (Model 1: Simple Linear Regression/ 
        Model 2: Multiple Linear Regression/ Model 3: Polynomial Fit)

        7- Predicting and Decision Making (Prediction/ Decision Making: Determining a Good Model Fit)

        8- Model Evaluation and Refinement (Model Evaluation/ cross-validation score/ over-fitting, under-fitting and model selection)

""")


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Published By: <a href="https://golytics.github.io/" target="_blank">Dr. Mohamed Gabr</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)