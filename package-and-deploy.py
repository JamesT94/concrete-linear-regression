import streamlit as st
import pandas as pd
import pickle


def predict_quality(model, df):
    predictions_data = model.predict(df)
    return predictions_data[0]


infile = open('saved_model', 'rb')
lr_model = pickle.load(infile)
infile.close()

st.title('Concrete Compression Strength Predictor Web App')
st.write('This is a web app to predict the compressive strength of concrete based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction of the model.')

cement = st.sidebar.slider(label='Cement', min_value=50.0, max_value=600.0, value=300.0, step=1.0)
blast_slag = st.sidebar.slider(label='Blast Furnace Slag', min_value=0.0, max_value=400.0, value=100.0, step=1.0)
fly_ash = st.sidebar.slider(label='Fly Ash', min_value=0.0, max_value=200.0, value=50.0, step=1.0)
water = st.sidebar.slider(label='Water', min_value=100.0, max_value=300.0, value=200.0, step=1.0)
superplast = st.sidebar.slider(label='Superplasticizer', min_value=0.0, max_value=50.0, value=10.0, step=1.0)
coarse_agg = st.sidebar.slider(label='Coarse Aggregrate', min_value=700.0, max_value=1200.0, value=900.0, step=1.0)
fine_agg = st.sidebar.slider(label='Fine Aggregrate', min_value=500.0, max_value=1000.0, value=700.0, step=1.0)
age = st.sidebar.slider(label='Age (days)', min_value=1.0, max_value=365.0, value=30.0, step=1.0)

features = {'Cement': cement, 'Blast Furnace Slag': blast_slag, 'Fly Ash': fly_ash, 'Water': water,
            'Superplasticizer': superplast, 'Coarse Aggregrate': coarse_agg, 'Fine Aggregate': fine_agg,
            'Age': age}

features_df = pd.DataFrame([features])

st.table(features_df)

if st.button('Predict'):
    prediction = predict_quality(lr_model, features_df)

    st.write(' Based on feature values, your compressive strength is ' + str(prediction) + 'MPa')
