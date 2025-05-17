import streamlit as st
import pandas as pd
from src.weather_data_processing import predict_input, get_plot

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(
   page_title="Predict Rain in Australia",
   layout="wide",
   initial_sidebar_state="expanded",
)

raw_df = pd.read_csv('data/weatherAUS.csv')
location_list = raw_df['Location'].dropna().unique().tolist()
wind_dir_list = raw_df['WindGustDir'].dropna().unique().tolist()

with st.sidebar:
    st.title('Select weather parameters')
    Location = st.selectbox('Select location:', location_list)
    RainToday = st.selectbox('Rain Today', ['No', 'Yes'])
    WindGustDir = st.selectbox('Strongest Wind Gust Direction', wind_dir_list)
    WindDir9am = st.selectbox('Wind Direction at 9am', wind_dir_list)
    WindDir3pm = st.selectbox('Wind Direction at 3pm', wind_dir_list)
    WindGustSpeed = st.slider('Strongest Wind Gust Speed', 0, 140, 35, step=1)
    WindSpeed9am = st.slider('Wind Speed at 9am', 0, 140, 9, step=1)
    WindSpeed3pm = st.slider('Wind Speed at 3pm', 0, 140, 13, step=1)
    Temp9am = st.slider('Temperature at 9am', -10., 50., 17.,
                        step=0.1, format="%0.1f")
    Temp3pm = st.slider('Temperature at 3pm', -10., 50., 20.,
                        step=0.1, format="%0.1f")
    MinTemp = st.slider('Minimum Temperature', -10., 50., 12.0,
                        step=0.1, format="%0.1f")
    MaxTemp = st.slider('Maximum Temperature', -10., 50., 22.6,
                        step=0.1, format="%0.1f")
    Sunshine = st.slider('Sunshine', 0., 15., 8.4,
                         step=0.1, format="%0.1f")
    Cloud9am = st.slider('Cloud Cover at 9am', 0, 9, 7, step=1)
    Cloud3pm = st.slider('Cloud Cover at 3pm', 0, 9, 7, step=1)
    Rainfall = st.slider('Rainfall', 0., 400., 0.,
                         step=0.1, format="%0.1f")
    Evaporation = st.slider('Evaporation', 0., 150., 4.8,
                            step=0.1, format="%0.1f")
    Humidity9am = st.slider('Humidity at 9am', 0, 100, 70, step=1)
    Humidity3pm = st.slider('Humidity at 3pm', 0, 100, 52, step=1)
    Pressure9am = st.slider('Atmospheric Pressure at 9am',
                            970., 1050., 1016.4,
                            step=0.1, format="%0.1f")
    Pressure3pm = st.slider('Atmospheric Pressure at 3pm',
                            970., 1050., 1015.3,
                            step=0.1, format="%0.1f")

input_data = {
    'Location': Location,
    'MinTemp': MinTemp,
    'MaxTemp': MaxTemp,
    'Rainfall': Rainfall,
    'Evaporation': Evaporation,
    'Sunshine': Sunshine,
    'WindGustDir': WindGustDir,
    'WindGustSpeed': WindGustSpeed,
    'WindDir9am': WindDir9am,
    'WindDir3pm': WindDir3pm,
    'WindSpeed9am': WindSpeed9am,
    'WindSpeed3pm': WindSpeed3pm,
    'Humidity9am': Humidity9am,
    'Humidity3pm': Humidity3pm,
    'Pressure9am': Pressure9am,
    'Pressure3pm': Pressure3pm,
    'Cloud9am': Cloud9am,
    'Cloud3pm': Cloud3pm,
    'Temp9am': Temp9am,
    'Temp3pm': Temp3pm,
    'RainToday': RainToday
}

col1, col2 = st.columns([2, 1], vertical_alignment='center')
col1.title('Predict Rain in Australia')

col2.image('img/images.png', width=200)

st.write('---')

if st.button(r"$\textit{\Large Get prediction}$", type='primary'):

    input_df = pd.DataFrame([input_data])
    model_path = 'models/aussie_rain_rf.joblib'

    pred, prob = predict_input(input_data, model_path)

    st.markdown('### It will no rain tomorrow.' if pred == 'No'
                else '### It will rain tomorrow.')
    st.markdown(f"#### Probability: {prob:.0%}")

st.markdown("####")

fig = get_plot(raw_df, Location)

st.pyplot(fig, use_container_width=False)
