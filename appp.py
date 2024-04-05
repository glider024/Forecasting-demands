import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
model=joblib.load('random_forest_model.joblib')
st.title('Energy Demand Prediction')
st.write('This app predicts energy demand using Random Forest Regression.')
hour = st.slider('Hour', 1, 24, step=1)
day_of_week = st.slider('Day of Week', 1, 6, step=1)
month = st.slider('Month', 1, 12, step=1)
input_data = np.array([[hour, day_of_week, month]])
prediction = model.predict(input_data)
st.write('Predicted Energy Consumption:', prediction[0])
