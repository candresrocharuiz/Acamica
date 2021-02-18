import streamlit as st
import pandas as pd
import pickle
import numpy as np

with open('model.pkl', 'rb') as m:
    model = pickle.load(m)

st.title('Aplicación de IRIS en Acámica')

lon_petalo = st.text_input("Seleccione la lonitud del petalo", 0)
lon_hoja = st.text_input("Seleccione la lonitud de la hoja", 0)
# ancho_petalo = st.text_input("Seleccione el ancho del pétalo", 0)
# ancho_hoja = st.text_input("Seleccione el ancho de la hoja", 0)

target_names = ['setosa', 'versicolor', 'virginica']

resultado = np.array([lon_petalo, lon_hoja]).reshape(1,2)

prediccion = model.predict(resultado)

st.text("El tipo de flor predicha es: " + target_names[prediccion[0]])



