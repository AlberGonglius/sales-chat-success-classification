from prepare_data import match_columns, CustomTransformer
import streamlit as st
import joblib
import pandas as pd

PIPE = joblib.load('prepare.pkl')
MODEL = joblib.load('classify.pkl')

st.markdown("""
         <style>
         .big-font1 {
            font-size:30px !important;
            color: yellow !important;
            vertical-align: bottom !important;
            padding-top: 500;
         }
         .big-font2 {
            font-size:30px !important;
            color: green !important;
            vertical-align: bottom !important;
         }
         .big-font3 {
            font-size:30px !important;
            color: red !important;
            vertical-align: bottom !important;
         }
         </style>
         """, unsafe_allow_html=True)

st.title("Clasificación de venta exitosa/No exitosa Mercately")

txt = st.text_area(
    'Introduzca su chat'
    )
label = ""
def check_text():
   if len(txt) >0:
      if len(txt.split(" "))<20:
         st.markdown("""
                  <p class="big-font1">Por favor, escriba un chat con más intervenciones</p>
                  """,unsafe_allow_html=True)
      else:
         data = pd.DataFrame({"text":[txt]})
         data = match_columns(PIPE, data)
         prediction = MODEL.predict(data)
         if 1 in prediction:
            st.markdown("""
                  <p class="big-font2">La venta realizada fue exitosa</p>
                  """,unsafe_allow_html=True)
         elif 0 in prediction:
            st.markdown("""
                  <p class="big-font3">La venta realizada no fue exitosa</p>
                  """,unsafe_allow_html=True)

st.button("Clasificar", on_click=check_text)
