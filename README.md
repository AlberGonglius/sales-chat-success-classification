# sales-chat-success-classification

## Descripción

Este proyecto tiene la finalidad de realizar un  modelo de clasificación en el que a partir de un chat se pueda saber si la venta de un objeto físico por internet es exitosa o no, sin necesidad de que una persona lea la conversación.

## Fuentes de Datos

Los datos con los que se realizó este modelo fueron extraidos a partir de la generación de conversaciones con chatgpt.

## Requerimientos

- Python 3.x
- Pandas
- NLTK
- scikit-learn == 1.2.2
- Streamlit
- Jupyter Notebook (Para ejecutar el Notebook respectivo)

## Uso
1. Para la revisión del entrenamiento del modelo leer y ejecutcar el archivo **model-training.ipynb** en el cual se encuentra la preparación de datos, entrenamiento y evaluación del modelo.

2. Los archivos de los modelos se encuentran en la carpeta app con la extensión pkl: **prepare_data.pkl** para la preparación de datos y **model.pkl** para la predicción de los datos.

3. En el archivo **requirements.txt** se encuentran las librerias empleadas, en el archivo **main.py** se encuentra la aplicación de sreamlit y en el archivo **prepare_data.py** se encuentra el código de la preparación de datos

## Link de aplicación
https://sales-chat-success-classification.streamlit.app/
