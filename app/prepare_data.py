# Libreria de NLP
import nltk
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Librerias de manejo de datos
import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unicodedata

# Entrenamiento y Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
      return self

    def quitar_tildes(self,texto):
      # Normalizar el texto en forma NFD (Normalization Form D)
      texto_normalizado = unicodedata.normalize('NFD', texto)
      # Filtrar los caracteres que no son marcas diacríticas
      texto_sin_tildes = ''.join(c for c in texto_normalizado if unicodedata.category(c) != 'Mn')
      # Devolver el texto sin tildes
      return texto_sin_tildes

    def manage_stop_words(self,tonekized_text):
      stop_words = set(stopwords.words("spanish"))
      words = [word for word in tonekized_text if word not in stop_words]
      return words

    def lemmatization(self,tonekized_text):
      lemmatizer = WordNetLemmatizer()
      words = [lemmatizer.lemmatize(word) for word in tonekized_text]
      return words

    def transform(self, X, y=None):
      X_copy = X.copy()
        
      # Data Cleaning
      X_copy['text'] = X_copy['text'].str.lower()
      X_copy['text'] = X_copy['text'].apply(lambda x : self.quitar_tildes(x))
      X_copy['text'] = X_copy['text'].apply(lambda x : re.sub(r'[^a-z\s]', '', x))
      #Tokenization
      X_copy['text'] = X_copy['text'].apply(lambda x : word_tokenize(x))
      #stop words
      X_copy['text'] = X_copy['text'].apply(lambda x : self.manage_stop_words(x))
      #lemmatization
      X_copy['text'] = X_copy['text'].apply(lambda x : self.lemmatization(x))

      X_copy['text'] = X_copy['text'].apply(lambda x : " ".join(x))

      vectorizer = CountVectorizer()
      X = vectorizer.fit_transform(X_copy['text'])
      vectorized_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

      return vectorized_df

def match_columns(pipe, data):
    trans_data = pipe.transform(data)
    model_words = []
    with open('data\model_words.txt', 'r') as file:
        model_words = file.read().split()

    list_of_columns_test = list(trans_data.columns)
    intersection_list = list(set(model_words).intersection(set(list_of_columns_test)))
    X_test_trans = trans_data[intersection_list]

    for c in model_words:
        if c not in list_of_columns_test:
            X_test_trans[c] = 0
    X_test_trans = X_test_trans[model_words]

    return X_test_trans
