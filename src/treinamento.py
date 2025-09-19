import numpy as np
import pandas as pd 
import os
import joblib #salvar modulos treinados

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_training():

    print("Iniciando o pipeline para treinamento do modelo ...")

    #caminho dos dados tratados
    cleaned_data_path = 'cleaned_data'
    input_file = os.path.join(cleaned_data_path, 'dados_consolidados.csv')

    #caminho para guardar os modelos
    models_path = 'models'
    os.makedirs(models_path, exist_ok=True)

    #tratamento de erros
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Arquivo nao encontrado ...")
        print("Por favor, rode o script de etl")
        return
    
    #Pré processamento dos dados e selecionando features

    #removendo as linhas onde a variável alvo é nula
    df.dropna(subset=['LifeExpectancy'],inplace=True)

    #selecionando as colunas que usaremos como feature (x)
    features_to_drop = [
        'CountryCode', 'Name', 'LocalName', 'GovernmentForm', 
        'HeadOfState', 'Capital', 'Region', 'GNPOld'
    ]
    df_model = df.drop(columns=features_to_drop) #eliminamos colunas inuteis

    #tratamento de variáveis categóricas
    df_model = pd.get_dummies(df_model, columns=['Continent'], drop_first=True)

    #preenchimento de colunas nulas com as medianas
    for col in df_model.columns: #iterando sobre todas colunas
        if df_model[col].isnull().any(): #verificando se existem valores ausentes
            if pd.api.types.is_numeric_dtype(df_model[col]): #verifica os tipos de dados da coluna
                median_value = df_model[col].median() #calculamos a mediana
                df_model[col].fillna(median_value,inplace=True) #substituimos os nulos pela mediana
        
        print("Pré-processamento dos dados realizado com sucesso")



