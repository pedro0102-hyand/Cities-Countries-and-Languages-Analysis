import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib  # Biblioteca para salvar o modelo treinado

def run_training():
   
    print("Iniciando o pipeline de treinamento do modelo...")

    cleaned_data_path = 'cleaned_data'
    input_file = os.path.join(cleaned_data_path, 'dados_consolidados.csv')
    
    # Pasta para salvar o modelo
    models_path = 'models'
    os.makedirs(models_path, exist_ok=True)

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{input_file}' não encontrado.")
        print("Por favor, execute o script '1_etl.py' primeiro.")
        return

    # Remove linhas onde a variável alvo é nula
    df.dropna(subset=['LifeExpectancy'], inplace=True)

    # Remove colunas que não serão usadas
    features_to_drop = [
        'CountryCode', 'Name', 'LocalName', 'GovernmentForm', 
        'HeadOfState', 'Capital', 'Region', 'GNPOld'
    ]
    df_model = df.drop(columns=[col for col in features_to_drop if col in df.columns])

    # Preenchendo valores nulos nas colunas numéricas com a mediana
    for col in df_model.columns:
        if df_model[col].isnull().any() and pd.api.types.is_numeric_dtype(df_model[col]):
            median_value = df_model[col].median()
            df_model[col] = df_model[col].fillna(median_value)

    print("Pré-processamento concluído.")

    # Separando variáveis independentes (X) e dependente (y)
    X = df_model.drop('LifeExpectancy', axis=1)
    y = df_model['LifeExpectancy']

    # --- One-Hot Encoding para todas as colunas categóricas restantes ---
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Dados divididos: {len(X_train)} amostras de treino, {len(X_test)} amostras de teste.")

    # Criação e treino do modelo RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print("Treinando o modelo RandomForestRegressor...")
    model.fit(X_train, y_train)
    print("Treinamento concluído.")

    # Previsões
    y_pred = model.predict(X_test)

    # Avaliação do modelo
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Performance do Modelo ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} anos")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} anos")
    print(f"R-squared (R²): {r2:.2f}")

    # Importância das features
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Top 10 Features Mais Importantes ---")
    print(feature_importances.head(10))

    # Salvando o modelo treinado
    model_filename = os.path.join(models_path, 'modelo_expectativa_vida.joblib')
    joblib.dump(model, model_filename)
    print(f"\nModelo salvo com sucesso em: {model_filename}")


if __name__ == '__main__':
    run_training()

          
    





