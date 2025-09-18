import pandas as pd
import os

def run_etl():

    # Define os caminhos relativos para os arquivos
    data_path = 'data'
    path_city = os.path.join(data_path, 'city.csv')
    path_country = os.path.join(data_path, 'country.csv')
    path_language = os.path.join(data_path, 'countrylanguage.csv')

    # Define o caminho para a nova pasta e o arquivo de saída
    cleaned_data_path = 'cleaned_data'
    os.makedirs(cleaned_data_path, exist_ok = True)
    output_path = os.path.join(cleaned_data_path, 'dados_consolidados.csv')
    

    # --- 1. Extração (Extract) ---
    df_city = pd.read_csv(path_city)
    df_country = pd.read_csv(path_country)
    df_language = pd.read_csv(path_language)

    # --- 2. Transformação (Transform) ---
    # Agrega a população das cidades por país
    df_city_agg = df_city.groupby('CountryCode')['Population'].sum().reset_index()
    df_city_agg.rename(columns={'Population': 'PopulationInCities'}, inplace=True)

    # Conta o número de idiomas por país
    df_lang_agg = df_language.groupby('CountryCode')['Language'].count().reset_index()
    df_lang_agg.rename(columns={'Language': 'NumberOfLanguages'}, inplace=True)

    # Padroniza a coluna de junção
    df_country.rename(columns={'Code': 'CountryCode'}, inplace=True)

    # Junta os dataframes
    df_merged = pd.merge(df_country, df_city_agg, on='CountryCode', how='left')
    df_final = pd.merge(df_merged, df_lang_agg, on='CountryCode', how='left')

    # --- 3. Carregamento (Load) ---
    # Salva o dataframe resultante
    df_final.to_csv(output_path, index=False)

    print(f"Processo de ETL concluído. Arquivo consolidado salvo em: {output_path}")


if __name__ == '__main__':
    run_etl()



