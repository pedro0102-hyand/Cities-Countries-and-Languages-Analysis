import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #vizualizacao de dados estatisticos
import os

def run_analysis():

    print("Iniciando a Análise Exploratória de Dados (EDA)...")

    cleaned_data_path = 'cleaned_data'
    input_file = os.path.join(cleaned_data_path, 'dados_consolidados.csv')
    
    reports_path = 'reports'
    figures_path = os.path.join(reports_path, 'figures')
    os.makedirs(figures_path, exist_ok=True) 

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{input_file}' não encontrado.")
        print("Por favor, execute o script '1_etl.py' primeiro para gerar os dados consolidados.")
        return

    print("\n--- Informações Gerais do DataFrame ---")
    df.info()

    print("\n--- Estatísticas Descritivas (Variáveis Numéricas) ---")
    print(df.describe())

    print("\n--- Contagem de Valores Nulos por Coluna ---")
    print(df.isnull().sum())

    #vizualizacao gráfica dos dados
    sns.set_style("whitegrid")
    
    #Distribuição da nossa variável alvo: Expectativa de Vida
    plt.figure(figsize=(10, 6))
    sns.histplot(df['LifeExpectancy'].dropna(), kde=True, bins=30)
    plt.title('Distribuição da Expectativa de Vida Global')
    plt.xlabel('Expectativa de Vida (Anos)')
    plt.ylabel('Frequência (Nº de Países)')
    plt.savefig(os.path.join(figures_path, '1_dist_expectativa_vida.png'))
    plt.close() 

    #Expectativa de Vida por Continente (Boxplot)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='Continent', y='LifeExpectancy', palette='viridis')
    plt.title('Expectativa de Vida por Continente')
    plt.xlabel('Continente')
    plt.ylabel('Expectativa de Vida (Anos)')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(figures_path, '2_boxplot_vida_por_continente.png'))
    plt.close()

    #Relação entre PNB (GNP) e Expectativa de Vida (Scatter Plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='GNP', y='LifeExpectancy', alpha=0.6)
    plt.title('Relação entre PNB e Expectativa de Vida')
    plt.xlabel('Produto Nacional Bruto (PNB)')
    plt.ylabel('Expectativa de Vida (Anos)')
    plt.xscale('log') 
    plt.savefig(os.path.join(figures_path, '3_scatter_pnb_vs_vida.png'))
    plt.close()

    #Matriz de Correlação 
    plt.figure(figsize=(14, 10))
    numeric_cols = df.select_dtypes(include=np.number)

    # Excluindo colunas que não fazem sentido na correlação
    numeric_cols = numeric_cols.drop(columns=['IndepYear', 'Capital'], errors='ignore') 
    sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
    plt.title('Matriz de Correlação entre Variáveis Numéricas')
    plt.savefig(os.path.join(figures_path, '4_matriz_correlacao.png'))
    plt.close()

    print(f"\nAnálise concluída. Gráficos salvos na pasta: {figures_path}")


if __name__ == '__main__':
    run_analysis()
