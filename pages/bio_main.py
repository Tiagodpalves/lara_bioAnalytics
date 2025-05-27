import streamlit as st
import pandas as pd
import plotly.express as px
from template_functions import *
import time



st.set_page_config(layout="wide", page_title="Bio Template", page_icon="📊")


if not st.session_state.get("connected", False):
    st.error("Você precisa fazer login para acessar esta página.")
    time.sleep(2)
    st.switch_page("login.py")


import yaml

with open("dashboard_params.yaml", "r", encoding="utf-8") as file: 
    config = yaml.safe_load(file) 

cfg = config["sources"]
dataframes = []

# Loop dinâmico usando session_state e cache controlado
for key, params in cfg.items():
    sheet_name = params["sheet_name"]
    sheet_tab_name = params.get("sheet_tab_name", key)
    session_key = f"data_{sheet_name}_{sheet_tab_name}"

    # Carrega os dados com cache do streamlit e armazena também em cache manual
    df = Data_API.load_data_API(
        sheet_name=sheet_name,
        sheet_tab_name=sheet_tab_name,
        date_field=params.get("date_field"),
        columns_to_clean=params.get("columns_to_clean", []),
        date_format=params.get("date_format")
    )

    # Guarda a tupla completa: df, nome da aba e chave de sessão
    dataframes.append((df, sheet_name, sheet_tab_name, session_key))




UI.sidebar_style()
UI.styled_btn()

st.sidebar.title("Navegação")
selected_page = st.sidebar.radio("Escolha a Página", ["Analisar Tabelas", "Testes", "Heat-Map"])



if selected_page == "Analisar Tabelas":    
    # Extrair nomes disponíveis
    table_names = [tab_name for _, _, tab_name,_ in dataframes]
    UI.header("Analisar Tabelas", SECONDARY_COLOR, size=2)

    # Dropdown para selecionar a tabela
    selected_tab = st.selectbox("Selecione a tabela para visualização:", table_names)

    df_tuple = next((df, sheetname, sheetTab, sheet_key) for df, sheetname, sheetTab, sheet_key in dataframes if sheetTab == selected_tab)
    df, sheetname, sheetTab, sheet_key = df_tuple

    st.markdown("---")
    UI.header(f"Tabela: {selected_tab}", size=4)
    UI.editable_table(sheetname, sheetTab, sheet_key, editable=True)
    st.markdown("---")

    # Gráfico interativo
    UI.header("Visualização Gráfica", SECONDARY_COLOR, size=3)
    Data_Visualization.plot_from_df(df)

    # Métricas agregadas
    st.markdown("---")
    UI.header("Métricas Agregadas", SECONDARY_COLOR,size=3)
    chosen_value = st.session_state.get("chosen_value", None)
    if chosen_value and chosen_value in df.columns:
        Data_Visualization.mostrar_metricas_aggregadas(df, chosen_value, cor=PRIMARY_COLOR)



elif selected_page == "Testes":
    df_dict = {name: df for df,_,name,_ in dataframes}
    UI.header("🔢 Análise Estatística", SECONDARY_COLOR, size=2)
    UI.header("Realize testes estatísticos com facilidade.", size=3)

    # Aba de seleção de análise individual ou comparativa
    modo = st.selectbox("Escolha o modo de análise:", ["Análise Individual", "Comparar Tabelas"])

    if modo == "Análise Individual":
        selected_table = st.selectbox("Selecione uma tabela para análise:", list(df_dict.keys()))
        df = df_dict[selected_table]

        colunas_numericas = df.select_dtypes(include="number").columns.tolist()
        categorias = df["Category"].dropna().unique().tolist() if "Category" in df.columns else []

        col1, col2 = st.columns(2)
        with col1:
            col_a = st.selectbox("Coluna A", colunas_numericas)
        with col2:
            col_b = st.selectbox("Coluna B", [c for c in colunas_numericas if c != col_a])

        teste = st.selectbox("Teste Estatístico", ["shapiro", "mannwhitney", "fisher"])

        categoria_ativa = None
        chosen_class = None
        if categorias:
            categoria_ativa = st.selectbox("Filtrar por categoria", [None] + categorias)
        if teste == "fisher":
            chosen_class = st.selectbox("Coluna de classificação para Fisher", df.columns.tolist())

        if st.button("Executar Teste"):
            try:
                resultado = Data_preparation.executar_teste(
                    df, col_a, col_b, test=teste,
                    chosen_class=chosen_class,
                    categoria=categoria_ativa
                )
                st.success("Teste realizado com sucesso!")
                UI.data_table(resultado)
                UI.header("Visualização Gráfica", SECONDARY_COLOR, size=3)
                Data_Visualization.plot_from_df(resultado)

                csv = resultado.to_csv(index=False).encode("utf-8")
                st.download_button("📂 Baixar Resultado CSV", csv, f"resultado_{teste}.csv", "text/csv")
            except Exception as e:
                st.error(str(e))

    else:
        selected_tables = st.multiselect("Escolha tabelas para comparar:", list(df_dict.keys()), max_selections=3)

        if len(selected_tables) >= 2:
            dfs = [df_dict[t] for t in selected_tables]
            shared_cols = set.intersection(*(set(df.select_dtypes(include="number").columns) for df in dfs))

            if shared_cols:
                col_a = st.selectbox("Coluna A (compartilhada)", list(shared_cols))
                col_b = st.selectbox("Coluna B (compartilhada)", [c for c in shared_cols if c != col_a])
                teste = st.selectbox("Teste Estatístico", ["shapiro", "mannwhitney"])

                if st.button("Executar Comparativo"):
                    resultados = []
                    for nome, df in [(n, df_dict[n]) for n in selected_tables]:
                        try:
                            resultado = Data_preparation.executar_teste(df, col_a, col_b, test=teste)
                            resultado.insert(0, "Tabela", nome)
                            resultados.append(resultado)
                        except Exception as e:
                            st.warning(f"Erro na tabela {nome}: {e}")

                    if resultados:
                        df_final = pd.concat(resultados, ignore_index=True)
                        UI.data_table(df_final)
                        UI.header("Visualização Gráfica", SECONDARY_COLOR, size=3)
                        Data_Visualization.plot_from_df(resultado)
                        
                        csv = df_final.to_csv(index=False).encode("utf-8")
                        st.download_button("📂 Baixar Comparativo CSV", csv, "comparativo.csv", "text/csv")
            else:
                st.info("As tabelas selecionadas não possuem colunas numéricas em comum.")
        else:
            st.info("Selecione ao menos 2 tabelas para comparação.")
