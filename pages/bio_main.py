# ------------------------------------------------------------------------------
# Bio Template Dashboard
# Author: Tiago de Paula Alves
# This project was developed using Streamlit for the visualization and analysis 
# of biological data. It includes custom modules for data loading, spreadsheet 
# editing, graphical visualization, and statistical testing.
# Last updated: June 2025
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from all_functions import *
import time
import yaml
import traceback

# Configure Streamlit layout
st.set_page_config(layout="wide", page_title="Bio Template", page_icon="📊")

# Check login status
if not st.session_state.get("connected", False):
    st.error("Você precisa fazer login para acessar esta página.")
    time.sleep(2)
    st.switch_page("login.py")

# Load configuration from YAML file
with open("dashboard_params.yaml", "r", encoding="utf-8") as file: 
    config = yaml.safe_load(file) 

cfg = config["sources"]
dataframes = []

# Load and cache data for each source
for key, params in cfg.items():
    sheet_name = params["sheet_name"]
    sheet_tab_name = params.get("sheet_tab_name", key)
    session_key = f"data_{sheet_name}_{sheet_tab_name}"

    df = Data_API.load_data_API(
        sheet_name=sheet_name,
        sheet_tab_name=sheet_tab_name,
        date_field=params.get("date_field"),
        columns_to_clean=params.get("columns_to_clean", []),
        date_format=params.get("date_format")
    )

    dataframes.append((df, sheet_name, sheet_tab_name, session_key))

# Apply sidebar styling
UI.sidebar_style()
UI.styled_btn()

# Sidebar navigation
st.sidebar.title("Navegação")
selected_page = st.sidebar.radio("Escolha a Página", ["Analisar Amostras", "Verificar Elisa", "Testes", "Heat-Map"])

# Page: Analisar Amostras
if selected_page == "Analisar Amostras":    
    table_names = [tab_name for _, _, tab_name, _ in dataframes]
    UI.header("Analisar Amostras", SECONDARY_COLOR, size=2)
    UI.header("Lara aqui o objetivo é poder observar as tabelas de Amostras apenas", PRIMARY_COLOR, size=6)

    df, sheet_name, sheet_tab_name, session_key = Data_info.get_tuple_by_sheet_tab_name(dataframes, "Final_result")

    UI.show_table(
        df,
        title="Dados por Categoria",
        category="Categories",
        key="produtos_table"
    )

    UI.header("Visualização Gráfica", SECONDARY_COLOR, size=3)
    Plot_Gen.plot_from_df(df)

    st.markdown("---")
    UI.header("Métricas Agregadas", SECONDARY_COLOR, size=3)
    chosen_value = st.session_state.get("chosen_value", None)
    if chosen_value and chosen_value in df.columns:
        Plot_Gen.mostrar_metricas_aggregadas(df, chosen_value, cor=PRIMARY_COLOR)

# Page: Verificar Elisa
elif selected_page == "Verificar Elisa":
    UI.header("Verificar Elisa", SECONDARY_COLOR, size=2)
    UI.header("Lara aqui você pode observar a Elisa que você tem atualmente em registro e alterá-la caso queira.", PRIMARY_COLOR, size=6)

    df, sheet_name, sheet_tab_name, session_key = Data_info.get_tuple_by_sheet_tab_name(dataframes, "Elisa")
    df2, sheet_name2, sheet_tab_name2, session_key2 = Data_info.get_tuple_by_sheet_tab_name(dataframes, "Final_result")

    df_loaded = None
    UI.editable_table(sheet_name, sheet_tab_name, editable=True)

    Data_info.comparar_samples(df, df2)

    if "load_csv" not in st.session_state:
        st.session_state["load_csv"] = False

    if st.button('📥 Carregar CSV'):
        st.session_state["load_csv"] = True

    if st.session_state["load_csv"] == True:
        df_loaded = Data_info.load_csv_interactively()

    if df_loaded is not None:
        st.write(df_loaded)
        df_loaded = df_loaded.where(pd.notnull(df_loaded), None)
        Data_API.overwrite_sheet(sheet_name, sheet_tab_name, df_loaded)
        st.success("✅ Planilha substituída com sucesso!")
        st.session_state["load_csv"] = False  

# Page: Testes
elif selected_page == "Testes":
    # Prepare required data
    dfs_disponiveis = {
        "Elisa": Data_info.get_tuple_by_sheet_tab_name(dataframes, "Elisa")[0],
        "Pacientes": Data_info.get_tuple_by_sheet_tab_name(dataframes, "Final_result")[0]
    }
    dfs_disponiveis = {k: v for k, v in dfs_disponiveis.items() if v is not None}
    if len(dfs_disponiveis) < 2:
        st.error("Preciso de Elisa e Pacientes para rodar os testes.")
        st.stop()

    UI.header("🔢 Análise Estatística", SECONDARY_COLOR, size=2)
    UI.header("Lara aqui você pode selecionar e executar testes estatísticos para analisar seus dados biológicos.", PRIMARY_COLOR, size=6)

    teste = st.selectbox("Tipo de Teste", ["shapiro", "mannwhitney", "fisher"])

    # Show test-specific UI
    if teste == "shapiro":
        with st.expander("Dica:"):
            st.write("""
            O teste Shapiro-Wilk verifica se os dados seguem uma distribuição normal.  
            É importante para validar suposições de normalidade antes de aplicar testes paramétricos.  
            Testa a hipótese nula de que os dados são normalmente distribuídos.  
            Um p-valor baixo indica desvio da normalidade.
            """)
        params = MenuEstatistico.menu_shapiro(dfs_disponiveis)
    elif teste == "mannwhitney":
        with st.expander("Dica:"):
            st.write("""
            O Mann-Whitney U é um teste não paramétrico usado para comparar duas amostras independentes.  
            É uma alternativa ao teste t quando a normalidade dos dados não pode ser assumida.  
            Compara as distribuições das duas amostras para verificar diferenças significativas.  
            Útil em dados ordinais ou amostras pequenas.
            """)
        params = MenuEstatistico.menu_mannwhitney(dfs_disponiveis)
    else:
        with st.expander("Dica:"):
            st.write("""
            O Teste Exato de Fisher avalia associação entre duas variáveis categóricas em tabelas 2x2.  
            Muito indicado para amostras pequenas ou quando as frequências esperadas são baixas.  
            Verifica a independência das variáveis testadas.  
            Um p-valor baixo sugere associação estatística significativa.
            """)
        params = MenuEstatistico.menu_fisher(dfs_disponiveis)

    # Execute selected test
    if st.button("Executar Teste"):
        with st.spinner("Processando..."):
            try:
                resultado = Data_op.executar_teste(
                    df1=params["df1"],
                    df2=params.get("df2", params["df1"]),
                    categories=params["categories"],
                    number_column=params["number_column"],
                    test_type=teste,
                    limiar=1,
                    chosen_class=params["chosen_class"],
                )
                st.session_state["resultado_atual"] = resultado
                st.session_state["teste_atual"] = teste

            except Exception as e:
                st.error("Erro ao executar o teste:")

    # Display result if available
    if "resultado_atual" in st.session_state and "teste_atual" in st.session_state:
        UI.header("Resultado do Teste", SECONDARY_COLOR, size=4)
        for df in st.session_state["resultado_atual"]:
            UI.show_table(df)

        

        show_df = st.checkbox("Gerar Grafico?")

        if show_df:
            selected_index = st.selectbox("Escolha o índice do dataframe", range(len(st.session_state["resultado_atual"])))
            Plot_Gen.plot_from_df(st.session_state["resultado_atual"][selected_index])    
        nome_personalizado = st.text_input("Nome do arquivo (opcional)")

        if st.button("📤 Salvar no Google Drive"):
            try:
                exporter = GoogleSheetExporter(st.secrets["google_sheets"])
                url = exporter.salvar_resultado(
                    resultado=st.session_state["resultado_atual"],
                    nome_teste=st.session_state["teste_atual"],
                    pasta_principal_id="1Wotum7hiQy7QyvzdlLAGwfZRox96o52l",
                    nome_personalizado=nome_personalizado or None
                )
                st.success(f"Resultado salvo com sucesso! [Abrir planilha]({url})")
                del st.session_state["resultado_atual"]
                del st.session_state["teste_atual"]
            except Exception as e:
                st.error(f"Erro ao salvar no Drive: {e}")

# Page: Heat-Map
elif selected_page == "Heat-Map":
    # Load available data options
    data_options = {
        session_key: (df, f"{sheet_name} - {sheet_tab_name}")
        for df, sheet_name, sheet_tab_name, session_key in dataframes
    }

    selected_key = st.selectbox("Selecione o conjunto de dados", options=['data_bio_maps_amostras', 'data_bio_maps_contami_parks'], format_func=lambda k: data_options[k][1])
    df_selected, label = data_options[selected_key]

    # Select plot configuration
    plot_type = st.selectbox("Tipo de Mapa", ["Pontos", "Heatmap"])
    category_col = st.selectbox("Categoria para Colorir (opcional)", [None] + list(df_selected.columns))
    metric_col = st.selectbox("Métrica para Tamanho/Intensidade (opcional)", [None] + list(df_selected.columns))

    # Generate selected plot
    if plot_type == "Pontos":
        fig = Plot_Gen.generate_map_plot(df_selected, category_col, metric_col)
    else:
        fig = Plot_Gen.generate_map_heatmap(df_selected, metric_col)

    # Show map
    st.plotly_chart(fig, use_container_width=True)
