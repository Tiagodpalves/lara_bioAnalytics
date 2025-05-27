import datetime
import glob
import os
from functools import reduce

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yaml

from fpdf import FPDF
from google.oauth2.service_account import Credentials
from scipy.stats import fisher_exact, mannwhitneyu, shapiro
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import gspread


PRIMARY_COLOR = "#ffb6c1"
SECONDARY_COLOR = "#416538"
HOVER_COLOR = "#ffc1cc"

class Data_API:
    # Cache interno para controle manual
    _manual_cache = {}

    @staticmethod
    @st.cache_data(ttl=43200)
    def load_data_API(sheet_name, date_field=None, columns_to_clean=[], date_format=None, sheet_tab_name=None):
        service_account_file = st.secrets["google_sheets"]
        if not os.path.exists(service_account_file):
            print(f"File \"{service_account_file}\" does not exist or \"DOCKER_CREDS_FILEPATH\" not correctly set")
            return None

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        credentials = Credentials.from_service_account_file(service_account_file, scopes=scopes)
        client = gspread.authorize(credentials)

        spreadsheet = client.open(sheet_name)
        sheet = spreadsheet.worksheet(sheet_tab_name) if sheet_tab_name else spreadsheet.sheet1
        valores = sheet.get_all_values()
        df = pd.DataFrame(valores[1:], columns=valores[0])

        if date_field:
            if date_format == 'ms':
                df[date_field] = pd.to_datetime(df[date_field], unit=date_format)
            df['date'] = df[date_field].dt.tz_localize(None)
            df = df.drop(date_field, axis=1)

        if columns_to_clean:
            for column in columns_to_clean:
                df[column] = Data_preparation.clean_numeric(df[column])

        # Armazena no cache interno
        Data_API._manual_cache[(sheet_name, sheet_tab_name)] = df.copy()

        return df

    @staticmethod
    def overwrite_sheet(sheet_name, sheet_tab_name, df):
        service_account_file = st.secrets["google_sheets"]
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        credentials = Credentials.from_service_account_file(service_account_file, scopes=scopes)
        client = gspread.authorize(credentials)
        sheet = client.open(sheet_name).worksheet(sheet_tab_name)
        sheet.clear()
        sheet.update([df.columns.values.tolist()] + df.values.tolist())

    @staticmethod
    def clear_single_cache(sheet_name, sheet_tab_name):
        """Limpa apenas o cache interno para uma aba específica."""
        key = (sheet_name, sheet_tab_name)
        if key in Data_API._manual_cache:
            del Data_API._manual_cache[key]


def interface_teste_estatistico(dfs_dict):
    st.header("Testes Estatísticos")

    dfs_selecionados = [nome for nome in dfs_dict if st.checkbox(f"Incluir tabela: {nome}")]

    for nome_df in dfs_selecionados:
        df = dfs_dict[nome_df]
        st.subheader(f"{nome_df}")

        col_a = st.selectbox(f"Coluna A ({nome_df})", df.columns, key=f"a_{nome_df}")
        col_b = st.selectbox(f"Coluna B ({nome_df})", df.columns, key=f"b_{nome_df}")
        test = st.selectbox(f"Tipo de teste ({nome_df})", ["shapiro", "mannwhitney", "fisher"], key=f"teste_{nome_df}")

        categoria = None
        if "Category" in df.columns and st.checkbox(f"Filtrar por categoria ({nome_df})"):
            categorias = df['Category'].dropna().unique().tolist()
            categoria = st.selectbox(f"Categoria ({nome_df})", categorias, key=f"cat_{nome_df}")

        chosen_class = None
        if test == 'fisher':
            chosen_class = st.selectbox(f"Coluna de classe para Fisher ({nome_df})", df.columns, key=f"class_{nome_df}")

        if st.button(f"Executar teste ({nome_df})"):
            try:
                resultado = Data_preparation.executar_teste(df, col_a, col_b, test, chosen_class, categoria)
                st.dataframe(resultado, use_container_width=True)
            except Exception as e:
                st.error(str(e))

class Data_preparation:
    @staticmethod
    def executar_teste(df, col_a, col_b, test='shapiro', chosen_class=None, categoria=None, categoria_column='Category'):
        df_foco = df.copy()

        # Filtro por categoria
        if categoria:
            if categoria_column not in df_foco.columns:
                raise ValueError(f"A coluna de categoria '{categoria_column}' não foi encontrada no DataFrame.")
            df_foco = df_foco[df_foco[categoria_column].astype(str).str.lower() == categoria.lower()]
            if df_foco.empty:
                raise ValueError(f"Nenhuma linha encontrada com a categoria '{categoria}' na coluna '{categoria_column}'.")

        def parse_column(col):
            return pd.to_numeric(df_foco[col].astype(str).str.replace(',', '.'), errors='coerce').dropna()

        grupo_a = parse_column(col_a)
        grupo_b = parse_column(col_b)

        if test.lower() == 'shapiro':
            stat_a, pval_a = shapiro(grupo_a)
            stat_b, pval_b = shapiro(grupo_b)
            return pd.DataFrame({
                'Grupo': [col_a, col_b],
                'W (Shapiro)': [stat_a, stat_b],
                'p-valor': [pval_a, pval_b],
                'Aprovado? (alpha=0.05)': ['Yes' if p > 0.05 else 'No' for p in [pval_a, pval_b]],
                'n': [len(grupo_a), len(grupo_b)]
            })

        elif test.lower() == 'mannwhitney':
            stat, pval = mannwhitneyu(grupo_a, grupo_b, alternative='two-sided', method='exact')
            return pd.DataFrame([{
                'Teste': 'Mann-Whitney U',
                'p-valor': pval,
                'Significativo (p < 0.05)?': 'Yes' if pval < 0.05 else 'No',
                'U': stat,
                'Mediana A': grupo_a.median(),
                'Mediana B': grupo_b.median(),
                'Diferença real': grupo_b.median() - grupo_a.median(),
                'Tamanho A': len(grupo_a),
                'Tamanho B': len(grupo_b),
                'Soma ranks A': grupo_a.rank().sum(),
                'Soma ranks B': grupo_b.rank().sum()
            }])

        elif test.lower() == 'fisher':
            if chosen_class is None:
                raise ValueError("Para o teste de Fisher, é necessário fornecer 'chosen_class'.")

            df_foco = df_foco[df_foco[chosen_class].notna() & df_foco[col_a].notna() & df_foco[col_b].notna()]
            df_grouped = df_foco.groupby(chosen_class)[[col_a, col_b]].sum(numeric_only=True)

            if df_grouped.shape != (2, 2):
                raise ValueError("Fisher requer exatamente duas categorias e duas colunas.")

            table = df_grouped.to_numpy().astype(int)
            categorias = df_grouped.index.tolist()
            oddsratio, pval = fisher_exact(table, alternative='two-sided')

            total = table.sum()
            row_totals = table.sum(axis=1, keepdims=True)
            col_totals = table.sum(axis=0, keepdims=True)

            percent_row = (table / row_totals) * 100
            percent_col = (table / col_totals) * 100
            percent_total = (table / total) * 100

            dados = []
            for i, categoria in enumerate(categorias):
                dados.append({
                    'Grupo': categoria,
                    f'Contagem {col_a}': table[i, 0],
                    f'Contagem {col_b}': table[i, 1],
                    f'% linha {col_a}': f"{percent_row[i, 0]:.2f}%",
                    f'% linha {col_b}': f"{percent_row[i, 1]:.2f}%",
                    f'% coluna {col_a}': f"{percent_col[i, 0]:.2f}%",
                    f'% coluna {col_b}': f"{percent_col[i, 1]:.2f}%",
                    f'% total {col_a}': f"{percent_total[i, 0]:.2f}%",
                    f'% total {col_b}': f"{percent_total[i, 1]:.2f}%",
                    'Teste': "Fisher's exact test",
                    'p-valor': f"{pval:.4g}",
                    'Significativo (p < 0.05)?': 'Yes' if pval < 0.05 else 'No',
                    'P value summary': 'ns' if pval > 0.05 else '*',
                    'One- or two-sided': 'Two-sided'
                })
            return pd.DataFrame(dados)

        else:
            raise ValueError("Teste não reconhecido. Use 'shapiro', 'mannwhitney' ou 'fisher'.")
        
    def time_filter(df, time_filter_option, date_column, key='', position = st.sidebar):
        period = 0
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])

        if df[date_column].dt.tz is not None:
            df[date_column] = df[date_column].dt.tz_convert(None)

        today = pd.Timestamp.now()

        if time_filter_option == "Últimas 24 horas":
            filtered_data = df[df[date_column] >= today - pd.Timedelta(days=1)]

        elif time_filter_option == "Últimos 7 dias":
            filtered_data = df[df[date_column] >= today - pd.Timedelta(days=7)]

        elif time_filter_option == "Últimos 30 dias":
            filtered_data = df[df[date_column] >= today - pd.Timedelta(days=30)]
            period = 30

        elif time_filter_option == "Selecionar Intervalo Personalizado":
            start_date = position.date_input("Data Inicial", value=today - pd.Timedelta(days=30), key=f'{key}_start_date')
            end_date = position.date_input("Data Final", value=today, key=f'{key}_end_date')
            period = end_date - start_date
            if start_date > end_date:
                position.error("A data inicial não pode ser maior que a data final.")
            else:
                filtered_data = df[
                    (df[date_column] >= pd.Timestamp(start_date)) &
                    (df[date_column] <= pd.Timestamp(end_date))
                ]
        
        elif time_filter_option == "Selecionar Ano":
            anos_disponiveis = sorted(df[date_column].dt.year.unique(), reverse=True)
            ano_escolhido = position.selectbox("Escolha o ano", anos_disponiveis, key=f"{key}_ano")
            filtered_data = df[df[date_column].dt.year == ano_escolhido]
            period = 365  # aproximado

        else:
            filtered_data = df

        return filtered_data, period

    @staticmethod
    def clean_numeric(series):
        if series.dtype == object or series.dtype == "string":
            series = series.replace('%', '', regex=True).str.strip()
            series = series.str.replace(',', '.', regex=False)
            series = pd.to_numeric(series, errors='coerce')
        elif not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors='coerce')
        
        return series

    def carregar_grupos_mesclados(raw_path, subpastas, salvar_em=None, verbose=True):
        dfs = []

        for subpasta in subpastas:
            grupo = subpasta.capitalize()
            caminho = os.path.join(raw_path, subpasta, "*.csv")
            arquivos = glob.glob(caminho)

            registros = []
            for arquivo in arquivos:
                nome = os.path.basename(arquivo)
                partes = nome.replace('.csv', '').split()
                categoria = '_'.join(partes[1:]) if len(partes) > 1 else 'Unknown'

                # leitura
                df = pd.read_csv(arquivo)

                # DETECÇÃO SIMPLIFICADA:
                # Se a primeira coluna for 'object' (texto), assumimos que é a coluna de categorias
                primeira_col = df.columns[0]
                if df[primeira_col].dtype == object:
                    df = df.rename(columns={primeira_col: grupo})
                else:
                    # caso contrário, não há coluna de categoria explícita
                    df[grupo] = 'All'

                df['Category'] = categoria
                registros.append(df)

            # concatena todas as leituras desse grupo
            df_grupo = pd.concat(registros, ignore_index=True)

            # identificar as colunas numéricas
            col_excluir = ['Category', grupo]
            col_numericas = [c for c in df_grupo.columns if c not in col_excluir]
            df_grupo[col_numericas] = df_grupo[col_numericas].apply(pd.to_numeric, errors='coerce')

            # só pivota se existirem ao menos 2 categorias distintas
            if df_grupo[grupo].nunique() > 1:
                df_pivot = df_grupo.pivot_table(
                    index=['Category'] + col_numericas,
                    columns=grupo,
                    aggfunc='size',
                    fill_value=0
                ).reset_index()
                df_pivot.columns.name = None

                id_cols = ['Category'] + col_numericas
                var_cols = [c for c in df_pivot.columns if c not in id_cols]

                # reconstrói no formato longo
                rows = []
                for _, row in df_pivot.iterrows():
                    for vc in var_cols:
                        if row[vc] > 0:
                            d = row[id_cols].to_dict()
                            d[grupo] = vc
                            rows.append(d)
                df_final = pd.DataFrame(rows)
            else:
                # mantém como está (ou seja, uma única categoria ou só métricas)
                df_final = df_grupo

            dfs.append(df_final)

        # merge final nas colunas em comum
        df_merged = reduce(
            lambda l, r: pd.merge(l, r, on=list(set(l.columns)&set(r.columns)), how='outer'),
            dfs
        )

        # salvar se pedido
        if salvar_em:
            if salvar_em.endswith('.atc'):
                df_merged.to_pickle(salvar_em)
            elif salvar_em.endswith('.csv'):
                df_merged.to_csv(salvar_em, index=False)

        if verbose:
            print("Subpastas:", subpastas)
            print("Shape:", df_merged.shape)
            print(df_merged.head())

        return df_merged



class Data_Visualization:
    @staticmethod
    def mostrar_metricas_aggregadas(df, coluna_valor, cor=PRIMARY_COLOR):
        if coluna_valor not in df.columns:
            st.warning(f"Coluna '{coluna_valor}' não encontrada para agregações.")
            return
        
        media = df[coluna_valor].mean()
        soma = df[coluna_valor].sum()
        desvio = df[coluna_valor].std()

        col1, col2, col3 = st.columns(3)
        UI.metric_box(col1, "Média", f"{media:.2f}", SECONDARY_COLOR)
        UI.metric_box(col2, "Soma", f"{soma:.2f}", PRIMARY_COLOR)
        UI.metric_box(col3, "Desvio Padrão", f"{desvio:.2f}", SECONDARY_COLOR)

    @staticmethod
    def plot_from_df(df, default_plot='bar'):
        # Detectar se existe coluna de categoria
        category_column = "Category" if "Category" in df.columns else None
        df_filtered = df.copy()

        if category_column:
            UI.header('Filtrar por Categoria:', size=5)
            unique_categories = df[category_column].dropna().unique().tolist()
            selected_category = st.selectbox("Escolha a categoria:", ["Todas"] + unique_categories)

            if selected_category != "Todas":
                df_filtered = df[df[category_column] == selected_category]

        # Selecionar tipo de gráfico
        plot_type = st.selectbox("Tipo de gráfico", ["bar", "pie", "line"], index=["bar", "pie", "line"].index(default_plot))

        # Selecionar colunas disponíveis
        numeric_columns = df_filtered.select_dtypes(include='number').columns.tolist()
        all_columns = df_filtered.columns.tolist()

        if not numeric_columns:
            st.warning("Não há colunas numéricas no DataFrame para plotagem.")
            return

        # Selecionar valor (y) e métrica (x ou labels)
        chosen_value = st.selectbox("Coluna numérica (valor)", numeric_columns)
        metric = st.selectbox("Coluna categórica ou temporal (eixo ou categoria)", [col for col in all_columns if col != chosen_value])

        # Destaque visual opcional
        selected_highlight = None
        if plot_type in ["bar", "pie"]:
            unique_options = df_filtered[metric].dropna().unique().tolist()
            if unique_options:
                selected_highlight = st.selectbox("Destaque (opcional):", ["Nenhum"] + unique_options)
                if selected_highlight == "Nenhum":
                    selected_highlight = None

        # Gerar gráfico
        fig = Data_Visualization._generate_plot(df_filtered, chosen_value, metric, plot_type, selected_highlight)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _generate_plot(data, chosen_value, metric, plot_type, selected_category=None):
        plot_width = 800
        plot_height = 600
        st.session_state.chosen_value = chosen_value
        if plot_type == 'pie':
            fig = px.pie(
                data,
                values=chosen_value,
                names=metric,
                title=f"Distribuição de {chosen_value}",
                color=metric,
                color_discrete_map={selected_category: PRIMARY_COLOR, **{c: SECONDARY_COLOR for c in data[metric].unique() if c != selected_category}} if selected_category else None
            )

        elif plot_type == 'bar':
            fig = px.bar(
                data,
                x=metric,
                y=chosen_value,
                title=f"Comparação de {chosen_value}",
                color=metric,
                color_discrete_map={selected_category: PRIMARY_COLOR, **{c: SECONDARY_COLOR for c in data[metric].unique() if c != selected_category}} if selected_category else None
            )
            fig.update_xaxes(showticklabels=True)

        elif plot_type == 'line':
            fig = px.line(
                data,
                x=metric,
                y=chosen_value,
                title=f"Evolução de {chosen_value}",
                markers=True
            )
            fig.update_traces(line_color=PRIMARY_COLOR)

        else:
            st.warning("Tipo de gráfico não reconhecido.")
            return None

        fig.update_layout(
            width=plot_width,
            height=plot_height,
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="top",
                font=dict(size=12)
            )
        )

        return fig

class UI:
    def metric_box(col, label, value, color):
        html = f"""
        <div style="
            background-color: {color};
            border-radius: 100%;
            width: 150px;
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            font-weight: bold;
            font-size: 20px;
            margin: auto;
            padding: 20px;
        ">
            <div style="font-size: 28px;">{value}</div>
            <div style="font-size: 16px;">{label}</div>
        </div>
        """
        col.markdown(html, unsafe_allow_html=True)

    def data_table(df: pd.DataFrame, title: str = "", key=None):
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            filter=True,
            sortable=True,
            resizable=True,
            wrapText=True,
            autoHeight=True,
            cellStyle={'textAlign': 'center'}
        )
        gb.configure_grid_options(domLayout='normal')
        grid_options = gb.build()

        css = {
            ".ag-header-cell-label": {
                "justify-content": "center !important",
                "font-size": "17px !important",
                "font-weight": "bold !important",
                "color": f"{SECONDARY_COLOR} !important"
            },
            ".ag-header": {
                "background-color": f"{PRIMARY_COLOR} !important"
            },
            ".ag-cell": {
                "background-color": "#ffffff !important",
                "color": f"{SECONDARY_COLOR} !important",
                "font-size": "15px !important",
                "text-align": "center !important",
                "padding": "10px !important",
                "border-bottom": "2px solid #ddd !important",
            },
            ".ag-row-hover .ag-cell": {
                "background-color": f"{HOVER_COLOR} !important"
            },
            ".ag-root-wrapper": {
                "width": "100% !important",
                "min-height": f"{min(400, 56.7*(len(df)+1))}px !important",
                "border": f"1px solid {SECONDARY_COLOR} !important",
                "border-radius": "8px !important"
            },
            ".ag-center-cols-container": {
                "min-width": "100% !important"
            },
        }

        if title:
            st.markdown(
                f"<h3 style='text-align: center; color: {PRIMARY_COLOR}'>{title}</h3>",
                unsafe_allow_html=True
            )

        AgGrid(
            df,
            gridOptions=grid_options,
            theme="alpine",
            height=min(400, 56.7*(len(df)+1)),
            custom_css=css,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            key=key,
        )

    def header(text, color=PRIMARY_COLOR, size=3, target=st):
        target.markdown(
            f"<h{size} style='color:{color};'>{text}</h{size}>", 
            unsafe_allow_html=True
        )

    def sidebar_style():
        st.markdown(f"""
            <style>
                [data-testid="stSidebar"] {{
                    background-color: {SECONDARY_COLOR};
                }}
                [data-testid="stSidebar"] h1, h2, h3, h4, h5, h6,
                [data-testid="stSidebar"] label {{
                    color: white;
                }}
                section[data-testid="stSidebar"] ul li span {{
                    color: {PRIMARY_COLOR} !important;
                    font-weight: bold;
                }}
                .stRadio label div p {{
                    color: white;
                }}
            </style>
        """, unsafe_allow_html=True)

        st.sidebar.image("Logo.png", use_container_width=True)

    def styled_btn():
        st.markdown(f"""
            <style>
                .stButton>button {{
                    background-color: {PRIMARY_COLOR};
                    color: {SECONDARY_COLOR};
                    border: none;
                    padding: 20px 30px;
                    border-radius: 30px;
                    font-size: 30px;
                    cursor: pointer;
                }}
                .stButton>button:hover {{
                    background-color: {HOVER_COLOR};
                    transform: scale(1.03);
                }}
            </style>
        """, unsafe_allow_html=True)

    def editable_table(sheet_name, tab_name, key="editable_table", editable=True):
        session_key = f"{key}_data"

        # Carrega os dados com cache apenas uma vez
        if session_key not in st.session_state:
            df = Data_API.load_data_API(sheet_name, sheet_tab_name=tab_name)
            st.session_state[session_key] = df.copy()

        df = st.session_state[session_key]

        # CSS customizado baseado no data_table
        css = {
            ".ag-header-cell-label": {
                "justify-content": "center !important",
                "font-size": "17px !important",
                "font-weight": "bold !important",
                "color": f"{SECONDARY_COLOR} !important"
            },
            ".ag-header": {
                "background-color": f"{PRIMARY_COLOR} !important"
            },
            ".ag-cell": {
                "background-color": "#ffffff !important",
                "color": f"{SECONDARY_COLOR} !important",
                "font-size": "15px !important",
                "text-align": "center !important",
                "padding": "10px !important",
                "border-bottom": "2px solid #ddd !important",
            },
            ".ag-row-hover .ag-cell": {
                "background-color": f"{HOVER_COLOR} !important"
            },
            ".ag-root-wrapper": {
                "width": "100% !important",
                "min-height": f"{min(400, 56.7*(len(df)+1))}px !important",
                "border": f"1px solid {SECONDARY_COLOR} !important",
                "border-radius": "8px !important"
            },
            ".ag-center-cols-container": {
                "min-width": "100% !important"
            },
        }

        # Título estilizado
        st.markdown(
            f"<h3 style='text-align: center; color: {PRIMARY_COLOR}'>Tabela: {tab_name}</h3>",
            unsafe_allow_html=True
        )

        if editable:
            st.markdown(
                f"<p style='text-align: center; color: {SECONDARY_COLOR}; font-size: 16px;'>✏️ Edite sua tabela:</p>",
                unsafe_allow_html=True
            )
            
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(
                editable=True, 
                resizable=True, 
                filter=True, 
                wrapText=True,
                autoHeight=True,
                cellStyle={'textAlign': 'center'}
            )
            gb.configure_grid_options(domLayout='normal')
            grid_options = gb.build()

            grid_response = AgGrid(
                df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.VALUE_CHANGED,
                editable=True,
                fit_columns_on_grid_load=True,
                theme="alpine",
                custom_css=css,
                height=min(400, 56.7*(len(df)+1)),
                allow_unsafe_jscode=True,
                key=key,
            )

            st.session_state[session_key] = grid_response['data']

            # Botões estilizados
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "➕ Add Row", 
                    key=f"{key}_add_row",
                    help="Adiciona uma nova linha à tabela"
                ):
                    st.session_state[session_key] = pd.concat(
                        [st.session_state[session_key], pd.DataFrame([[""] * len(df.columns)], columns=df.columns)],
                        ignore_index=True,
                    )

            with col2:
                col_name = st.text_input(
                    "Column name:", 
                    key=f"{key}_new_col",
                    placeholder="Digite o nome da nova coluna"
                )
                if st.button(
                    "➕ Add Column", 
                    key=f"{key}_add_col",
                    disabled=not col_name,
                    help="Adiciona uma nova coluna à tabela"
                ) and col_name:
                    if col_name not in df.columns:
                        st.session_state[session_key][col_name] = ""

            with col3:
                if st.button(
                    "💾 Save Changes", 
                    key=f"{key}_save",
                    type="primary",
                    help="Salva as alterações na tabela"
                ):
                    Data_API.clear_single_cache(sheet_name, tab_name)  # limpa o cache dessa aba
                    Data_API.overwrite_sheet(sheet_name, tab_name, st.session_state[session_key])
                    st.success("Tabela atualizada com sucesso.")
        else:
            # Se não for editável, usa o mesmo estilo da data_table
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(
                filter=True,
                sortable=True,
                resizable=True,
                wrapText=True,
                autoHeight=True,
                cellStyle={'textAlign': 'center'}
            )
            gb.configure_grid_options(domLayout='normal')
            grid_options = gb.build()

            AgGrid(
                df,
                gridOptions=grid_options,
                theme="alpine",
                height=min(400, 56.7*(len(df)+1)),
                custom_css=css,
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=True,
                key=key,
            )
