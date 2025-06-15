import datetime
import glob
import os
from functools import reduce
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yaml

from fpdf import FPDF
from scipy.stats import fisher_exact, mannwhitneyu, shapiro
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


PRIMARY_COLOR = "#ffb6c1"
SECONDARY_COLOR = "#416538"
HOVER_COLOR = "#ffc1cc"

# Constantes para configuração do mapa
UBERLANDIA_COORDS = dict(lat=-18.9186, lon=-48.2772)
DEFAULT_ZOOM = 8
MAP_STYLE = "carto-positron" 


class Data_API:
    _manual_cache = {}
    @staticmethod
    @st.cache_data(ttl=43200)
    def load_data_API(sheet_name, date_field=None, columns_to_clean=[], date_format=None, sheet_tab_name=None):
        # Carrega as credenciais diretamente do secrets
        service_account_info = st.secrets["google_sheets"]
        credentials = Credentials.from_service_account_info(
            service_account_info,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        client = gspread.authorize(credentials)
        spreadsheet = client.open(sheet_name)
        sheet = spreadsheet.worksheet(sheet_tab_name) if sheet_tab_name else spreadsheet.sheet1
        valores = sheet.get_all_values()
        df = pd.DataFrame(valores[1:], columns=valores[0])


        if columns_to_clean:
            for column in columns_to_clean:
                df[column] = Data_op.clean_numeric(df[column])

        # Armazena no cache interno
        Data_API._manual_cache[(sheet_name, sheet_tab_name)] = df.copy()

        return df

    @staticmethod
    def overwrite_sheet(sheet_name, sheet_tab_name, df):
        service_account_info = st.secrets["google_sheets"]
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


class Data_op:
    @staticmethod
    def executar_teste(df1, df2, categories, number_column, test_type='shapiro', limiar=1, chosen_class=None):
        merged = Data_op.merge_tables(df1, df2, on_df1='Sample', on_df2='Sample')
        merged[number_column] = Data_op.clean_numeric(merged[number_column])
        
        if test_type == 'shapiro':
            return Data_op._teste_shapiro(merged, categories, number_column)
        elif test_type == 'mannwhitney':
            # Chama a função auxiliar de Mann–Whitney
            return Data_op._teste_mannwhitney(merged, categories, number_column)
        elif test_type == 'fisher':
            return Data_op._teste_fisher(merged, categories[0], number_column, limiar, chosen_class[0])
        else:
            raise ValueError("Teste inválido. Use 'shapiro', 'mannwhitney' ou 'fisher'.")
        
    def _teste_shapiro(df, categories, number_column):
        if not isinstance(df, pd.DataFrame):
            st.error(f"Erro: tipo inválido para df: {type(df)}")
            return pd.DataFrame()

        resultados = []

        for categoria in categories:
            try:
                dados = df[df['Categories'] == categoria][number_column].dropna()
                if len(dados) > 2:
                    stat, pval = shapiro(dados)
                    resultados.append({
                        'Categoria': categoria,
                        'W': stat,
                        'p-valor': pval,
                        'Normal (α=0,05)': 'Yes' if pval > 0.05 else 'No',
                        'n': len(dados)
                    })
            except Exception as e:
                st.error(f"Erro ao processar categoria '{categoria}': {e}")

        if not resultados:
            st.warning("Nenhuma categoria com dados suficientes para o teste de Shapiro-Wilk.")
            return pd.DataFrame(columns=['Categoria', 'W', 'p-valor', 'Normal (α=0,05)', 'n'])

        return [pd.DataFrame(resultados)]

    def _teste_mannwhitney(df, categories, number_column):
        if len(categories) != 2:
            raise ValueError("Para Mann–Whitney, selecione exatamente duas categorias.")

        cat_a, cat_b = categories
        df_a = df[df['Categories'] == cat_a][[number_column]].dropna().copy()
        df_b = df[df['Categories'] == cat_b][[number_column]].dropna().copy()

        # Teste de Mann–Whitney
        stat, pval = mannwhitneyu(df_a[number_column], df_b[number_column], alternative='two-sided')
        sum_ranks = f"{df_a[number_column].sum():.0f}, {df_b[number_column].sum():.0f}"

        # Sumário do p-valor
        if pval < 0.0001:
            summary = '****'
        elif pval < 0.001:
            summary = '***'
        elif pval < 0.01:
            summary = '**'
        elif pval < 0.05:
            summary = '*'
        else:
            summary = 'ns'

        # Tabela do teste
        table_test = pd.DataFrame({
            '': [
                'P value',
                'Exact or approximate P value?',
                'P value summary',
                'Significantly different (P < 0.05)?',
                'One- or two-tailed P value?',
                'Sum of values (proxy for ranks) in A, B',
                'Mann–Whitney U'
            ],
            'Value': [
                f'{pval:.4f}',
                'Exact',
                summary,
                'Yes' if pval < 0.05 else 'No',
                'Two-tailed',
                sum_ranks,
                f'{stat:.4f}'
            ]
        })

        # Tabela de diferenças de medianas
        med_a = df_a[number_column].median()
        med_b = df_b[number_column].median()
        diff_median = pd.DataFrame({
            '': [
                f"Median of {cat_a}, n={len(df_a)}",
                f"Median of {cat_b}, n={len(df_b)}",
                'Difference: Actual',
                'Difference: Hodges-Lehmann (aprox.)'
            ],
            'Value': [
                f'{med_a:.4f}',
                f'{med_b:.4f}',
                f'{(med_a - med_b):.4f}',
                f'{((med_a + med_b) / 2):.4f}'
            ]
        })

        return table_test.reset_index(), diff_median.reset_index()

    def _teste_fisher(df, category, number_column, limiar, chosen_class):
        """
        Executa o teste de Fisher e retorna uma descrição markdown + tabela de contingência como DataFrame.
        """
        df_category = df[df['Categories'] == category].copy()

        if df_category.empty or chosen_class not in df_category.columns or number_column not in df_category.columns:
            return None  # Dados inválidos

        grupo_bin = df_category[number_column] > limiar

        categorias = df_category[chosen_class].dropna().unique()
        if len(categorias) != 2:
            return None  # Precisamos exatamente de duas classes para o teste de Fisher

        classe_0, classe_1 = sorted(categorias)
        classe_series = df_category[chosen_class]

        # Tabela de contingência
        contingencia = pd.crosstab(classe_series, grupo_bin)
        if contingencia.shape != (2, 2):
            return None  # Contingência inválida para teste de Fisher

        # Valores absolutos
        c0_ig_neg = contingencia.loc[classe_0, False]
        c0_ig_pos = contingencia.loc[classe_0, True]
        c1_ig_neg = contingencia.loc[classe_1, False]
        c1_ig_pos = contingencia.loc[classe_1, True]
        total_0 = c0_ig_neg + c0_ig_pos
        total_1 = c1_ig_neg + c1_ig_pos
        total_neg = c0_ig_neg + c1_ig_neg
        total_pos = c0_ig_pos + c1_ig_pos
        total_all = total_neg + total_pos

        # Teste de Fisher
        oddsratio, pval = fisher_exact(contingencia)

        # Resumo p-valor
        if pval < 0.0001:
            summary = '****'
        elif pval < 0.001:
            summary = '***'
        elif pval < 0.01:
            summary = '**'
        elif pval < 0.05:
            summary = '*'
        else:
            summary = 'ns'

        # Markdown com resumo
        markdown_text = f"""
        <span style="color:{SECONDARY_COLOR}; font-weight:600; font-size:20px">Teste de Associação - Fisher</span>

        <ul style="list-style-type:none; padding-left: 0;">
        <li><span style="color:{PRIMARY_COLOR}; font-weight:600">Table Analyzed:</span> {number_column}</li>
        <li><span style="color:{PRIMARY_COLOR}; font-weight:600">Comparison:</span> {chosen_class}</li>
        <li><span style="color:{PRIMARY_COLOR}; font-weight:600">Test:</span> Fisher's exact test</li>
        <li><span style="color:{PRIMARY_COLOR}; font-weight:600">P value:</span> {pval:.4f}</li>
        <li><span style="color:{PRIMARY_COLOR}; font-weight:600">P value summary:</span> {summary}</li>
        <li><span style="color:{PRIMARY_COLOR}; font-weight:600">Statistically significant (P < 0.05)?</span> {"✅ Yes" if pval < 0.05 else "❌ No"}</li>
        <li><span style="color:{PRIMARY_COLOR}; font-weight:600">One- or two-sided:</span> Two-sided</li>
        <li><span style="color:{PRIMARY_COLOR}; font-weight:600">Data analyzed:</span> {classe_0} vs {classe_1}</li>
        <li><span style="color:{PRIMARY_COLOR}; font-weight:600">Group (IG status):</span> IG-, IG+</li>
        </ul>
        """
        st.markdown(markdown_text, unsafe_allow_html=True)

        # Tabela descritiva
        tabela_descritiva = pd.DataFrame({
            'Table Analyzed': [number_column],
            'Comparison': [chosen_class],
            'Test': ["Fisher's exact test"],
            'P value': [f'{pval:.4f}'],
            'P values summary': [summary],
            'Statistically significant (P < 0.05)?': ["Yes" if pval < 0.05 else "No"],
            'Data analyzed': [f'{classe_0} vs {classe_1}']
        })

        # Tabela com valores absolutos
        tabela_resultado = pd.DataFrame({
            'IG-': [c0_ig_neg, c1_ig_neg, total_neg],
            'IG+': [c0_ig_pos, c1_ig_pos, total_pos],
            'Total': [total_0, total_1, total_all]
        }, index=[classe_0, classe_1, 'Total'])

        # Percentuais por linha
        row_percent = tabela_resultado.loc[[classe_0, classe_1]].iloc[:, :2].div(
            tabela_resultado.loc[[classe_0, classe_1]]['Total'], axis=0
        ) * 100
        row_percent = row_percent.round(2)
        row_percent['Total'] = row_percent.sum(axis=1)
        row_percent.index.name = 'Row %'
        row_percent.reset_index(inplace=True)

        # Percentuais por coluna
        column_percent = tabela_resultado.loc[[classe_0, classe_1]].iloc[:, :2].div(
            tabela_resultado.loc['Total'][['IG-', 'IG+']], axis=1
        ) * 100
        column_percent = column_percent.round(2)
        column_percent['Total'] = column_percent.sum(axis=1)
        column_percent.index = [classe_0, classe_1]
        column_percent.loc['Total'] = column_percent.sum()
        column_percent.index.name = 'Column %'
        column_percent.reset_index(inplace=True)

        return [
            tabela_resultado.reset_index(),
            tabela_descritiva,
            row_percent,
            column_percent
        ]
    @staticmethod
    def clean_numeric(series):
        if series.dtype == object or series.dtype == "string":
            series = series.replace('%', '', regex=True).str.strip()
            series = series.str.replace(',', '.', regex=False)
            series = pd.to_numeric(series, errors='coerce')
        elif not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors='coerce')
        
        return series

    def merge_tables(df1, df2, on_df1="Sample", on_df2="Sample", suffixes=('_df1', '_df2')):
        """
        Realiza o merge entre duas tabelas tratando as colunas de junção como strings em maiúsculas
        
        Parâmetros:
        df1, df2 -- DataFrames para unir
        on_df1, on_df2 -- nomes das colunas de junção
        suffixes -- sufixos para colunas duplicadas
        
        Retorna:
        DataFrame unido
        """
        # Criar cópias para não alterar os originais
        df1_temp = df1.copy()
        df2_temp = df2.copy()
        
        # Converter colunas de junção para string e maiúsculas
        df1_temp[on_df1] = df1_temp[on_df1].astype(str).str.upper().str.strip()
        df2_temp[on_df2] = df2_temp[on_df2].astype(str).str.upper().str.strip()
        
        # Realizar o merge
        merged = pd.merge( df1_temp, df2_temp, left_on=on_df1, right_on=on_df2, how='inner', suffixes=suffixes)
        
        # Verificar se o merge foi bem sucedido
        if merged.empty:
            raise ValueError("Nenhuma correspondência encontrada após o merge. Verifique as colunas de junção.")
        
        # Remover coluna duplicada se os nomes forem diferentes
        if on_df1 != on_df2:
            merged = merged.drop(columns=[on_df2])
        
        return merged


class Data_info:
    def get_tuple_by_sheet_tab_name(dataframes, sheet_tab_name_alvo):
        return next(
            (t for t in dataframes if t[2] == sheet_tab_name_alvo),
            None  # valor padrão se não encontrar
        )

    def comparar_samples(df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Compara a coluna 'Sample' do df1 com a do df2 e imprime métricas úteis:
        - Quantas amostras de df1 estão em df2
        - Porcentagem de match
        - Quais categorias estão associadas às amostras em comum
        """
        # Garantir colunas esperadas
        if 'Sample' not in df1.columns or 'Sample' not in df2.columns:
            st.write("Erro: Ambos os DataFrames precisam ter a coluna 'Sample'.")
            return
        
        if 'Categories' not in df2.columns:
            st.write("Erro: O df2 precisa ter a coluna 'Categories'.")
            return

        samples_df1 = set(df1['Sample'].dropna())
        samples_df2 = set(df2['Sample'].dropna())

        intersecao = samples_df1 & samples_df2

        total_df1 = len(samples_df1)
        total_em_comum = len(intersecao)
        percentual = (total_em_comum / total_df1) * 100 if total_df1 > 0 else 0

        # Filtrar df2 pelas amostras em comum para extrair categorias
        df2_filtrado = df2[df2['Sample'].isin(intersecao)]
        categorias_validas = df2_filtrado['Categories'].dropna().unique()
        col1, col2, col3 = st.columns(3)
        UI.metric_box(col1, 'Total',total_df1 )
        UI.metric_box(col2, 'Total',total_em_comum )
        UI.metric_box(col3, 'Total',f'{percentual:.2f}%' )
        st.write(f" - Categorias disponíveis com base nessas amostras: {list(categorias_validas)}")

    def load_csv_interactively():
        if "load_csv" not in st.session_state or not st.session_state["load_csv"]:
            return None  # Não renderiza se a flag não está ativa

        st.warning("⚠️ Atenção: essa ação irá substituir todos os dados da tabela original!")

        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv", key="csv_uploader")
        df_loaded = None

        if uploaded_file is not None:
            try:
                df_loaded = pd.read_csv(uploaded_file)
                st.success("✅ CSV carregado com sucesso!")

                # Reseta a flag após o carregamento bem-sucedido
                st.session_state["load_csv"] = False
            except Exception as e:
                st.error(f"❌ Erro ao carregar CSV: {e}")

        return df_loaded


class Plot_Gen:
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
        fig = Plot_Gen._generate_plot(df_filtered, chosen_value, metric, plot_type, selected_highlight)
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

    def format_coords_as_float(data, lat_col="latitude", lon_col="longitude"):
        def convert_coord(x):
            # Verifica se o valor é None, vazio ou string vazia
            if x is None or str(x).strip() == "":
                return None  # ou np.nan se estiver usando pandas
            
            x_str = str(x).replace(',', '.').strip()  # Remove espaços
            
            # Se já tem ponto decimal, converte diretamente
            if '.' in x_str:
                try:
                    return round(float(x_str), 7)
                except ValueError:
                    return None  # ou np.nan
            
            # Se não tem ponto decimal, divide por 1e7
            try:
                return round(float(x_str) / 1e7, 7)
            except ValueError:
                return None  # ou np.nan
        
        # Aplica a conversão nas colunas
        data[lat_col] = data[lat_col].apply(convert_coord)
        data[lon_col] = data[lon_col].apply(convert_coord)
        
        return data



    def generate_map_plot(data, category_col=None, metric_col=None, selected_category=None):
        lat_col = "latitude"
        lon_col = "longitude"
        data = Plot_Gen.format_coords_as_float(data, lat_col, lon_col)
        
        # Configuração de cores com destaque para selecionados
        color_discrete_map = None
        if selected_category and category_col:
            color_discrete_map = {
                selected_category: PRIMARY_COLOR,
                **{c: SECONDARY_COLOR for c in data[category_col].unique() if c != selected_category}
            }

        fig = px.scatter_mapbox(
            data,
            lat=lat_col,
            lon=lon_col,
            color=category_col,
            size=metric_col,
            size_max=15,  # Controla o tamanho máximo dos marcadores
            hover_name=category_col,
            hover_data={col: True for col in data.columns},  # Mostrar todos os dados
            zoom=DEFAULT_ZOOM,
            center=UBERLANDIA_COORDS,  # Foco em Uberlândia
            height=600,
            color_discrete_map=color_discrete_map,
            opacity=0.8,  # Transparência para melhor visualização
            mapbox_style=MAP_STYLE
        )

        fig.update_layout(
            title=dict(
                text="Mapa de Pontos - Uberlândia/Região",
                x=0.5,
                font=dict(size=20)
            ),
            margin={"r": 20, "t": 60, "l": 20, "b": 20},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Melhorar dicas de ferramenta
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br><br>" +
            "<b>Latitude</b>: %{lat:.4f}<br>" +
            "<b>Longitude</b>: %{lon:.4f}<br>" +
            "<extra></extra>"
        )
        
        return fig

    @staticmethod
    def generate_map_heatmap(data, weight_col=None):
        lat_col = "latitude"
        lon_col = "longitude"
        data = Plot_Gen.format_coords_as_float(data, lat_col, lon_col)
        
        # Filtro para região de Uberlândia (opcional)
        # data = data.query("-19.5 <= latitude <= -18.0 and -49.0 <= longitude <= -47.5")
        
        fig = px.density_mapbox(
            data,
            lat=lat_col,
            lon=lon_col,
            z=weight_col,
            radius=18,  # Ajuste fino para melhor granularidade
            zoom=DEFAULT_ZOOM,
            center=UBERLANDIA_COORDS,  # Foco central
            mapbox_style=MAP_STYLE,
            height=600,
            opacity=0.7,  # Camada semi-transparente
            color_continuous_scale="Viridis"  # Escala de cores moderna
        )

        fig.update_layout(
            title=dict(
                text="Heatmap de Densidade - Uberlândia/Região",
                x=0.5,
                font=dict(size=20)
            ),
            margin={"r": 20, "t": 60, "l": 20, "b": 20},
            coloraxis_colorbar=dict(
                title="Densidade",
                thickness=20
            )
        )
        
        return fig

class UI:
    def metric_box(col, label: str, value, color: str = PRIMARY_COLOR) -> None:
        '''
            Renders a circular metric box with a label and value using custom HTML.
        '''
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

    def show_table(df: pd.DataFrame, title: str = "", key = None, category = None) -> None:
        '''
            Displays a DataFrame using AgGrid with custom CSS and optional title.
        '''
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
        gb.configure_grid_options(autoSizeAllColumns=True)
        gb.configure_grid_options(suppressColumnVirtualisation=True)
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
                "border": f"1px solid {SECONDARY_COLOR} !important",
                "border-radius": "8px !important"
            }
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
            fit_columns_on_grid_load=False, 
            key=key,
        )

    def header(text: str, color: str = PRIMARY_COLOR, size: int = 3, target = st) -> None:
        '''
            Displays a colored header using markdown in Streamlit.
        '''
        target.markdown(
            f"<h{size} style='color:{color};'>{text}</h{size}>", 
            unsafe_allow_html=True
        )

    def sidebar_style() -> None:
        '''
            Applies custom styles to the Streamlit sidebar and displays the logo.
        '''
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

    def styled_btn() -> None:
        '''
            Applies custom styling to Streamlit buttons for consistent UI.
        '''
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

    def editable_table(sheet_name: str, tab_name: str, key: str = "editable_table", editable: bool = True) -> None:
        '''
            Displays an editable (or read-only) table from a sheet with options to add/edit rows and columns.
        '''
        session_key = f"{key}_data"

        if session_key not in st.session_state:
            df = Data_API.load_data_API(sheet_name, sheet_tab_name=tab_name)
            st.session_state[session_key] = df.copy()

        df = st.session_state[session_key]

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
                    Data_API.clear_single_cache(sheet_name, tab_name)
                    Data_API.overwrite_sheet(sheet_name, tab_name, st.session_state[session_key])
                    st.success("Tabela atualizada com sucesso.")
        else:
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

class MenuEstatistico:
    @staticmethod
    def menu_shapiro(dfs_disponiveis: Dict[str, pd.DataFrame]):
        '''
            Builds the interface for selecting parameters to perform the Shapiro-Wilk test
        '''
        UI.header("🧪 Shapiro–Wilk")
        tabela_valores = st.selectbox("Tabela de valores numéricos", list(dfs_disponiveis.keys()), key="mw_val_t1")
        df1 = dfs_disponiveis[tabela_valores]
        df2 = dfs_disponiveis["Pacientes"]

        comuns = set(df1['Sample'].dropna()) & set(df2['Sample'].dropna())
        df1 = df1[df1['Sample'].isin(comuns)]
        df2 = df2[df2['Sample'].isin(comuns)]

        categorias_validas = df2['Categories'].dropna().unique().tolist()
        
        valor_col = st.selectbox("Coluna de Valores Numericos", df1.columns.tolist(), key="shapiro_val_col")

        categoria = st.multiselect(
            "Selecione até duas categorias para comparar", categorias_validas,
            default=categorias_validas[:2],
            key="shapiro_choices"
        )

        return {
            "df1": df1,
            "df2": df2,
            "number_column": valor_col,
            "categories": categoria,
            "chosen_class": None
        }

    @staticmethod
    def menu_mannwhitney(dfs_disponiveis: Dict[str, pd.DataFrame]):
        '''
            Builds the interface for selecting parameters to perform the Mann–Whitney test
        '''
        UI.header("🔍 Mann–Whitney")
        tabela_valores = st.selectbox("Tabela de valores numéricos", list(dfs_disponiveis.keys()), key="mw_val_t1")
        df1 = dfs_disponiveis[tabela_valores]
        df2 = dfs_disponiveis["Pacientes"]

        comuns = set(df1['Sample'].dropna()) & set(df2['Sample'].dropna())
        df1 = df1[df1['Sample'].isin(comuns)]
        df2 = df2[df2['Sample'].isin(comuns)]

        categorias_validas = df2['Categories'].dropna().unique().tolist()
        
        valor_col = st.selectbox("Coluna de Valores Numericos", df1.columns.tolist(), key="mw_val_col")

        categoria = st.multiselect(
            "Selecione até duas categorias para comparar", categorias_validas,
            default=categorias_validas[:2],
            key="mw_choices"
        )
        if len(categoria) > 2:
            st.error("Selecione no máximo duas categorias.")
            return None

        return {
            "df1": df1,
            "df2": df2,
            "number_column": valor_col,
            "categories": categoria,
            "chosen_class": None
        }

    @staticmethod
    def menu_fisher(dfs_disponiveis: Dict[str, pd.DataFrame]):
        '''
            Builds the interface for selecting parameters to perform the Fisher’s exact test
        '''
        UI.header("🎲 Fisher’s Exact Test")
        df_elisa = dfs_disponiveis["Elisa"]
        df_pac = dfs_disponiveis["Pacientes"]

        elisa_column = st.selectbox("Coluna da Elisa (Tabela 1)", df_elisa.columns.tolist())

        samples_df1 = set(df_elisa['Sample'].dropna())
        samples_df2 = set(df_pac['Sample'].dropna())
        intersecao = samples_df1 & samples_df2
        df2_filtrado = df_pac[df_pac['Sample'].isin(intersecao)].copy()

        categorias_validas = df2_filtrado['Categories'].dropna().unique().tolist()
        categoria = st.selectbox(
            "Filtrar por categoria específica (opcional)",
            [None] + sorted(categorias_validas),
            format_func=lambda x: "Todos" if x is None else x
        )

        colunas_binarias = df_pac.columns[df_pac.nunique() <= 10].tolist()
        chosen_class = st.selectbox(
            "Coluna de Classificação (ex: Gênero)",
            [None] + colunas_binarias,
            format_func=lambda x: "Nenhuma" if x is None else x
        )

        return {
            "df1": df_elisa,
            "df2": df_pac,
            "number_column": elisa_column,
            "categories": [categoria],
            "chosen_class": [chosen_class]
        }


class GoogleSheetExporter:
    def __init__(self, service_account_info: dict):
        self.scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        self.credentials = Credentials.from_service_account_info(
            service_account_info, scopes=self.scopes
        )
        self.client = gspread.authorize(self.credentials)
        self.drive_service = build('drive', 'v3', credentials=self.credentials)

    def criar_ou_pegar_subpasta(self, nome_subpasta: str, id_pasta_pai: str) -> str:
        query = (
            f"'{id_pasta_pai}' in parents and "
            f"mimeType='application/vnd.google-apps.folder' and "
            f"name='{nome_subpasta}' and trashed=false"
        )
        response = self.drive_service.files().list(
            q=query, spaces='drive', fields='files(id, name)'
        ).execute()

        arquivos = response.get('files', [])
        if arquivos:
            return arquivos[0]['id']
        else:
            file_metadata = {
                'name': nome_subpasta,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [id_pasta_pai]
            }
            folder = self.drive_service.files().create(
                body=file_metadata, fields='id').execute()
            return folder['id']

    def salvar_resultado(self, resultado, nome_teste: str, pasta_principal_id: str, nome_personalizado: str = None) -> str:
        # 1. Criar ou pegar subpasta do teste
        subpasta_id = self.criar_ou_pegar_subpasta(nome_teste, pasta_principal_id)

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d_%Hh%M')

        nome_arquivo = nome_personalizado or f"{nome_teste}_{timestamp}"

        # 3. Criar planilha
        sh = self.client.create(nome_arquivo)
        file_id = sh.id

        # 4. Mover planilha para subpasta
        file = self.drive_service.files().get(fileId=file_id, fields='parents').execute()
        previous_parents = ",".join(file.get('parents'))
        self.drive_service.files().update(
            fileId=file_id,
            addParents=subpasta_id,
            removeParents=previous_parents,
            fields='id, parents'
        ).execute()

        # 5. Formatar resultado como lista de DataFrames com nomes de abas
        if isinstance(resultado, pd.DataFrame):
            resultado = [(resultado, "Resultado")]
        else:
            resultado = [
                (df, f"Aba_{i+1}") if not isinstance(df, tuple) else df
                for i, df in enumerate(resultado)
            ]

        # 6. Preencher primeira aba (substitui default)
        ws = sh.get_worksheet(0)
        ws.update_title(resultado[0][1])
        ws.clear()
        ws.update([resultado[0][0].columns.tolist()] + resultado[0][0].values.tolist())

        # 7. Preencher demais abas
        for df, aba_nome in resultado[1:]:
            nova_ws = sh.add_worksheet(title=aba_nome, rows=df.shape[0]+10, cols=df.shape[1]+10)
            nova_ws.update([df.columns.tolist()] + df.values.tolist())

        # 8. Retornar URL da planilha criada
        return f"https://docs.google.com/spreadsheets/d/{file_id}/edit"


