# login.py
import streamlit as st
import json
from pathlib import Path
from st_pages import Page, show_pages


# =========================
# 🎨 Configuração da Página
# =========================
st.set_page_config(
    page_title="Login System",
    page_icon="🔒",
    layout="centered"
)

# =========================
# 🎨 Cores
# =========================
primary_color = "#ffb6c1"      # Rosa claro
secondary_color = "#416538"    # Verde escuro

# =========================
# 🎨 Estilo CSS
# =========================
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {secondary_color};
            color: white;
        }}
        h1 {{
            color: {primary_color};
            text-align: center;
        }}
        .login-content {{
            text-align: center;
            margin-top: 50px;
        }}
        .stButton>button {{
            background-color: {primary_color};
            color: black;
            border: None;
        }}
        .stTextInput>div>div>input {{
            background-color: white;
            color: black;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 📄 Gerenciamento de Páginas
# =========================
show_pages([
    Page("login.py", "Login", "🏠"),
    Page("pages/bio_main.py", "Painel", ":bear:"),
])

# =========================
# 🔐 Gerenciamento de Credenciais
# =========================
CREDENTIALS_FILE = "credentials.json"

def load_credentials():
    if Path(CREDENTIALS_FILE).exists():
        try:
            with open(CREDENTIALS_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return {}
    else:
        return {}

def save_credentials(credentials):
    with open(CREDENTIALS_FILE, "w") as file:
        json.dump(credentials, file, indent=4)

# 🔥 Garante que o arquivo existe
if not Path(CREDENTIALS_FILE).exists():
    default_credentials = {
        "admin": {
            "email": "admin@empresa.com",
            "password": "admin123"
        },
        "usuario": {
            "email": "usuario@empresa.com",
            "password": "senha123"
        }
    }
    save_credentials(default_credentials)

# =========================
# 🏷️ Estado da Sessão
# =========================
if "connected" not in st.session_state:
    st.session_state.connected = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {}

# =========================
# 🔄 Redirecionamento se já logado
# =========================
if st.session_state.connected:
    st.switch_page("pages/bio_main.py")

# =========================
# 🚪 Interface de Login
# =========================
st.title("Acesso ao Sistema")

with st.container():
    st.subheader("Faça seu login")
    username = st.text_input("Usuário", key="username")
    password = st.text_input("Senha", type="password", key="password")

    login = st.button("Entrar")

    if login:
        credentials = load_credentials()

        if username in credentials:
            stored_password = credentials[username]["password"]
            if password == stored_password:
                st.session_state.connected = True
                st.session_state.user_info = {
                    "email": credentials[username]["email"],
                    "username": username
                }
                st.success(f"Bem-vindo, {username}!")
                st.switch_page("pages/bio_main.py")
            else:
                st.error("Senha incorreta!", icon="⚠️")
        else:
            st.error("Usuário não encontrado!", icon="⚠️")
