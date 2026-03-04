"""Página de Chat — Research Copilot."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title='Chat — Research Copilot', page_icon='💬', layout='wide')
st.title('💬 Chat con los Papers')


@st.cache_resource(show_spinner='Cargando pipeline RAG...')
def get_pipeline():
    return RAGPipeline(
        papers_dir='papers',
        catalog_path='papers/paper_catalog.json',
    )


pipeline = get_pipeline()

# Sidebar: configuración
with st.sidebar:
    st.header('Configuración')
    strategy = st.selectbox(
        'Estrategia de prompt',
        options=['v1', 'v2', 'v3', 'v4'],
        format_func=lambda x: {
            'v1': 'V1 — Delimitadores',
            'v2': 'V2 — JSON output',
            'v3': 'V3 — Few-shot',
            'v4': 'V4 — Chain of thought',
        }[x],
    )
    n_results = st.slider('Chunks a recuperar', min_value=3, max_value=10, value=5)
    st.markdown('---')
    if st.button('Limpiar historial'):
        st.session_state.messages = []
        st.rerun()

# Historial de conversación
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if 'sources' in msg:
            with st.expander('Fuentes'):
                for src in msg['sources']:
                    st.write(f'- {src}')

# Input del usuario
if question := st.chat_input('Haz una pregunta sobre regulación de IA y privacidad...'):
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        with st.spinner('Buscando en la base de datos local...'):
            # Forzamos un threshold estricto de 0.50 para evitar alucinaciones
            result = pipeline.query(question, strategy=strategy, n_results=n_results, threshold=0.50)
        
        st.markdown(result['answer'])
        
        # Mostramos información de depuración para que el usuario sepa qué está pasando
        if result['n_chunks_used'] > 0:
            with st.expander(f"📚 Fuentes Verificadas ({len(result['sources'])} documentos reales)"):
                for src in result['sources']:
                    st.write(f'✅ **{src}**')
        else:
            st.warning("⚠️ El sistema bloqueó esta respuesta porque no encontró evidencia en tus PDFs.")

    st.session_state.messages.append({
        'role': 'assistant',
        'content': result['answer'],
        'sources': result['sources']
    })

