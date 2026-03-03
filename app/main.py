"""Streamlit main entry point — Research Copilot."""
import os
import sys

# Asegurar que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title='Research Copilot',
    page_icon='📚',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.title('📚 Research Copilot')
st.markdown(
    'Sistema RAG para consultar 20 papers académicos sobre '
    '**regulación de IA y privacidad de datos**.'
)

st.markdown('---')

col1, col2, col3 = st.columns(3)
with col1:
    st.info('💬 **Chat**\n\nHaz preguntas en lenguaje natural.')
with col2:
    st.info('📄 **Papers**\n\nExplora el catálogo de 20 artículos.')
with col3:
    st.info('📊 **Analytics**\n\nEstadísticas del corpus y del RAG.')

st.markdown('---')
st.caption('Usa el menú lateral para navegar entre páginas.')
