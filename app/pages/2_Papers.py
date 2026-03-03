"""Página de Papers — Research Copilot."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

st.set_page_config(page_title='Papers — Research Copilot', page_icon='📄', layout='wide')
st.title('📄 Catálogo de Papers')


@st.cache_data
def load_catalog():
    with open('papers/paper_catalog.json', 'r', encoding='utf-8') as f:
        return json.load(f)


catalog = load_catalog()
papers = catalog['papers']

# Filtros en sidebar
with st.sidebar:
    st.header('Filtros')
    years = sorted({p['year'] for p in papers})
    selected_years = st.multiselect('Año', years, default=years)
    all_topics = sorted({t for p in papers for t in p.get('topics', [])})
    selected_topic = st.selectbox('Tema', ['Todos'] + all_topics)

# Filtrar papers
filtered = [
    p for p in papers
    if p['year'] in selected_years
    and (selected_topic == 'Todos' or selected_topic in p.get('topics', []))
]

st.markdown(f'**{len(filtered)} papers** encontrados')
st.markdown('---')

for paper in filtered:
    with st.expander(f"[{paper['id']}] {paper['title']} ({paper['year']})"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Autores:** {', '.join(paper['authors'])}")
            st.markdown(f"**Venue:** {paper['venue']}")
            st.markdown(f"**DOI:** `{paper['doi']}`")
        with col2:
            st.markdown(f"**Archivo:** `{paper['filename']}`")
            if paper.get('topics'):
                st.markdown('**Temas:**')
                for t in paper['topics']:
                    st.markdown(f'  - {t}')
        if paper.get('abstract'):
            st.markdown('**Abstract:**')
            st.markdown(paper['abstract'])
