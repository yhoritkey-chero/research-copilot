"""Página de Analytics — Research Copilot."""
import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

st.set_page_config(page_title='Analytics — Research Copilot', page_icon='📊', layout='wide')
st.title('📊 Analytics del Corpus')

from src.vectorstore.chroma_store import ChromaVectorStore
store = ChromaVectorStore(persist_directory='./chroma_db')
store.create_collection('research_papers')
chunks_count = store.count()
st.sidebar.metric('Chunks en Vector DB', chunks_count)


@st.cache_data
def load_catalog():
    with open('papers/paper_catalog.json', 'r', encoding='utf-8') as f:
        return json.load(f)


catalog = load_catalog()
papers = catalog['papers']

# Métricas generales
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total papers', len(papers))
col2.metric('Años cubiertos', f"{min(p['year'] for p in papers)}–{max(p['year'] for p in papers)}")
col3.metric('Venues únicas', len({p['venue'] for p in papers}))
col4.metric('Temas únicos', len({t for p in papers for t in p.get('topics', [])}))

st.markdown('---')

# Distribución por año
st.subheader('Papers por año')
year_counts = Counter(p['year'] for p in papers)
years_sorted = sorted(year_counts.keys())
st.bar_chart({str(y): year_counts[y] for y in years_sorted})

# Distribución por venue
st.subheader('Papers por venue')
venue_counts = Counter(p['venue'] for p in papers)
st.bar_chart(dict(venue_counts.most_common(10)))

# Temas más frecuentes
st.subheader('Temas más frecuentes')
topic_counts = Counter(t for p in papers for t in p.get('topics', []))
st.bar_chart(dict(topic_counts.most_common(15)))

# Tabla resumen
st.subheader('Tabla resumen')
rows = [
    {
        'ID': p['id'],
        'Título': p['title'][:60] + '...' if len(p['title']) > 60 else p['title'],
        'Año': p['year'],
        'Venue': p['venue'],
        'N° temas': len(p.get('topics', [])),
    }
    for p in papers
]
st.dataframe(rows, use_container_width=True)
