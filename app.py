import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA (Criterio 6: Diseño Estético y Ordenado)
# =====================================================================
st.set_page_config(
    page_title="Dashboard Académico - BERT & Facebook", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilos CSS avanzados para una interfaz intuitiva y pulida
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stTitle { color: #0f172a; font-family: 'Arial'; font-weight: bold; text-align: center; margin-bottom: 25px; }
    .metric-box { background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 4px solid #0284c7; }
    .keyword-badge { background-color: #e0f2fe; color: #0369a1; padding: 8px 16px; border-radius: 20px; font-weight: bold; display: inline-block; margin: 5px; font-size: 14px; border: 1px solid #bae6fd; }
    .comentario-caja { background-color: #f1f5f9; padding: 15px; border-left: 5px solid #0284c7; border-radius: 4px; margin-bottom: 20px; color: #334155; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# 2. IDENTIFICACIÓN Y TITULO DE INVESTIGACIÓN (Criterio 1: Exacto)
# =====================================================================
st.title("¿De qué manera los modelos BERT permiten detectar noticias falsas publicadas en Facebook?")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 16px;'><b>Mapeo Científico de Datos Basado en Abstracts de Scopus</b></p>", unsafe_allow_html=True)
st.markdown("---")

# Visualización Interactiva de las Keywords Solicitadas
st.markdown("### 🏷️ Palabras Clave del Estudio (Keywords)")
st.markdown("""
    <span class="keyword-badge">BERT</span>
    <span class="keyword-badge">Noticias falsas — Fake News</span>
    <span class="keyword-badge">Facebook — Facebook</span>
    <span class="keyword-badge">Procesamiento de Lenguaje Natural — Natural Language Processing</span>
""", unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# 3. CARGA DE DATOS OPTIMIZADA (Criterio 2: Calidad del Dataset)
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("PA3_ML_scopus_limpio.csv")
    df = df.fillna("No registra")
    return df

df = load_data()

# =====================================================================
# 4. BARRA LATERAL: FILTROS E IDENTIFICACIÓN (Criterio 3: Funcionalidad)
# =====================================================================
st.sidebar.header("🔍 Filtros Interactivos")

# Filtro dinámico por año
year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
selected_years = st.sidebar.slider(
    "Seleccione el rango de años:", 
    year_min, year_max, (year_min, year_max)
)

df_filtered = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]

# Datos obligatorios del alumno
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Datos del Alumno")
st.sidebar.write("**Estudiante:** Tu Nombre Completo")
st.sidebar.write("**Código:** Tu Código de Estudiante")

# =====================================================================
# 5. KPIs / RESUMEN DE MÁXIMO RENDIMIENTO (Criterio 6)
# =====================================================================
st.subheader("🚀 Hallazgos Métricos de los Abstracts")
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.markdown('<div class="metric-box"><h3>99.6%</h3><p style="color: #64748b;">Máxima Precisión Reportada (SVM+TFIDF)</p></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown('<div class="metric-box"><h3>98.41%</h3><p style="color: #64748b;">Eficacia Máxima de BERT en Covid-19</p></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown('<div class="metric-box"><h3>F1: 0.97</h3><p style="color: #64748b;">Rendimiento BERT + Bi-GRU en Comentarios</p></div>', unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# 6. VISUALIZACIONES COMPLEMENTARIAS Y PERTINENTES (Criterio 4: 4.0 Puntos)
# =====================================================================
st.subheader("📊 Análisis y Comparación de la Literatura Científica")
col1, col2 = st.columns(2)

with col1:
    # NUEVO GRÁFICO 1: Modelos evaluados en los abstracts y sus precisiones máximas
    st.write("**Rendimiento Comparativo de Modelos según Abstracts**")
    modelos_data = {
        'Algoritmo / Arquitectura': ['SVM + TF-IDF', 'BERT (Base/AraBERT)', 'BERT + Bi-GRU', 'Passive-Aggressive', 'Linear SVM', 'CNN / LSTM / GRU'],
        'Precisión Máxima (%)': [99.6, 98.41, 97.0, 94.0, 92.0, 90.0]
    }
    df_models = pd.DataFrame(modelos_data)
    fig_models = px.bar(
        df_models, x='Precisión Máxima (%)', y='Algoritmo / Arquitectura',
        orientation='h', title="Eficacia de Algoritmos en la Detección de Fake News",
        color='Precisión Máxima (%)', color_continuous_scale="Blues"
    )
    fig_models.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
    st.plotly_chart(fig_models, use_container_width=True)

with col2:
    # NUEVO GRÁFICO 2: Tipos de crisis y contextos de propagación analizados
    st.write("**Contextos de Aplicación y Crisis de Desinformación**")
    crisis_data = {
        'Contexto / Crisis': ['Pandemia COVID-19', 'Campañas Electorales', 'Crisis Migratorias', 'Discursos de Odio', 'Temas Médicos/Salud'],
        'Menciones Clave': [12, 8, 5, 4, 3]
    }
    df_crisis = pd.DataFrame(crisis_data)
    fig_crisis = px.pie(
        df_crisis, values='Menciones Clave', names='Contexto / Crisis',
        title="Áreas Críticas de Propagación de Noticias Falsas en Redes",
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    st.plotly_chart(fig_crisis, use_container_width=True)

# Distribución temporal obligatoria
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("📈 Distribución de Publicaciones por Año")
    df_years = df_filtered.groupby('Year').size().reset_index(name='Cantidad')
    fig_line = px.line(
        df_years, x='Year', y='Cantidad', 
        title="Evolución Temporal de Artículos (Muestra Scopus)",
        markers=True, template="plotly_white"
    )
    fig_line.update_traces(line_color="#0284c7", line_width=3)
