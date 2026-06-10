import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA (Criterio 6: Empatía e Interfaz Intuitiva)
# =====================================================================
st.set_page_config(
    page_title="Dashboard Académico - BERT & Facebook", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilos CSS 
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stTitle { color: #0f172a; font-family: 'Arial'; font-weight: bold; }
    .metric-box { background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 4px solid #0284c7; }
    .keyword-badge { background-color: #e0f2fe; color: #0369a1; padding: 6px 12px; border-radius: 20px; font-weight: bold; display: inline-block; margin: 5px; font-size: 13px; border: 1px solid #bae6fd; }
    .comentario-caja { background-color: #f1f5f9; padding: 15px; border-left: 5px solid #0284c7; border-radius: 4px; margin-bottom: 20px; color: #334155; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# 2. IDENTIFICACIÓN Y PREGUNTA DE INVESTIGACIÓN 
# =====================================================================
st.title("¿De qué manera los modelos BERT permiten detectar noticias falsas publicadas en Facebook?")
st.markdown("🔍 **Análisis Avanzado de Minería de Datos y Mapeo Científico**")
st.markdown("---")

# Despliegue estético de los Keywords Clave solicitados
st.markdown("### 🏷️ Palabras Clave del Estudio (Keywords)")
st.markdown("""
    <span class="keyword-badge">BERT</span>
    <span class="keyword-badge">Noticias falsas — Fake News</span>
    <span class="keyword-badge">Facebook — Facebook</span>
    <span class="keyword-badge">Procesamiento de Lenguaje Natural — Natural Language Processing</span>
""", unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# 3. CARGA DE DATOS 
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("PA3_ML_scopus_limpio.csv")
    df = df.fillna("No registra")
    # Limpieza de espacios en blanco en columnas clave para evitar errores
    if 'Country' in df.columns:
        df['Country'] = df['Country'].str.strip()
    return df

df = load_data()

# =====================================================================
# 4. BARRA LATERAL: FILTROS E IDENTIFICACIÓN 
# =====================================================================
st.sidebar.header("🔍 Filtros Dinámicos")

# Filtro interactivo por rango de años
year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
selected_years = st.sidebar.slider(
    "Filtrar rango de años de publicación:", 
    year_min, year_max, (year_min, year_max)
)

# Aplicar filtro dinámico al conjunto de datos
df_filtered = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]

# Identificación Obligatoria del Alumno con tus datos asignados
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Datos del Alumno")
st.sidebar.write("**Estudiante:** Grupo 2")
st.sidebar.write("**Código:** 6817")

# =====================================================================
# 5. KPIs / SECCIÓN RESUMEN EJECUTIVO 
# =====================================================================
st.subheader("🚀 Indicadores Clave del Dataset")
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    total_articulos = len(df_filtered)
    st.markdown(f'<div class="metric-box"><h3>{total_articulos} Artículos</h3><p style="color: #64748b;">Muestra de Literatura Científica</p></div>', unsafe_allow_html=True)
with kpi2:
    # Evitamos contar "Unknown" o vacíos para calcular el país líder real
    df_valid_countries = df_filtered[df_filtered['Country'] != "No registra"]
    pais_lider = df_valid_countries['Country'].value_counts().idxmax() if not df_valid_countries.empty else "No registra"
    st.markdown(f'<div class="metric-box"><h3>{pais_lider}</h3><p style="color: #64748b;">País con Mayor Producción</p></div>', unsafe_allow_html=True)
with kpi3:
    citas_totales = int(df_filtered['Cited by'].sum()) if 'Cited by' in df_filtered.columns else 0
    st.markdown(f'<div class="metric-box"><h3>{citas_totales} Citas</h3><p style="color: #64748b;">Impacto Total en la Comunidad</p></div>', unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# 6. NUEVA SECCIÓN: HALLAZGOS TÉCNICOS EXTRAÍDOS DE LOS ABSTRACTS REALES
# =====================================================================
st.subheader("📊 Mapeo de Evidencia Científica (Datos Extraídos de Abstracts)")
col_abs1, col_abs2 = st.columns(2)

with col_abs1:
    st.write("**Eficacia Comparativa de Algoritmos en la Detección de Fake News**")
    # Datos de precisión explícitos de la muestra enviada
    modelos_data = {
        'Algoritmo / Arquitectura': ['SVM + TF-IDF (Máximo)', 'BERT (COVID-19 Dataset)', 'BERT + Bi-GRU (FNID)', 'Passive-Aggressive', 'Linear SVM', 'BERT + Bi-GRU (FNFD)', 'CNN / LSTM Baseline'],
        'Precisión Máxima (%)': [99.6, 98.41, 97.0, 94.0, 92.0, 91.0, 90.0]
    }
    df_models = pd.DataFrame(modelos_data)
    fig_models = px.bar(
        df_models, x='Precisión Máxima (%)', y='Algoritmo / Arquitectura',
        orientation='h', title="Porcentaje de Accuracy / F1-Score por Arquitectura",
        color='Precisión Máxima (%)', color_continuous_scale="Blues"
    )
    fig_models.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
    st.plotly_chart(fig_models, use_container_width=True)

with col_abs2:
    st.write("**Ejes de Propagación de Desinformación y Modelos Multilingües**")
    # Contextos de crisis y variantes lingüísticas analizadas en los artículos
    crisis_data = {
        'Contexto Temático': ['Pandemia COVID-19 / Infodemia', 'Campañas Electorales y Política', 'Grupos Médicos y de Salud', 'Crisis Migratorias / Minorías', 'Multimodalidad (Memes + Texto)'],
        'Artículos Críticos': [14, 9, 6, 5, 3]
    }
    df_crisis = pd.DataFrame(crisis_data)
    fig_crisis = px.pie(
        df_crisis, values='Artículos Críticos', names='Contexto Temático',
        title="Distribución de Entornos de Desinformación Evaluados",
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    st.plotly_chart(fig_crisis, use_container_width=True)

st.markdown("---")

# =====================================================================
# 7. VISUALIZACIONES DINÁMICAS ORIGINALES 
# =====================================================================
st.subheader("📈 Métricas Estructuradas del Repositorio")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución de Publicaciones por Año")
    df_years = df_filtered.groupby('Year').size().reset_index(name='Cantidad de Artículos')
    fig_line = px.line(
        df_years, x='Year', y='Cantidad de Artículos', 
        title="Evolución Temporal de la Investigación",
        markers=True, template="plotly_white"
    )
    fig_line.update_traces(line_color="#0284c7", line_width=3)
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader("Autores Más Citados en la Temática")
    if 'Cited by' in df_filtered.columns and 'Authors' in df_filtered.columns:
        top_authors = df_filtered.sort_values(by='Cited by', ascending=False).head(8)
        fig_author = px.bar(
            top_authors, x='Cited by', y='Authors',
            orientation='h', title="Top 8 Autores con Mayor Impacto Académico",
            labels={'Cited by': 'Número de Citas', 'Authors': 'Autor'},
            color='Cited by', color_continuous_scale="Blues"
        )
        fig_author.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
        st.plotly_chart(fig_author, use_container_width=True)
    else:
        st.warning("La columna 'Authors' o 'Cited by' no está disponible.")

# =====================================================================
# 8. NUBE DE PALABRAS EN ABSTRACTS 
# =====================================================================
st.markdown("---")
st.subheader("☁️ Análisis Semántico de Palabras en Abstracts")
st.markdown("Este mapa visual identifica los términos más recurrentes en los resúmenes científicos, revelando las metodologías y enfoques dominantes:")

# Verificamos si existe la columna Abstract o usamos Author Keywords de respaldo
columna_texto = 'Abstract' if 'Abstract' in df.columns else 'Author Keywords'
text_abstracts = " ".join(df_filtered[columna_texto].astype(str))

# Lista de palabras vacías comunes en inglés para limpiar la nube
stop_words = ["the", "and", "a", "of", "to", "in", "is", "that", "for", "on", "with", "as", "by", "an", "it", "this", "from", "this", "No registra"]

if text_abstracts.strip() and len(text_abstracts) > 100:
    wordcloud = WordCloud(
        width=1100, height=350, 
        background_color='white', 
        colormap='Blues',
        stopwords=set(stop_words)
    ).generate(text_abstracts)
    
    fig_wc, ax = plt.subplots(figsize=(11, 3.5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wc)
else:
    st.info("Agrega la columna 'Abstract' en tu archivo CSV para procesar el análisis lingüístico avanzado requerido.")

# =====================================================================
# 9. SECCIÓN DE IMÁGENES DE COLAB
# =====================================================================
st.markdown("---")
st.subheader("🖼️ Análisis Exploratorio Avanzado (Resultados de Google Colab)")
st.markdown("Métricas experimentales obtenidas durante el entrenamiento y evaluación de los modelos:")

# Comentarios asignados perfectamente sin saltos de línea (Cero SyntaxError)
comentario_grafico_1 = "Este gráfico de barras logarítmico resume las métricas clave del dataset de Scopus, permitiendo una comparación rápida de valores muy diferentes."
comentario_grafico_2 = "Este gráfico de líneas muestra la evolución de los temas de investigación a lo largo del tiempo, revelando que 'BERT' es un tema central y dominante en las publicaciones, mostrando una presencia constante y en crecimiento, o un pico significativo que lo posiciona como un área de gran interés."

col_img1, col_img2 = st.columns(2)

with col_img1:
    st.image("grafico1.jpg", caption="Gráfico 1: Rendimiento General de Métricas", use_container_width=True)
    st.markdown(f'<div class="comentario-caja">{comentario_grafico_1}</div>', unsafe_allow_html=True)

with col_img2:
    st.image("grafico2.png", caption="Gráfico 2: Evolución Temática en el Tiempo", use_container_width=True)
    st.markdown(f'<div class="comentario-caja">{comentario_grafico_2}</div>', unsafe_allow_html=True)

# =====================================================================
# 10. VISUALIZACIÓN DE LA TABLA DE DATOS 
# =====================================================================
st.markdown("---")
with st.expander("📂 Desplegar Registro de Literatura Científica (Muestra de Scopus)"):
    st.dataframe(
        df_filtered[["Title", "Authors", "Year", "Source title", "Cited by", "Country"]], 
        use_container_width=True
    )

# =====================================================================
# 11. ASIGNACIÓN DE LICENCIA OBLIGATORIA
# =====================================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <p>Este aplicativo interactivo y su código fuente están protegidos bajo la siguiente licencia de distribución:</p>
        <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
            <img alt="Licencia Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
        </a>
        <br />
        <span><b>Licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0)</b></span>
    </div>
    """, unsafe_allow_html=True)
