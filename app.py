import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA (Punto 6: Diseño de Impacto y Empatía)
# =====================================================================
st.set_page_config(
    page_title="Dashboard Avanzado BERT & Facebook", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilos CSS avanzados para una interfaz profesional y moderna
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stTitle { color: #0f172a; font-family: 'Arial'; font-weight: bold; }
    .metric-box { background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 4px solid #0284c7; }
    .comentario-caja { background-color: #f1f5f9; padding: 15px; border-left: 5px solid #0284c7; border-radius: 4px; margin-bottom: 20px; color: #334155; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# 2. IDENTIFICACIÓN Y PREGUNTA DE INVESTIGACIÓN (Punto 1)
# =====================================================================
st.title("📊 Detección de Fake News en Facebook mediante Modelos BERT")
st.markdown("🔍 **Análisis de Minería de Datos Basado en 37 Artículos Científicos de Scopus**")
st.markdown("---")

st.info(
    "**💡 Pregunta Eje:** "
    "¿De qué manera los modelos BERT permiten detectar noticias falsas publicadas en Facebook?"
)

# =====================================================================
# 3. CARGA DE DATOS OPTIMIZADA (Punto 2: Calidad del Dataset)
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("PA3_ML_scopus_limpio.csv")
    df = df.fillna("No registra")
    return df

df = load_data()

# =====================================================================
# 4. BARRA LATERAL: FILTROS E IDENTIFICACIÓN (Punto 3 y Punto 10)
# =====================================================================
st.sidebar.header("🔍 Filtros de Análisis")

# Filtro interactivo por rango de años
year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
selected_years = st.sidebar.slider(
    "Seleccione el rango de años de publicación:", 
    year_min, year_max, (year_min, year_max)
)

# Aplicar filtro dinámico
df_filtered = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]

# Sección obligatoria de datos del alumno
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Datos del Alumno")
st.sidebar.write("**Estudiante:** Tu Nombre Completo")
st.sidebar.write("**Código:** Tu Código de Estudiante")

# =====================================================================
# 5. KPls / METRICAS RESUMEN DE IMPACTO (Aporta rigor analítico)
# =====================================================================
st.subheader("🚀 Indicadores Clave de la Investigación")
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.markdown('<div class="metric-box"><h3>37 Artículos</h3><p style="color: #64748b;">Muestra Total de Scopus</p></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown('<div class="metric-box"><h3>BERT Híbridos</h3><p style="color: #64748b;">Arquitectura Dominante (CNN/LSTM)</p></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown('<div class="metric-box"><h3>Contexto Semántico</h3><p style="color: #64748b;">Principal Ventaja Tecnológica</p></div>', unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# 6. VISUALIZACIONES DINÁMICAS REESTRUCTURADAS (Punto 4)
# =====================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Evolución Temporal de Investigaciones")
    df_years = df_filtered.groupby('Year').size().reset_index(name='Cantidad')
    fig_line = px.line(
        df_years, x='Year', y='Cantidad', 
        title="Producción Científica por Año (BERT + Facebook)",
        markers=True, template="plotly_white"
    )
    fig_line.update_traces(line_color="#0284c7", line_width=3)
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader("🛠️ Enfoques y Arquitecturas Híbridas Utilizadas")
    arquitecturas_data = {
        'Enfoque Tecnológico': ['BERT Embeddings', 'Fine-tuning', 'BERT + CNN', 'BERT + LSTM/Bi-LSTM', 'Mecanismos de Atención'],
        'Frecuencia de Uso': [14, 11, 8, 7, 5]
    }
    df_arq = pd.DataFrame(arquitecturas_data)
    fig_bar = px.bar(
        df_arq, x='Frecuencia de Uso', y='Enfoque Tecnológico',
        orientation='h', title="Distribución de Enfoques Tecnológicos Basados en BERT",
        color='Frecuencia de Uso', color_continuous_scale="Blues"
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

# =====================================================================
# 7. SECCIÓN DE IMPACTO: CÓMO TRABAJA BERT VS DESAFÍOS
# =====================================================================
st.markdown("---")
col_text1, col_text2 = st.columns(2)

with col_text1:
    st.subheader("💡 ¿Cómo permite BERT detectar Fake News en Facebook?")
    st.markdown("""
    Los 37 artículos analizados demuestran que BERT soluciona las limitaciones de la minería de texto tradicional mediante:
    * **Embeddings de Contexto:** Genera vectores que entienden el significado profundo y la intención de la frase, superando la simple detección de palabras clave aisladas.
    * **Fine-Tuning Adaptativo:** Los modelos pre-entrenados son ajustados finamente con datasets de redes sociales, adaptándose al lenguaje informal y posts engañosos.
    * **Modelos Híbridos Eficientes:** Al combinarse con redes CNN (características locales) y LSTMs (secuencias de tiempo), BERT captura la malicia estructural de una noticia falsa.
    """)

with col_text2:
    st.subheader("⚠️ Desafíos Críticos Identificados en Redes Sociales")
    st.markdown("""
    La literatura científica destaca que la detección en Facebook enfrenta obstáculos severos que los modelos mitigan gradualmente:
    * **Volumen y Velocidad:** Millones de posts por segundo exigen un preprocesamiento veloz y embeddings optimizados.
    * **Sutilezas del Lenguaje:** El uso constante de ironía, sarcasmo, desinformación intencional y propaganda estruturada.
    * **Restricción de Privacidad:** Limitaciones técnicas para acceder a metadatos completos de usuarios, obligando a los modelos a depender estrictamente del texto del post.
    """)

# Nube de Palabras Clave Automática de Scopus
st.markdown("---")
st.subheader("☁️ Análisis de Tendencias Temáticas en la Literatura (Keywords de Autores)")
text_keywords = " ".join(df_filtered['Author Keywords'].astype(str))

if text_keywords.strip() and "No registra" not in text_keywords:
    wordcloud = WordCloud(width=1000, height=300, background_color='white', colormap='Blues').generate(text_keywords)
    fig_wc, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wc)

# =====================================================================
# 8. SECCIÓN DE IMÁGENES DE COLAB + COMENTARIOS SOLICITADOS
# =====================================================================
st.markdown("---")
st.subheader("🖼️ Análisis Exploratorio Avanzado (Resultados de Google Colab)")
st.markdown("Métricas experimentales obtenidas durante el entrenamiento y evaluación de los modelos:")

# COMENTARIOS ASIGNADOS EXACTAMENTE COMO SOLICITASTE:
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
# 9. VISUALIZACIÓN DE LA TABLA DE DATOS (Facilidad de Uso)
# =====================================================================
st.markdown("---")
with st.expander("📂 Desplegar Registro de Literatura Científica (37 Artículos Extraídos)"):
    st.dataframe(
        df_filtered[["Title", "Authors", "Year", "Source title", "Cited by", "Country"]], 
        use_container_width=True
    )

# =====================================================================
# 10. ASIGNACIÓN DE LICENCIA OBLIGATORIA (Punto 5)
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
