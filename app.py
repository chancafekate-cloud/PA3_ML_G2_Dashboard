import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =====================================================================
st.set_page_config(
    page_title="Dashboard Académico - BERT & Facebook", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilos CSS para asegurar una interfaz estéticamente ordenada y profesional
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
# 3. CARGA DE DATOS OPTIMIZADA CON ROBUSTEZ CONTRA ERRORES
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("PA3_ML_scopus_limpio.csv")
    df = df.fillna("No registra")
    
    # Limpieza preventiva de espacios en blanco en nombres de columnas para evitar fallos de llaves
    df.columns = df.columns.str.strip()
    
    # Búsqueda inteligente e igualación por variaciones del campo País
    col_pais_detectada = None
    for col in df.columns:
        if col.lower() in ['country', 'país', 'pais']:
            col_pais_detectada = col
            break
            
    if col_pais_detectada:
        df = df.rename(columns={col_pais_detectada: 'Country'})
        df['Country'] = df['Country'].astype(str).str.strip()
        
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

# Identificación del Alumno con tus datos asignados
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
    if 'Country' in df_filtered.columns:
        df_valid_countries = df_filtered[df_filtered['Country'] != "No registra"]
        pais_lider = df_valid_countries['Country'].value_counts().idxmax() if not df_valid_countries.empty else "No registra"
    else:
        pais_lider = "No registra"
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
# 7. SECCIÓN COMPLEMENTARIA: RESUMEN CONSOLIDADO DEL ANÁLISIS DE SCOPUS
# =====================================================================
st.subheader("### Resumen Consolidado del Análisis del Dataset de Scopus para Dashboards")
st.markdown("""
Este documento presenta un resumen de los análisis realizados sobre tu dataset de Scopus, centrándose en la identificación de modelos de IA, la distribución de tipos de documento, editores, idiomas y otros identificadores clave. Esta información es fundamental para construir un dashboard integral que visualice las tendencias y características de la investigación en tu área de interés.
""")

st.markdown("#### 1. Análisis de Modelos de Inteligencia Artificial Más Utilizados")
st.markdown("""
El análisis de títulos, resúmenes y palabras clave de autores para la mención de modelos de IA revela lo siguiente:
*   **BERT** emerge como el modelo más prominente, siendo mencionado en **37** artículos, lo que subraya su importancia central en la investigación contenida en tu dataset, especialmente en el contexto de la detección de noticias falsas.
*   **LSTM** le sigue con **12** menciones, y **CNN** con **10** menciones, indicando que las arquitecturas de redes neuronales profundas siguen siendo fundamentales, a menudo en combinación con BERT o como enfoques independientes.
*   Otros modelos como **SVM (7)**, **RNN (5)** y **Random Forest (5)** también están presentes, aunque con menor frecuencia, lo que podría sugerir su uso en contextos más específicos o como líneas base de comparación.
*   Variantes como **RoBERTa (4)** también aparecen, mientras que **DistilBERT (0)** y **XGBoost (0)** tienen una presencia muy limitada o nula en este dataset.

**Conclusión:** La preeminencia de BERT destaca su influencia y la tendencia actual en el procesamiento del lenguaje natural para la detección de noticias falsas, a menudo en sinergia con arquitecturas de aprendizaje profundo clásicas.
""")

st.markdown("#### 2. Distribución de Publicaciones por Tipo de Documento")
st.markdown("""
La mayoría de las publicaciones en tu dataset se concentran en formatos académicos clave, mostrando una clara preferencia por la difusión en foros peer-reviewed:
*   **Conference paper:** Con **20** publicaciones, es el tipo de documento más frecuente, lo cual es indicativo de campos de rápida evolución donde las conferencias son vitales para la difusión temprana de resultados.
*   **Article:** Con **15** publicaciones, los artículos de revista representan el segundo formato más común, asegurando una revisión más exhaustiva y una consolidación de la investigación.
*   **Book chapter:** Con **2** publicaciones, los capítulos de libro tienen una representación menor, lo que puede indicar una síntesis o compilación de conocimientos en áreas específicas.

**Conclusión:** El dataset está fuertemente inclinado hacia las publicaciones de conferencias y artículos de revistas, reflejando las vías principales de difusión de la investigación en este dominio. La escasa presencia de otros tipos de documentos resalta el enfoque en la literatura primaria de investigación.
""")

st.markdown("#### 3. Top 10 Editores por Número de Publicaciones")
st.markdown("""
Los editores más prolíficos en tu dataset, aquellos que lideran la publicación de investigaciones en la detección de noticias falsas y modelos de IA, son:
*   **Institute of Electrical and Electronics Engineers (IEEE):** Con **14** publicaciones, es el editor más dominante, un actor fundamental en el campo de la ingeniería y la computación.
*   **Springer Science and Business Media Deutschland GmbH:** Con **6** publicaciones, destaca como un editor importante en la literatura científica y técnica.
*   **Springer:** Con **4** publicaciones, esta editorial complementa la presencia de su grupo en la difusión de conocimiento.
*   **World Scientific Publishing Co.:** Con **2** publicaciones, contribuye a la difusión de la investigación especializada.
*   **ACM (Association for Computing Machinery):** Con **2** publicaciones, es una entidad clave en el ámbito de la informática y las ciencias de la computación.
*   **Inderscience Publishers:** Con **1** publicación, aporta a la diversidad de las plataformas de difusión.
*   **CEUR-WS:** Con **1** publicación, es relevante para la difusión de trabajos de conferencias.
*   **Incoma Ltd:** Con **1** publicación, forma parte del ecosistema de publicaciones.
*   **Elsevier Ltd:** Con **1** publicación, aunque con menor presencia en este top 10 específico, es una de las mayores editoriales científicas.
*   **IOS Press:** Con **1** publicación, también contribuye a la literatura en el área.

**Conclusión:** La lista de los Top 10 Editores muestra una concentración de publicaciones en instituciones académicas y editoriales de prestigio como IEEE y Springer, lo que subraya la calidad y el alcance de la investigación en el dataset. Su influencia es crucial para la diseminación de avances en IA y detección de noticias falsas.
""")

st.markdown("#### 4. Distribución de Publicaciones por Idioma")
st.markdown("""
El idioma predominante en las publicaciones es:
*   **English:** La vasta mayoría de los artículos están publicados en inglés (**37** publicaciones), lo cual es un estándar global en la investigación científica y tecnológica.

**Conclusión:** El inglés es la lingua franca de la investigación en tu dataset, lo cual es un patrón consistente con la difusión del conocimiento científico a nivel global, facilitando el acceso a una audiencia internacional.
""")

st.markdown("#### 5. Análisis de 'PubMed ID'")
st.markdown("""
La columna 'PubMed ID' identifica artículos indexados en la base de datos biomédica PubMed. En este dataset, se encontraron **0** PubMed IDs únicos. Esto sugiere que los artículos en tu colección no están principalmente relacionados con el ámbito biomédico o de la salud, o que no están indexados en esta base de datos específica.
""")

st.markdown("#### 6. Análisis de 'Correspondence Address'")
st.markdown("""
La columna 'Correspondence Address' proporciona la dirección de contacto principal para los autores. Se encontraron **35** Direcciones de Correspondencia únicas en el dataset. Estas direcciones pueden indicar la afiliación institucional primaria o la ubicación geográfica de los autores principales, ofreciendo un potencial para futuros análisis de colaboración geográfica, aunque no se ha realizado un análisis detallado de estas direcciones en este resumen.

---
Esta información es esencial para crear un dashboard que visualice las tendencias, los actores clave y las características demográficas de la investigación en tu campo, permitiendo una comprensión más profunda y contextualizada de tu dataset.
""")

st.markdown("---")

# =====================================================================
# 8. VISUALIZACIONES DINÁMICAS ORIGINALES
# =====================================================================
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
# 9. NUBE DE PALABRAS EN ABSTRACTS
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
# 10. SECCIÓN DE IMÁGENES DE COLAB + COMENTARIOS SOLICITADOS (CUADRÍCULA COMPLETA)
# =====================================================================
st.markdown("---")
st.subheader("🖼️ Análisis Exploratorio Avanzado (Resultados de Google Colab)")
st.markdown("Métricas experimentales obtenidas durante el entrenamiento y evaluación de los modelos:")

# Asignación exacta de comentarios sin rupturas de sintaxis
comentario_grafico_1 = "Este gráfico de barras logarítmico resume las métricas clave del dataset de Scopus, permitiendo una comparación rápida de valores muy diferentes."
comentario_grafico_2 = "Este gráfico de líneas muestra la evolución de los temas de investigación a lo largo del tiempo, revelando que 'BERT' es un tema central y dominante en las publicaciones, mostrando una presencia constante y en crecimiento, o un pico significativo que lo posiciona como un área de gran interés."
comentario_grafico_3 = "Este gráfico de barras visualiza las 10 revistas y conferencias que han publicado artículos relacionados con la detección de noticias falsas utilizando BERT"
comentario_grafico_4 = "Basándonos en la tabla de frecuencia y el gráfico de barras, podemos identificar los países que tienen una mayor representación en tu en el ámbito de la detección de noticias falsas."

# Fila 1: Gráficos 1 y 2
col_img1, col_img2 = st.columns(2)
with col_img1:
    st.image("grafico1.jpg", caption="Gráfico 1: Rendimiento General de Métricas", use_container_width=True)
    st.markdown(f'<div class="comentario-caja">{comentario_grafico_1}</div>', unsafe_allow_html=True)
with col_img2:
    st.image("grafico2.png", caption="Gráfico 2: Evolución Temática en el Tiempo", use_container_width=True)
    st.markdown(f'<div class="comentario-caja">{comentario_grafico_2}</div>', unsafe_allow_html=True)

# Fila 2: Gráficos 3 y 4
col_img3, col_img4 = st.columns(2)
with col_img3:
    st.image("grafico3.png", caption="Gráfico 3: Revistas y Conferencias Destacadas", use_container_width=True)
    st.markdown(f'<div class="comentario-caja">{comentario_grafico_3}</div>', unsafe_allow_html=True)
with col_img4:
    st.image("grafico4.png", caption="Gráfico 4: Representación Geográfica Global", use_container_width=True)
    st.markdown(f'<div class="comentario-caja">{comentario_grafico_4}</div>', unsafe_allow_html=True)

# =====================================================================
# 11. VISUALIZACIÓN DE LA TABLA DE DATOS
# =====================================================================
st.markdown("---")
with st.expander("📂 Desplegar Registro de Literatura Científica (Muestra de Scopus)"):
    st.dataframe(
        df_filtered[["Title", "Authors", "Year", "Source title", "Cited by", "Country"]] if 'Country' in df_filtered.columns else df_filtered[["Title", "Authors", "Year", "Source title", "Cited by"]], 
        use_container_width=True
    )

# =====================================================================
# 12. ASIGNACIÓN DE LICENCIA OBLIGATORIA
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
