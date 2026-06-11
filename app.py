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
# 4. BARRA LATERAL: FILTROS, EMPATÍA CON EL USUARIO E IDENTIFICACIÓN
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

# Mejora de Empatía solicitada por el profesor: acceso inmediato al dataset modelo de Scopus
st.sidebar.markdown("---")
st.sidebar.markdown("### ⭳ Plantilla del Dataset")
st.sidebar.markdown("<small>Descarga el archivo CSV por defecto configurado para este análisis específico si deseas replicar la estructura de columnas requerida:</small>", unsafe_allow_html=True)

# Conversión segura del DataFrame cargado para habilitar la descarga directa
csv_data = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Descargar plantilla CSV",
    data=csv_data,
    file_name="plantilla_scopus_bert.csv",
    mime="text/csv"
)

# Identificación del Alumno con tus datos asignados
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Datos del Alumno")
st.sidebar.write("**Estudiante:** Grupo 2")
st.sidebar.markdown("""
* Chancafe Pisfil Liz
* Alzamora Graciela
* Marquez Marcelo
* Retamozo Sheila
""")
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
    pais_lider = "India"
    st.markdown(f'<div class="metric-box"><h3>{pais_lider}</h3><p style="color: #64748b;">País con Mayor Producción</p></div>', unsafe_allow_html=True)
with kpi3:
    citas_totales = int(df_filtered['Cited by'].sum()) if 'Cited by' in df_filtered.columns else 0
    st.markdown(f'<div class="metric-box"><h3>{citas_totales} Citas</h3><p style="color: #64748b;">Impacto Total en la Comunidad</p></div>', unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# 6. ANÁLISIS DE MODELOS DE INTELIGENCIA ARTIFICIAL Y ARQUITECTURAS COMPARATIVAS
# =====================================================================
st.subheader("🤖 Análisis de Modelos de Inteligencia Artificial Más Utilizados")
col_abs1, col_abs2 = st.columns(2)

with col_abs1:
    st.write("**Eficacia Comparativa de Algoritmos en la Detección de Fake News**")
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
    st.write("**Presencia y Menciones de Modelos de IA en el Dataset**")
    menciones_ia = {
        'Modelo / Variante': ['BERT', 'LSTM', 'CNN', 'SVM', 'RNN', 'Random Forest', 'RoBERTa', 'DistilBERT', 'XGBoost'],
        'Menciones': [37, 12, 10, 7, 5, 5, 4, 0, 0]
    }
    df_menciones = pd.DataFrame(menciones_ia)
    fig_menciones = px.bar(
        df_menciones, x='Modelo / Variante', y='Menciones',
        title="Frecuencia de Modelos Identificados en Títulos y Resúmenes",
        color='Menciones', color_continuous_scale="Blues"
    )
    fig_menciones.update_layout(template="plotly_white")
    st.plotly_chart(fig_menciones, use_container_width=True)

st.markdown("""
<div class="comentario-caja">
BERT emerge como el modelo más prominente con 37 menciones, subrayando su importancia central en el procesamiento de lenguaje natural para la detección de noticias falsas. Le siguen arquitecturas profundas como LSTM (12) y CNN (10), indicando que las redes complejas dominan de forma independiente o combinada frente a líneas base tradicionales como SVM o Random Forest.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# 7. ESTRUCTURA DE DOCUMENTOS Y ECOSISTEMA DE EDITORES
# =====================================================================
st.subheader("📄 Distribución de Formatos Académicos y Ecosistema Editorial")
col_doc1, col_doc2 = st.columns(2)

with col_doc1:
    st.write("**Distribución de Publicaciones por Tipo de Documento**")
    tipo_doc_data = {
        'Tipo de Documento': ['Conference paper', 'Article', 'Book chapter'],
        'Cantidad': [20, 15, 25]
    }
    df_tipos = pd.DataFrame(tipo_doc_data)
    fig_tipos = px.pie(
        df_tipos, values='Cantidad', names='Tipo de Documento',
        title="Preferencia de Difusión en Foros Peer-Reviewed",
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    st.plotly_chart(fig_tipos, use_container_width=True)

with col_doc2:
    st.write("**Top Editores por Número de Publicaciones**")
    editores_data = {
        'Editor': ['IEEE', 'Springer Deutschland', 'Springer', 'World Scientific', 'ACM', 'Inderscience', 'CEUR-WS', 'Incoma Ltd', 'Elsevier Ltd', 'IOS Press'],
        'Publicaciones': [14, 6, 4, 2, 2, 1, 1, 1, 1, 1]
    }
    df_editores = pd.DataFrame(editores_data)
    fig_editores = px.bar(
        df_editores, x='Publicaciones', y='Editor',
        orientation='h', title="Top Editores en la Literatura Analizada",
        color='Publicaciones', color_continuous_scale="Blues"
    )
    fig_editores.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
    st.plotly_chart(fig_editores, use_container_width=True)

st.markdown("""
<div class="comentario-caja">
La mayoría de publicaciones se concentran en Conference Papers (20) y Articles (15), reflejando un campo de rápida evolución científica. El ecosistema está fuertemente dominado por el Institute of Electrical and Electronics Engineers (IEEE) con 14 publicaciones y el grupo Springer, consolidando la legitimidad y alto estándar técnico del repositorio.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# 8. VARIABLES GEOGRÁFICAS, IDIOMAS Y CONTEXTOS TEMÁTICOS
# =====================================================================
st.subheader("🌐 Variables Demográficas, Idioma y Entornos de Desinformación")
col_geo1, col_geo2 = st.columns(2)

with col_geo1:
    st.write("**Ejes de Propagación de Desinformación y Modelos Multilingües**")
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

with col_geo2:
    st.write("**Análisis de Identificadores Adicionales**")
    identificadores_data = {
        'Métrica Evaluada': ['Artículos en Inglés', 'Direcciones Correspondencia Únicas', 'PubMed IDs Únicos'],
        'Valor Registrado': [37, 35, 0]
    }
    df_ids = pd.DataFrame(identificadores_data)
    fig_ids = px.bar(
        df_ids, x='Métrica Evaluada', y='Valor Registrado',
        title="Distribución Lingüística e Identificadores Biométricos",
        color='Valor Registrado', color_continuous_scale="Blues"
    )
    fig_ids.update_layout(template="plotly_white")
    st.plotly_chart(fig_ids, use_container_width=True)

st.markdown("""
<div class="comentario-caja">
La desinformación ligada a la pandemia de COVID-19 e infodemia lidera los entornos temáticos de evaluación con 14 artículos científicos. El inglés se ratifica como la lingua franca absoluta con 37 publicaciones, mientras que el hallazgo de 0 PubMed IDs confirma que los alcances de la investigación están orientados a las ciencias de la computación e ingeniería, ajenos al sector clínico directo.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# 9. VISUALIZACIONES DINÁMICAS ORIGINALES DEL REPOSITORIO
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
# 10. NUBE DE PALABRAS EN ABSTRACTS
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
# 11. SECCIÓN DE IMÁGENES DE COLAB 
# =====================================================================
st.markdown("---")
st.subheader("🖼️ Análisis Exploratorio Avanzado (Resultados de Google Colab)")
st.markdown("Métricas experimentales obtenidas durante el entrenamiento y evaluación de los modelos:")

# Asignación exacta de comentarios sin rupturas de sintaxis
comentario_grafico_1 = "Este gráfico de barras logarítmico resume las métricas clave del dataset de Scopus, permitiendo una comparación rápida de valores muy diferentes."
comentario_grafico_2 = "Este gráfico de líneas muestra la evolución de los temas de investigación a lo largo del tiempo, revelando que 'BERT' es un tema central and dominante en las publicaciones, mostrando una presencia constante y en crecimiento, o un pico significativo que lo posiciona como un área de gran interés."
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
# 12. VISUALIZACIÓN DE LA TABLA DE DATOS
# =====================================================================
st.markdown("---")
with st.expander("📂 Desplegar Registro de Literatura Científica (Muestra de Scopus)"):
    st.dataframe(
        df_filtered[["Title", "Authors", "Year", "Source title", "Cited by", "Country"]] if 'Country' in df_filtered.columns else df_filtered[["Title", "Authors", "Year", "Source title", "Cited by"]], 
        use_container_width=True
    )

# =====================================================================
# 13. ASIGNACIÓN DE LICENCIA DE SOFTWARE (APACHE LICENSE 2.0)
# =====================================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; margin-top: 20px; padding: 10px; color: #475569;">
        <p>Este aplicativo interactivo y su código fuente asociado están distribuidos bajo los términos de la legislación de código abierto:</p>
        <span style="font-size: 16px; font-weight: bold; color: #0f172a;"><b>Apache License, Version 2.0 (Apache-2.0)</b></span>
        <br />
        <small style="display: block; margin-top: 5px; color: #64748b;">Permite la libre utilización, modificación y distribución comercial del software, asegurando la flexibilidad necesaria para su adaptación empresarial o académica (tesis) bajo el resguardo de la atribución de autoría original.</small>
    </div>
    """, unsafe_allow_html=True)





==========================================================
=======================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =====================================================================
st.set_page_config(
    page_title="Dashboard Académico - BERT y Facebook",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para asegurar una interfaz ordenada y profesional
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

# Palabras clave simplificadas y sin redundancias
st.markdown("### 🏷️ Palabras Clave del Estudio")
st.markdown("""
    <span class="keyword-badge">BERT</span>
    <span class="keyword-badge">Noticias falsas</span>
    <span class="keyword-badge">Facebook</span>
    <span class="keyword-badge">Procesamiento de Lenguaje Natural</span>
""", unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# 3. CARGA DE DATOS OPTIMIZADA CON ROBUSTEZ CONTRA ERRORES
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("PA3_ML_scopus_limpio.csv")
    df = df.fillna("No registra")

    # Limpieza preventiva de espacios en blanco
    df.columns = df.columns.str.strip()

    # Detección inteligente de columna País
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
# 4. BARRA LATERAL
# =====================================================================
st.sidebar.header("🔍 Filtros Dinámicos")

year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
selected_years = st.sidebar.slider(
    "Filtrar rango de años de publicación:",
    year_min, year_max, (year_min, year_max)
)

df_filtered = df[
    (df['Year'] >= selected_years[0]) &
    (df['Year'] <= selected_years[1])
]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⭳ Plantilla del Dataset")
st.sidebar.markdown(
    "<small>Descarga el archivo CSV configurado para replicar la estructura de columnas requerida para este análisis.</small>",
    unsafe_allow_html=True
)

csv_data = df.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="Descargar plantilla CSV",
    data=csv_data,
    file_name="plantilla_scopus_bert.csv",
    mime="text/csv"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Datos del Alumno")
st.sidebar.write("**Estudiante:** Grupo 2")
st.sidebar.markdown("""
* Chancafe Pisfil Liz
* Alzamora Graciela
* Marquez Marcelo
* Retamozo Sheila
""")
st.sidebar.write("**Código:** 6817")

# =====================================================================
# 5. KPIs
# =====================================================================
st.subheader("🚀 Indicadores Clave del Dataset")

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    total_articulos = len(df_filtered)
    st.markdown(
        f'<div class="metric-box"><h3>{total_articulos} Artículos</h3><p style="color: #64748b;">Muestra de Literatura Científica</p></div>',
        unsafe_allow_html=True
    )

with kpi2:
    pais_lider = "India"
    st.markdown(
        f'<div class="metric-box"><h3>{pais_lider}</h3><p style="color: #64748b;">País con Mayor Producción</p></div>',
        unsafe_allow_html=True
    )

with kpi3:
    citas_totales = int(df_filtered['Cited by'].sum()) if 'Cited by' in df_filtered.columns else 0
    st.markdown(
        f'<div class="metric-box"><h3>{citas_totales} Citas</h3><p style="color: #64748b;">Impacto Total en la Comunidad</p></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# =====================================================================
# 6. ANÁLISIS DE MODELOS DE IA
# =====================================================================
st.subheader("🤖 Análisis de Modelos de Inteligencia Artificial Más Utilizados")

col_abs1, col_abs2 = st.columns(2)

with col_abs1:
    st.write("**Eficacia Comparativa de Algoritmos en la Detección de Noticias Falsas**")

    modelos_data = {
        'Algoritmo / Arquitectura': [
            'SVM + TF-IDF (Máximo)',
            'BERT (COVID-19 Dataset)',
            'BERT + Bi-GRU (FNID)',
            'Passive-Aggressive',
            'Linear SVM',
            'BERT + Bi-GRU (FNFD)',
            'CNN / LSTM Baseline'
        ],
        'Precisión Máxima (%)': [99.6, 98.41, 97.0, 94.0, 92.0, 91.0, 90.0]
    }

    df_models = pd.DataFrame(modelos_data)

    fig_models = px.bar(
        df_models,
        x='Precisión Máxima (%)',
        y='Algoritmo / Arquitectura',
        orientation='h',
        title="Porcentaje de Accuracy / F1-Score por Arquitectura",
        color='Precisión Máxima (%)',
        color_continuous_scale="Blues"
    )

    fig_models.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white"
    )

    st.plotly_chart(fig_models, use_container_width=True)

with col_abs2:
    st.write("**Presencia y Menciones de Modelos de IA en el Dataset**")

    menciones_ia = {
        'Modelo / Variante': [
            'BERT',
            'LSTM',
            'CNN',
            'SVM',
            'RNN',
            'Random Forest',
            'RoBERTa',
            'DistilBERT',
            'XGBoost'
        ],
        'Menciones': [37, 12, 10, 7, 5, 5, 4, 0, 0]
    }

    df_menciones = pd.DataFrame(menciones_ia)

    fig_menciones = px.bar(
        df_menciones,
        x='Modelo / Variante',
        y='Menciones',
        title="Frecuencia de Modelos Identificados en Títulos y Resúmenes",
        color='Menciones',
        color_continuous_scale="Blues"
    )

    fig_menciones.update_layout(template="plotly_white")

    st.plotly_chart(fig_menciones, use_container_width=True)

st.markdown("""
<div class="comentario-caja">
BERT emerge como el modelo más prominente con 37 menciones, subrayando su importancia central en el procesamiento de lenguaje natural para la detección de noticias falsas. Le siguen arquitecturas profundas como LSTM (12) y CNN (10), indicando que las redes complejas dominan de forma independiente o combinada frente a líneas base tradicionales como SVM o Random Forest.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# 11. SECCIÓN DE IMÁGENES DE COLAB
# =====================================================================
st.subheader("🖼️ Análisis Exploratorio Avanzado (Resultados de Google Colab)")
st.markdown("Métricas experimentales obtenidas durante el entrenamiento y evaluación de los modelos.")

comentario_grafico_1 = "Este gráfico de barras logarítmico resume las métricas clave del dataset de Scopus, permitiendo una comparación rápida de valores muy diferentes."

comentario_grafico_2 = "Este gráfico de líneas muestra la evolución de los temas de investigación a lo largo del tiempo, revelando que BERT es un tema central y dominante en las publicaciones."

comentario_grafico_3 = "Este gráfico de barras visualiza las 10 revistas y conferencias que han publicado artículos relacionados con la detección de noticias falsas utilizando BERT."

comentario_grafico_4 = "Basándonos en la tabla de frecuencia y el gráfico de barras, podemos identificar los países con mayor representación en el ámbito de la detección de noticias falsas."

# ---------------------------------------------------------------------
# GRÁFICO 1
# ---------------------------------------------------------------------
st.markdown("### 📊 Gráfico 1: Rendimiento General de Métricas")

st.image(
    "grafico1.jpg",
    caption="Rendimiento General de Métricas",
    use_container_width=True
)

st.markdown(
    f'<div class="comentario-caja">{comentario_grafico_1}</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------------------------
# GRÁFICO 2
# ---------------------------------------------------------------------
st.markdown("### 📈 Gráfico 2: Evolución Temática en el Tiempo")

st.image(
    "grafico2.png",
    caption="Evolución Temática en el Tiempo",
    use_container_width=True
)

st.markdown(
    f'<div class="comentario-caja">{comentario_grafico_2}</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------------------------
# GRÁFICO 3
# ---------------------------------------------------------------------
st.markdown("### 📰 Gráfico 3: Revistas y Conferencias Destacadas")

st.image(
    "grafico3.png",
    caption="Revistas y Conferencias Destacadas",
    use_container_width=True
)

st.markdown(
    f'<div class="comentario-caja">{comentario_grafico_3}</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------------------------
# GRÁFICO 4
# ---------------------------------------------------------------------
st.markdown("### 🌍 Gráfico 4: Representación Geográfica Global")

st.image(
    "grafico4.png",
    caption="Representación Geográfica Global",
    use_container_width=True
)

st.markdown(
    f'<div class="comentario-caja">{comentario_grafico_4}</div>',
    unsafe_allow_html=True
)

# =====================================================================
# 12. VISUALIZACIÓN DE TABLA DE DATOS
# =====================================================================
st.markdown("---")

with st.expander("📂 Desplegar Registro de Literatura Científica (Muestra de Scopus)"):

    columnas_base = [
        "Title",
        "Authors",
        "Year",
        "Source title",
        "Cited by"
    ]

    if 'Country' in df_filtered.columns:
        columnas_base.append("Country")

    # Agregar abstract de forma segura
    if 'Abstract' in df_filtered.columns:
        columnas_base.append("Abstract")

    st.dataframe(
        df_filtered[columnas_base],
        use_container_width=True
    )

# =====================================================================
# 13. LICENCIA
# =====================================================================
st.markdown("---")

st.markdown("""
    <div style="text-align: center; margin-top: 20px; padding: 10px; color: #475569;">
        <p>Este aplicativo interactivo y su código fuente asociado están distribuidos bajo los términos de la legislación de código abierto:</p>

        <span style="font-size: 16px; font-weight: bold; color: #0f172a;">
            <b>Apache License, Version 2.0 (Apache-2.0)</b>
        </span>

        <br />

        <small style="display: block; margin-top: 5px; color: #64748b;">
            Permite la libre utilización, modificación y distribución comercial del software,
            asegurando la flexibilidad necesaria para su adaptación empresarial o académica
            bajo el resguardo de la atribución de autoría original.
        </small>
    </div>
""", unsafe_allow_html=True)





