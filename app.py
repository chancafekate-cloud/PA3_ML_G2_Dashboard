import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA (Punto 6: Diseño y Empatía)
# =====================================================================
st.set_page_config(
    page_title="Dashboard BERT & Facebook", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilo personalizado para una interfaz limpia y moderna
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stTitle { color: #0f172a; font-family: 'Arial'; font-weight: bold; }
    .comentario-caja { background-color: #f1f5f9; padding: 15px; border-left: 5px solid #0284c7; border-radius: 4px; margin-bottom: 20px; color: #334155; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# 2. IDENTIFICACIÓN Y PREGUNTA DE INVESTIGACIÓN (Punto 1)
# =====================================================================
st.title("📊 Detección de Fake News en Facebook mediante Modelos BERT")
st.markdown("---")

st.info(
    "**🔍 Pregunta de Investigación:** "
    "¿De qué manera los modelos BERT permiten detectar noticias falsas publicadas en Facebook?"
)

# =====================================================================
# 3. CARGA DE DATOS OPTIMIZADA (Punto 2: Calidad del Dataset)
# =====================================================================
@st.cache_data
def load_data():
    # Usamos el nombre exacto de tu archivo cargado
    df = pd.read_csv("PA3_ML_scopus_limpio.csv")
    # Limpieza visual obligatoria: Cambiamos los NaN restantes por "No registra"
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

# Aplicar filtro al conjunto de datos
df_filtered = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]

# Sección obligatoria de datos del alumno (Criterio de evaluación)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Datos del Alumno")
st.sidebar.write("**Estudiante:** Tu Nombre Completo")
st.sidebar.write("**Código:** Tu Código de Estudiante")

# =====================================================================
# 5. VISUALIZACIONES DINÁMICAS INTERACTIVAS (Punto 4)
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
    st.subheader("🌍 Top 10 Países Líderes en la Temática")
    top_countries = df_filtered['Country'].value_counts().head(10)
    fig_bar = px.bar(
        top_countries, x=top_countries.values, y=top_countries.index, 
        orientation='h', title="Publicaciones por País de Afiliación",
        labels={'x': 'Cantidad de Artículos', 'y': 'País'},
        color=top_countries.values, color_continuous_scale="Blues"
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

# Nube de Palabras Clave Automática
st.markdown("---")
st.subheader("☁️ Análisis de Tendencias Temáticas (Keywords)")
text_keywords = " ".join(df_filtered['Author Keywords'].astype(str))

if text_keywords.strip() and "No registra" not in text_keywords:
    wordcloud = WordCloud(width=1000, height=300, background_color='white', colormap='Blues').generate(text_keywords)
    fig_wc, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wc)

# =====================================================================
# 6. SECCIÓN DE IMÁGENES DE COLAB + COMENTARIOS (Soporte Visual Avanzado)
# =====================================================================
st.markdown("---")
st.subheader("🖼️ Análisis Exploratorio Avanzado (Resultados de Google Colab)")
st.markdown("A continuación, se presentan las métricas de rendimiento extraídas del entorno experimental:")

# ✍️ REEMPLAZA LOS TEXTOS ENTRE COMILLAS CON TUS COMENTARIOS REALES:
comentario_grafico_1 = "Este gráfico de barras logarítmico resume las métricas clave del dataset de Scopus, permitiendo una comparación rápida de valores muy diferentes.

Dominio de Citas: Las 'Total Citas' (636) y 'Total Autores' (134) son las métricas con valores más altos, lo que sugiere un impacto colectivo considerable y una amplia participación autoral en el dataset, a pesar de que el número total de artículos es de 37.
Producción Anual Clave: El año 2022 es claramente el más prolífico con 12 artículos publicados, destacando como un período de alta actividad en el dataset. Este pico es un punto importante para futuros análisis de tendencias específicas de ese año.
En resumen, el dataset refleja una comunidad de investigación activa con un impacto considerable en términos de citas, concentrando su mayor producción en el año 2022."
comentario_grafico_2 = " 
Este gráfico de líneas muestra la evolución de los temas de investigación a lo largo del tiempo, revelando:

Dominio de BERT: Se observa que 'BERT' es un tema central y dominante en las publicaciones, mostrando una presencia constante y en crecimiento, o un pico significativo que lo posiciona como un área de gran interés.

Tendencias Clave: Identifica el crecimiento, picos de interés o disminución en la producción de artículos por cada palabra clave. Podemos ver cómo otros temas se desarrollan en relación con el auge de 'BERT'.

Conexiones entre Temas: La cercanía o cruce de líneas puede sugerir áreas de investigación relacionadas o el uso de técnicas similares, especialmente en combinación con 'BERT' (e.g., 'Fake news detection', 'Machine learning').

Estabilidad o Declive: Se puede observar si campos de investigación mantienen una producción constante o si su interés está disminuyendo, permitiendo comparar la estabilidad de 'BERT' frente a otros temas.

Conclusión: Este análisis directo de las tendencias por tema es crucial para entender la dinámica del conocimiento en el dataset. La prevalencia y trayectoria de 'BERT' resaltan su impacto significativo en el campo de estudio, mientras que la evolución de otros temas muestra la diversidad y las interconexiones dentro de la investigación.
"

# Creamos dos columnas adicionales para poner tus fotos lado a lado
col_img1, col_img2 = st.columns(2)

with col_img1:
    # Llama a tu primera imagen (subida a GitHub como grafico1.jpg)
    st.image("grafico1.jpg", caption="Gráfico 1: Rendimiento del Modelo BERT", use_container_width=True)
    # Muestra tu comentario debajo en una caja limpia
    st.markdown(f'<div class="comentario-caja">{comentario_grafico_1}</div>', unsafe_allow_html=True)

with col_img2:
    # Llama a tu segunda imagen (subida a GitHub como grafico2.png)
    st.image("grafico2.png", caption="Gráfico 2: Matriz de Confusión / Distribución", use_container_width=True)
    # Muestra tu comentario debajo en una caja limpia
    st.markdown(f'<div class="comentario-caja">{comentario_grafico_2}</div>', unsafe_allow_html=True)

# =====================================================================
# 7. VISUALIZACIÓN DE LA TABLA DE DATOS (Facilidad de Uso)
# =====================================================================
st.markdown("---")
with st.expander("📂 Desplegar Registro de Literatura Científica"):
    st.dataframe(
        df_filtered[["Title", "Authors", "Year", "Source title", "Cited by", "Country"]], 
        use_container_width=True
    )

# =====================================================================
# 8. ASIGNACIÓN DE LICENCIA OBLIGATORIA (Punto 5)
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
