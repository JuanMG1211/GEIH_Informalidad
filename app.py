import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="Informalidad Laboral · Colombia 2024",
    page_icon="🇨🇴",
    layout="wide",
)

# ── Rutas ───────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent
MODEL_PATH    = BASE / "outputs" / "champion_geih.pkl"
META_PATH     = BASE / "outputs" / "champion_geih_meta.json"
DATA_PATH     = BASE / "outputs" / "datos_procesados.parquet"
PREPROC_PATH  = BASE / "parquet"  / "preprocessor.joblib"
METRICS_PATH  = BASE / "outputs" / "metrics_comparison.csv"
IMG_SHAP_BAR  = BASE / "outputs" / "shap_importance.png"
IMG_SHAP_BEE  = BASE / "outputs" / "shap_summary.png"
IMG_CONFUSION = BASE / "outputs" / "confusion_matrix.png"
IMG_COMP      = BASE / "outputs" / "model_comparativa.png"

# ── Catálogos ────────────────────────────────────────────────────────────
DPTO_INFO = {
    5:  ("Antioquia",          6.70, -75.50),
    8:  ("Atlántico",         10.90, -74.80),
    11: ("Bogotá D.C.",        4.60, -74.08),
    13: ("Bolívar",            9.90, -75.01),
    15: ("Boyacá",             5.54, -73.37),
    17: ("Caldas",             5.07, -75.51),
    18: ("Caquetá",            1.61, -75.60),
    19: ("Cauca",              2.44, -76.62),
    20: ("Cesar",             10.48, -73.25),
    23: ("Córdoba",            8.86, -75.88),
    25: ("Cundinamarca",       5.00, -74.00),
    27: ("Chocó",              5.69, -76.65),
    41: ("Huila",              2.92, -75.28),
    44: ("La Guajira",        11.54, -72.91),
    47: ("Magdalena",         10.47, -74.23),
    50: ("Meta",               4.15, -73.63),
    52: ("Nariño",             1.21, -77.28),
    54: ("N. de Santander",    7.89, -72.50),
    63: ("Quindío",            4.53, -75.68),
    66: ("Risaralda",          4.81, -75.70),
    68: ("Santander",          7.13, -73.12),
    70: ("Sucre",              9.30, -75.40),
    73: ("Tolima",             4.44, -75.24),
    76: ("Valle del Cauca",    3.43, -76.54),
    81: ("Arauca",             7.09, -70.76),
    85: ("Casanare",           5.31, -72.39),
    86: ("Putumayo",           1.00, -76.59),
    88: ("San Andrés",        12.52, -81.72),
    91: ("Amazonas",          -1.05, -71.94),
    94: ("Guainía",            2.58, -68.52),
    95: ("Guaviare",           2.57, -72.65),
    97: ("Vaupés",             1.05, -70.23),
    99: ("Vichada",            4.22, -67.93),
}

POSICION = {
    1: "Empleado particular", 2: "Empleado del gobierno",
    3: "Empleado doméstico",  4: "Cuenta propia",
    5: "Empleador / Patrón",  6: "Familiar sin remuneración",
    7: "Jornalero / Peón",    8: "Otro",
}

ESTADO_CIVIL = {
    1: "No unido/a", 2: "Unión libre", 3: "Casado/a",
    4: "Separado/a", 5: "Viudo/a",    6: "NS/NR",
}

CONTRATO_TIPO = {1: "Verbal", 2: "Escrito", 9: "No sabe / No aplica"}

TAMANO_EMP = {
    1: "1 persona",      2: "2–5 personas",   3: "6–10 personas",
    4: "11–19 personas", 5: "20–30 personas",  6: "31–50 personas",
    7: "51–100",         8: "101–200",         9: "201 o más",
    10: "No sabe",
}

RAMA_CIIU = {
    1:  "Agricultura/Pesca",       5:  "Minería",
    10: "Industria manufacturera", 36: "Agua/Servicios",
    41: "Construcción",            45: "Comercio/Vehículos",
    49: "Transporte",              55: "Alojamiento/Restaurantes",
    58: "Comunicaciones",          64: "Financiero",
    68: "Inmobiliario",            75: "Veterinaria",
    78: "Servicios adm.",          84: "Administración pública",
    85: "Educación",               86: "Salud",
    90: "Artes/Entretenimiento",   97: "Hogares con servicio dom.",
    99: "Otro/NS",
}

LABELS_EDU = {
    1: "Ninguno",       2: "Preescolar",     3: "Primaria inc.",
    4: "Primaria",      5: "Sec. inc.",       6: "Secundaria",
    7: "Media inc.",    8: "Media",           9: "Técnica/Tecn.",
    10: "Universitaria",11: "Especialización",12: "Maestría",
    13: "Doctorado",
}

# ── Carga de recursos ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROC_PATH)

@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_parquet(DATA_PATH)
    return None

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None

# ── Helper: tasa ponderada ───────────────────────────────────────────────
def tasa_pond_grp(g):
    return (g["INFORMAL"] * g["FEX_C18"]).sum() / g["FEX_C18"].sum() * 100

# ── Helper: construir input para el preprocessor ─────────────────────────
def construir_input(p6040, p3271, p6070, clase, dpto,
                    p6430, p6450, p6800, p3069, rama,
                    anios_edu, pluriempleo):
    mapa_edu = {1:0, 2:1, 3:3, 4:5, 5:7, 6:9, 7:10,
                8:11, 9:14, 10:16, 11:17, 12:18, 13:21}
    bins   = [14, 24, 34, 44, 54, 64, 120]
    labels = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    edad_grupo = pd.cut([p6040], bins=bins, labels=labels)[0]
    row = {
        "P3271":           p3271,
        "P6040":           p6040,
        "EDAD_GRUPO":      str(edad_grupo),
        "ANIOS_EDU":       anios_edu,
        "P6070":           p6070,
        "CLASE":           clase,
        "DPTO":            dpto,
        "P6430":           p6430,
        "P6450":           p6450,
        "P6800":           p6800,
        "P3069":           p3069,
        "RAMA2D_R4":       rama,
        "CUENTA_PROPIA":   int(p6430 == 4),
        "MICROEMPRESA":    int(p3069 in [1, 2, 3]),
        "SUBEMPLEADO":     int(p6800 < 32),
        "PLURIEMPLEO":     pluriempleo,
        "CONTRATO_VERBAL": int(p6450 == 1),
    }
    return pd.DataFrame([row])

# ══════════════════════════════════════════════════════════════════════════
# HEADER — Identidad y Pregunta de Oro
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.pregunta-oro {
    background: linear-gradient(135deg, #1e3a5f 0%, #2e6da4 100%);
    color: white; padding: 1.2rem 1.6rem; border-radius: 10px; margin-bottom: 0.5rem;
    font-size: 1.05rem; line-height: 1.6;
}
.pregunta-oro b { color: #ffd54f; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🇨🇴 Predicción de Informalidad Laboral · Colombia 2024")
st.caption(
    "Proyecto Final de Maestría · Juan Andrés Montoya · Julián David Mejía · "
    "GEIH 2024 – DANE  |  SI7006 · SI7007 · SI7009"
)

st.markdown("""
<div class="pregunta-oro">
<b>🏆 Pregunta de Oro:</b> ¿Cuáles son las características socioeconómicas y laborales que mejor
predicen la informalidad de un trabajador colombiano, y cómo puede este conocimiento focalizar
las intervenciones del Ministerio del Trabajo y el SENA hacia los perfiles con mayor probabilidad
de informalidad?
</div>
""", unsafe_allow_html=True)

# ── Carga de recursos ────────────────────────────────────────────────────
model = preprocessor = meta = None
try:
    model        = load_model()
    preprocessor = load_preprocessor()
    meta         = load_meta()
except Exception as e:
    st.warning(f"⚠️ Modelo no cargado — ejecuta primero `4.Modelamiento.ipynb`. ({e})")

df = load_data()

# ── KPIs ─────────────────────────────────────────────────────────────────
if meta:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Modelo campeón",   meta.get("model", "—"))
    k2.metric("F1-Score (test)",  f"{meta.get('test_f1',        0):.4f}",
              delta="≥ 0.75 ✓" if meta.get("test_f1", 0) >= 0.75 else None)
    k3.metric("AUC-ROC (test)",   f"{meta.get('test_auc',       0):.4f}",
              delta="≥ 0.80 ✓" if meta.get("test_auc", 0) >= 0.80 else None)
    k4.metric("Precisión (test)", f"{meta.get('test_precision', 0):.4f}")
    k5.metric("Recall (test)",    f"{meta.get('test_recall',    0):.4f}")

    tasa_real = 56.0
    if df is not None:
        tasa_real = round(
            (df["INFORMAL"] * df["FEX_C18"]).sum() / df["FEX_C18"].sum() * 100, 1
        )

    ia, ib, ic = st.columns(3)
    ia.metric("Tasa informalidad ponderada (DANE)",
              f"{tasa_real:.1f}%", help="Estimación con factores de expansión FEX_C18")
    ib.metric("Trabajadores en muestra", f"{meta.get('n_train', 0) + meta.get('n_test', 0):,}")
    ic.metric("Variables del modelo", f"{meta.get('n_features', 0)}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab_arq, tab_mapa, tab_eda, tab_modelo, tab_pred = st.tabs([
    "🏛️ Arquitectura & Pipeline",
    "🗺️ Mapa Departamental",
    "📊 Análisis Exploratorio",
    "📈 Desempeño del Modelo",
    "🔮 Predicción Individual",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Arquitectura & Pipeline  (SI7006)
# ══════════════════════════════════════════════════════════════════════════
with tab_arq:
    st.subheader("Ciclo de vida de los datos — SI7006 Almacenamiento y Procesamiento")

    col_pipe, col_tech = st.columns([1, 1], gap="large")

    with col_pipe:
        st.markdown("#### 🔄 Flujo de datos (Batch Pipeline)")
        st.code("""
ORIGEN
  DANE – GEIH 2024 (microdatos públicos)
  24 archivos CSV: 12 × Características Generales
                   12 × Ocupados
  ~817 550 registros totales

        │  Ingesta batch mensual (Pandas)
        │  Selección de columnas relevantes
        ▼

ALMACENAMIENTO CRUDO
  parquet/geih_2024_crudo.parquet  [6.2 MB]
  geih_2024.duckdb                 [tabla "geih", 358K filas]

        │  JOIN por DIRECTORIO + SECUENCIA_P + ORDEN
        │  Construcción variable INFORMAL (P6920)
        │  Imputación · encoding · feature engineering
        ▼

ALMACENAMIENTO PROCESADO
  parquet/train.parquet  [281 376 filas · 45 features]
  parquet/test.parquet   [ 70 345 filas · 45 features]

        │  Entrenamiento LR · RF · LightGBM · XGBoost
        │  Early stopping · threshold tuning
        │  Registro en MLflow (local)
        ▼

DESPLIEGUE
  outputs/champion_geih.pkl        [LightGBM · 11.5 MB]
  outputs/champion_geih_meta.json  [métricas y umbral]
  app.py → Streamlit Community Cloud (GitHub)
        """, language="text")

    with col_tech:
        st.markdown("#### 🖥️ Ambiente Tecnológico")
        st.markdown("""
| Capa | Tecnología | Rol |
|------|-----------|-----|
| **Ingesta** | Python · Pandas | Carga batch de 24 CSV GEIH |
| **Almacenamiento NoSQL** | Parquet (PyArrow) | Almacenamiento columnar eficiente |
| **Motor SQL analítico** | **DuckDB** | JOINs y consultas in-process sobre Parquet |
| **Procesamiento** | Pandas · NumPy | ETL y feature engineering |
| **ML — modelos** | scikit-learn · LightGBM | Entrenamiento y evaluación |
| **Tuning** | Optuna | Búsqueda bayesiana de hiperparámetros |
| **Tracking** | MLflow (local) | Registro de experimentos y artefactos |
| **Interpretabilidad** | SHAP | Importancia de variables por predicción |
| **Visualización** | Plotly · Streamlit | Dashboard interactivo |
| **Nube** | Streamlit Community Cloud | Despliegue público sin costo |
| **Control de versiones** | GitHub | Código, pipeline y reproducibilidad |
        """)

        st.markdown("#### 🏗️ Arquitectura de referencia")
        st.markdown("""
        Arquitectura **batch local → nube**:

        ```
        [Fuente CSV]  →  [Parquet + DuckDB]  →  [Pipeline ML]  →  [API Streamlit]
             ↑                  ↑                      ↑                 ↑
           DANE             PyArrow              scikit-learn         GitHub
                           in-process           LightGBM             Cloud
        ```

        - **Sin servicios de pago**: Google Colab + Drive para cómputo, Streamlit Cloud para despliegue.
        - **Escalable**: el pipeline se re-ejecuta con nuevas descargas anuales de la GEIH.
        - **Ingesta**: modo **batch** (mensual). No se requiere streaming para datos censales anuales.
        """)

    st.divider()
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### 📦 Origen y cobertura")
        st.markdown("""
- **Fuente:** DANE – Gran Encuesta Integrada de Hogares (GEIH) 2024
- **Cobertura:** 33 dominios geográficos (todo el país)
- **Período:** Enero – Diciembre 2024
- **Módulos utilizados:**
  - *Características Generales* (sociodemográfico)
  - *Ocupados* (laboral + variable objetivo)
- **Universo encuestado:** ~817 550 personas
- **Muestra analítica:** 351 721 ocupados con P6920 válido
- **Acceso:** [microdata.dane.gov.co](https://microdatos.dane.gov.co/index.php/catalog/819) — datos públicos anonimizados
        """)

    with col_b:
        st.markdown("#### 💾 Almacenamiento y persistencia")
        st.markdown("""
**Parquet (NoSQL columnar):**
- `geih_2024_crudo.parquet` — 6.2 MB
- `train.parquet` · `test.parquet`
- `datos_procesados.parquet` — 6.1 MB

**DuckDB (SQL analítico):**
- Tabla `geih` — 358 029 registros
- JOIN por clave compuesta:
  `DIRECTORIO + SECUENCIA_P + ORDEN`
- Consultas SQL sobre Parquet directamente

**Modelo serializado:**
- `champion_geih.pkl` — LightGBM (11.5 MB)
- `preprocessor.joblib` — pipeline scikit-learn
- `champion_geih_meta.json` — métricas y umbral óptimo
        """)

    with col_c:
        st.markdown("#### 🔗 Despliegue del modelo")
        st.markdown("""
**Persistencia:**
- Modelo serializado con `pickle`
- Preprocessor con `joblib`
- Publicados en repositorio GitHub

**API de predicción:**
- `model.predict_proba(X)` → score [0,1]
- Umbral óptimo: 0.41 (threshold tuning sobre validación)
- Input: 17 variables socioeconómicas y laborales

**Visualización (este dashboard):**
- Mapa interactivo por departamento
- EDA con filtros por zona y sector
- Predictor individual con gauge de riesgo
- Comparativa de modelos + SHAP

**Publicación:**
- Streamlit Community Cloud (GitHub → deploy)
- Enlace público sin costo
        """)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Mapa Departamental
# ══════════════════════════════════════════════════════════════════════════
with tab_mapa:
    st.subheader("Tasa de informalidad por departamento · GEIH 2024")

    if df is not None:
        f_col1, f_col2 = st.columns([2, 2])
        with f_col1:
            zona_sel = st.radio(
                "Filtrar por zona",
                ["Todas", "Cabecera municipal", "Rural"],
                horizontal=True,
            )
        with f_col2:
            sector_opts = {"Todos los sectores": None}
            sector_opts.update({v: k for k, v in RAMA_CIIU.items()})
            sector_sel = st.selectbox("Filtrar por sector económico", list(sector_opts.keys()))

        df_mapa = df.copy()
        if zona_sel == "Cabecera municipal":
            df_mapa = df_mapa[df_mapa["CLASE"] == 1]
        elif zona_sel == "Rural":
            df_mapa = df_mapa[df_mapa["CLASE"] == 2]
        if sector_opts[sector_sel] is not None:
            df_mapa = df_mapa[df_mapa["RAMA2D_R4"] == sector_opts[sector_sel]]

        if len(df_mapa) == 0:
            st.warning("No hay datos para la combinación de filtros seleccionada.")
        else:
            inf_dpto = (
                df_mapa[df_mapa["DPTO"].notna()]
                .groupby("DPTO", observed=True)
                .apply(tasa_pond_grp, include_groups=False)
                .reset_index()
                .rename(columns={0: "tasa"})
                .query("tasa > 0")
            )
            inf_dpto["DPTO"]     = inf_dpto["DPTO"].astype(int)
            inf_dpto["nombre"]   = inf_dpto["DPTO"].map(lambda d: DPTO_INFO.get(d, (str(d), 0, 0))[0])
            inf_dpto["lat"]      = inf_dpto["DPTO"].map(lambda d: DPTO_INFO.get(d, ("", 4.5, -74))[1])
            inf_dpto["lon"]      = inf_dpto["DPTO"].map(lambda d: DPTO_INFO.get(d, ("", 4.5, -74))[2])
            inf_dpto["tasa_pct"] = inf_dpto["tasa"].round(1)

            titulo_mapa = f"Tasa de informalidad (%) — {zona_sel} · {sector_sel}"
            fig_map = px.scatter_geo(
                inf_dpto, lat="lat", lon="lon",
                size="tasa_pct", color="tasa_pct",
                hover_name="nombre",
                hover_data={"tasa_pct": True, "lat": False, "lon": False},
                color_continuous_scale="RdYlBu_r",
                range_color=[20, 90], size_max=45,
                labels={"tasa_pct": "% Informal"},
                title=titulo_mapa,
            )
            fig_map.update_geos(
                visible=False, showcountries=True, countrycolor="gray",
                showsubunits=True, subunitcolor="lightgray",
                lonaxis_range=[-83, -65], lataxis_range=[-5, 14],
                bgcolor="aliceblue",
            )
            fig_map.update_layout(height=520, margin=dict(r=0, t=40, l=0, b=0))
            st.plotly_chart(fig_map, use_container_width=True)

            m_col1, m_col2 = st.columns([1, 1])
            with m_col1:
                st.markdown("**Departamentos con mayor informalidad**")
                st.dataframe(
                    inf_dpto[["nombre", "tasa_pct"]]
                        .sort_values("tasa_pct", ascending=False)
                        .head(10)
                        .rename(columns={"nombre": "Departamento", "tasa_pct": "% Informal"})
                        .reset_index(drop=True),
                    use_container_width=True, height=320,
                )
            with m_col2:
                st.markdown("**Departamentos con menor informalidad**")
                st.dataframe(
                    inf_dpto[["nombre", "tasa_pct"]]
                        .sort_values("tasa_pct", ascending=True)
                        .head(10)
                        .rename(columns={"nombre": "Departamento", "tasa_pct": "% Informal"})
                        .reset_index(drop=True),
                    use_container_width=True, height=320,
                )
    else:
        np.random.seed(42)
        rows = [(k, round(np.random.uniform(0.35, 0.82), 3)) for k in DPTO_INFO]
        inf_dpto = pd.DataFrame(rows, columns=["DPTO", "tasa"])
        inf_dpto["nombre"]   = inf_dpto["DPTO"].map(lambda d: DPTO_INFO.get(d, (str(d), 0, 0))[0])
        inf_dpto["lat"]      = inf_dpto["DPTO"].map(lambda d: DPTO_INFO.get(d, ("", 4.5, -74))[1])
        inf_dpto["lon"]      = inf_dpto["DPTO"].map(lambda d: DPTO_INFO.get(d, ("", 4.5, -74))[2])
        inf_dpto["tasa_pct"] = (inf_dpto["tasa"] * 100).round(1)
        st.info("Datos de ejemplo — ejecuta los notebooks para datos reales.")
        fig_map = px.scatter_geo(
            inf_dpto, lat="lat", lon="lon",
            size="tasa_pct", color="tasa_pct",
            hover_name="nombre",
            color_continuous_scale="RdYlBu_r",
            range_color=[30, 85], size_max=40,
            labels={"tasa_pct": "% Informal"},
            title="Tasa de informalidad laboral (%) por departamento",
        )
        fig_map.update_geos(
            visible=False, showcountries=True, countrycolor="gray",
            showsubunits=True, subunitcolor="lightgray",
            lonaxis_range=[-83, -65], lataxis_range=[-5, 14],
            bgcolor="aliceblue",
        )
        fig_map.update_layout(height=520, margin=dict(r=0, t=40, l=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA Interactivo
# ══════════════════════════════════════════════════════════════════════════
with tab_eda:
    if df is None:
        st.info("Ejecuta los notebooks para cargar datos reales.")
    else:
        # Fila 1: Distribución general y por zona
        c1, c2 = st.columns(2)
        with c1:
            n_f = (df["INFORMAL"] == 0).sum()
            n_i = (df["INFORMAL"] == 1).sum()
            fig_pie = go.Figure(go.Pie(
                labels=["Formal", "Informal"],
                values=[n_f, n_i],
                marker_colors=["#2196F3", "#FF5722"],
                textinfo="label+percent",
                hole=0.35,
            ))
            fig_pie.update_layout(
                title=f"Distribución Formal / Informal (n={n_f+n_i:,})",
                height=330, margin=dict(t=40, b=10),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            t_zona = (
                df[df["CLASE"].notna()]
                .groupby("CLASE", observed=True)
                .apply(tasa_pond_grp, include_groups=False)
                .reset_index()
                .rename(columns={0: "tasa"})
            )
            t_zona["pct"]      = t_zona["tasa"].round(1)
            t_zona["etiqueta"] = t_zona["CLASE"].map({1: "Cabecera", 2: "Rural"})
            fig_zona = px.bar(
                t_zona, x="etiqueta", y="pct",
                color="etiqueta",
                color_discrete_sequence=["#5DCAA5", "#EF9F27"],
                title="Informalidad por zona (%, ponderada)",
                labels={"pct": "% Informal", "etiqueta": ""},
                text="pct",
            )
            fig_zona.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_zona.update_layout(height=330, showlegend=False,
                                   margin=dict(t=40, b=10), yaxis_range=[0, 100])
            st.plotly_chart(fig_zona, use_container_width=True)

        # Fila 2: Por posición ocupacional
        LABELS_POS = {
            1: "Empleado particular", 2: "Empleado gobierno",
            3: "Doméstico",           4: "Cuenta propia",
            5: "Empleador",           6: "Familiar s/rem.",
            7: "Jornalero",           8: "Otro",
        }
        t_pos = (
            df[df["P6430"].notna()]
            .groupby("P6430", observed=True)
            .apply(tasa_pond_grp, include_groups=False)
            .reset_index()
            .rename(columns={0: "tasa"})
            .sort_values("tasa")
        )
        t_pos["pct"]      = t_pos["tasa"].round(1)
        t_pos["etiqueta"] = t_pos["P6430"].map(LABELS_POS)
        fig_pos = px.bar(
            t_pos, x="pct", y="etiqueta", orientation="h",
            color="pct", color_continuous_scale="RdYlGn_r",
            title="Informalidad por posición ocupacional (%, ponderada)",
            labels={"pct": "% Informal", "etiqueta": ""},
            text="pct",
        )
        fig_pos.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_pos.update_layout(coloraxis_showscale=False, height=320,
                               margin=dict(t=40, b=10))
        st.plotly_chart(fig_pos, use_container_width=True)

        # Fila 3: Por nivel educativo y grupo de edad
        c3, c4 = st.columns(2)
        with c3:
            if "P3042" in df.columns:
                t_edu = (
                    df[df["P3042"].notna()]
                    .groupby("P3042", observed=True)
                    .apply(tasa_pond_grp, include_groups=False)
                    .reset_index()
                    .rename(columns={0: "tasa"})
                )
                t_edu["pct"]      = t_edu["tasa"].round(1)
                t_edu["etiqueta"] = t_edu["P3042"].map(LABELS_EDU)
                fig_edu = px.bar(
                    t_edu, x="etiqueta", y="pct",
                    color="pct", color_continuous_scale="RdYlGn_r",
                    title="Informalidad por nivel educativo (%, ponderada)",
                    labels={"pct": "% Informal", "etiqueta": ""},
                    text="pct",
                )
                fig_edu.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
                fig_edu.update_layout(coloraxis_showscale=False, height=340,
                                       xaxis_tickangle=-35, margin=dict(t=40, b=80),
                                       yaxis_range=[0, 110])
                st.plotly_chart(fig_edu, use_container_width=True)

        with c4:
            if "P6040" in df.columns:
                df_age = df[df["P6040"].between(15, 74)].copy()
                df_age["EDAD_Q"] = pd.cut(
                    df_age["P6040"],
                    bins=range(14, 76, 5),
                    labels=[f"{i}-{i+4}" for i in range(15, 75, 5)],
                )
                t_edad = (
                    df_age[df_age["EDAD_Q"].notna()]
                    .groupby("EDAD_Q", observed=True)
                    .apply(tasa_pond_grp, include_groups=False)
                    .reset_index()
                    .rename(columns={0: "tasa"})
                )
                t_edad["pct"] = t_edad["tasa"].round(1)
                fig_edad = go.Figure()
                fig_edad.add_trace(go.Scatter(
                    x=t_edad["EDAD_Q"].astype(str),
                    y=t_edad["pct"],
                    mode="lines+markers+text",
                    line=dict(color="#E05C5C", width=2.5),
                    marker=dict(size=8),
                    fill="tozeroy",
                    fillcolor="rgba(224,92,92,0.10)",
                    text=t_edad["pct"].astype(str) + "%",
                    textposition="top center",
                ))
                fig_edad.update_layout(
                    title="Informalidad por grupo de edad (%, ponderada)",
                    xaxis_title="Grupo de edad",
                    yaxis_title="% Informal",
                    height=340,
                    margin=dict(t=40, b=40),
                    yaxis_range=[0, 100],
                )
                st.plotly_chart(fig_edad, use_container_width=True)

        # Fila 4: Distribución de ingresos (si disponible)
        if "INGLABO" in df.columns:
            df_ing  = df[df["INGLABO"].notna() & (df["INGLABO"] > 0)].copy()
            df_samp = df_ing.sample(min(6000, len(df_ing)), random_state=42)
            df_samp["Condición"] = df_samp["INFORMAL"].map({0: "Formal", 1: "Informal"})
            df_samp["log_ing"]   = np.log1p(df_samp["INGLABO"])
            fig_violin = px.violin(
                df_samp, x="Condición", y="log_ing",
                color="Condición",
                color_discrete_map={"Formal": "#2196F3", "Informal": "#FF5722"},
                box=True,
                title="Distribución de log(Ingreso laboral) por condición de empleo",
                labels={"log_ing": "log(INGLABO + 1)", "Condición": ""},
            )
            fig_violin.update_layout(height=360, showlegend=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig_violin, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Desempeño del Modelo
# ══════════════════════════════════════════════════════════════════════════
with tab_modelo:
    st.subheader("Comparativa y desempeño del modelo campeón")

    # Tabla de métricas
    df_metr = load_metrics()
    if df_metr is not None:
        st.markdown("#### Comparativa de modelos — test set (n = 70 345)")
        st.dataframe(
            df_metr.style
                .highlight_max(subset=["F1", "AUC-ROC", "Precision", "Recall"],
                               color="#d4edda")
                .format({"F1": "{:.4f}", "AUC-ROC": "{:.4f}",
                         "Precision": "{:.4f}", "Recall": "{:.4f}"}),
            use_container_width=True,
        )

        # Gráfica comparativa interactiva
        metricas = ["F1", "AUC-ROC", "Precision", "Recall"]
        df_long = df_metr.melt(id_vars="Modelo", value_vars=metricas,
                                var_name="Métrica", value_name="Valor")
        fig_comp = px.bar(
            df_long, x="Métrica", y="Valor", color="Modelo",
            barmode="group",
            title="Métricas por modelo (test set)",
            labels={"Valor": "Score"},
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig_comp.update_layout(height=380, yaxis_range=[0.85, 1.0],
                                margin=dict(t=40, b=10))
        st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # Imágenes generadas en los notebooks
    img_c1, img_c2 = st.columns(2)
    with img_c1:
        if IMG_CONFUSION.exists():
            st.markdown("#### Matriz de confusión — LightGBM (test set)")
            st.image(str(IMG_CONFUSION), use_container_width=True)
        if IMG_COMP.exists():
            st.markdown("#### Comparativa visual de modelos")
            st.image(str(IMG_COMP), use_container_width=True)

    with img_c2:
        if IMG_SHAP_BAR.exists():
            st.markdown("#### Importancia de variables (|SHAP| medio)")
            st.image(str(IMG_SHAP_BAR), use_container_width=True)
        if IMG_SHAP_BEE.exists():
            st.markdown("#### SHAP beeswarm — distribución de impactos")
            st.image(str(IMG_SHAP_BEE), use_container_width=True)

    if not any([IMG_CONFUSION.exists(), IMG_COMP.exists(),
                IMG_SHAP_BAR.exists(), IMG_SHAP_BEE.exists()]):
        st.info("Ejecuta `4.Modelamiento.ipynb` y `5.Visualizacion.ipynb` para generar las imágenes.")

    st.divider()
    st.markdown("""
    **Interpretación del modelo campeón (LightGBM):**

    | Variable | Dirección del impacto | Importancia SHAP |
    |---|---|---|
    | Microempresa (≤10 empleados) | ↑ Mayor informalidad | **#1** (1.66) |
    | Contrato verbal | ↑ Mayor informalidad | **#2** (0.58) |
    | Años de educación | ↓ Reduce informalidad | **#3** (0.56) |
    | Tamaño del establecimiento | ↓ Mayor empresa = menor riesgo | **#4** (0.51) |
    | Departamento | Varía por región | **#5** (0.40) |
    | Cuenta propia | ↑ Mayor informalidad | **#6** (0.40) |
    | Edad | ↑ Curva no lineal (jóvenes y mayores) | **#7** (0.37) |
    """)

# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — Predicción Individual
# ══════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("Estima la probabilidad de informalidad de un trabajador")
    st.markdown(
        "Ingresa las características de un trabajador para obtener su **score de riesgo de informalidad** "
        "según el modelo LightGBM entrenado sobre GEIH 2024."
    )

    if model is None or preprocessor is None:
        st.error("Carga el modelo ejecutando `4.Modelamiento.ipynb` primero.")
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Datos demográficos**")
            edad      = st.slider("Edad (años)", 15, 75, 32)
            anios_edu = st.slider("Años de educación acumulados", 0, 25, 11)
            sexo_lbl  = st.selectbox("Sexo", ["Hombre", "Mujer"])
            p3271     = 1 if sexo_lbl == "Hombre" else 2
            ecivil_lbl = st.selectbox("Estado civil", list(ESTADO_CIVIL.values()))
            p6070     = {v: k for k, v in ESTADO_CIVIL.items()}[ecivil_lbl]
            zona_lbl  = st.selectbox("Zona", ["Cabecera municipal", "Rural"])
            clase     = 1 if zona_lbl == "Cabecera municipal" else 2
            dpto_lbl  = st.selectbox("Departamento", [v[0] for v in DPTO_INFO.values()])
            dpto      = {v[0]: k for k, v in DPTO_INFO.items()}.get(dpto_lbl, 11)

        with col_r:
            st.markdown("**Datos laborales**")
            pos_lbl   = st.selectbox("Posición ocupacional", list(POSICION.values()))
            p6430     = {v: k for k, v in POSICION.items()}[pos_lbl]
            cont_lbl  = st.selectbox("Tipo de contrato", list(CONTRATO_TIPO.values()))
            p6450     = {v: k for k, v in CONTRATO_TIPO.items()}[cont_lbl]
            p6800     = st.slider("Horas trabajadas / semana", 1, 100, 48)
            tam_lbl   = st.selectbox("Tamaño del establecimiento", list(TAMANO_EMP.values()))
            p3069     = {v: k for k, v in TAMANO_EMP.items()}[tam_lbl]
            rama_lbl  = st.selectbox("Rama de actividad (CIIU)", list(RAMA_CIIU.values()))
            rama      = {v: k for k, v in RAMA_CIIU.items()}[rama_lbl]
            pluriemp  = st.checkbox("¿Tiene otro empleo adicional (pluriempleo)?", value=False)
            p7040_val = 1 if pluriemp else 0

        input_raw = construir_input(
            p6040=edad,    p3271=p3271,  p6070=p6070,  clase=clase,  dpto=dpto,
            p6430=p6430,   p6450=p6450,  p6800=p6800,  p3069=p3069,  rama=rama,
            anios_edu=anios_edu, pluriempleo=p7040_val,
        )

        try:
            X_input = preprocessor.transform(input_raw)
            thresh  = meta.get("threshold", 0.5) if meta else 0.5
            prob    = model.predict_proba(X_input)[0, 1]
            pred    = int(prob >= thresh)

            st.divider()
            m1, m2, m3 = st.columns(3)
            riesgo = "Alto" if prob > 0.65 else "Medio" if prob > 0.4 else "Bajo"
            color_delta = "off"
            m1.metric("Probabilidad de informalidad", f"{prob:.1%}")
            m2.metric("Nivel de riesgo",              riesgo)
            m3.metric("Clasificación",                "INFORMAL" if pred == 1 else "FORMAL")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#FF5722" if prob >= thresh else "#2196F3"},
                    "steps": [
                        {"range": [0,  40],  "color": "#E3F2FD"},
                        {"range": [40, 65],  "color": "#FFF9C4"},
                        {"range": [65, 100], "color": "#FFEBEE"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": thresh * 100,
                    },
                },
                title={"text": f"Score de informalidad (umbral = {thresh:.0%})"},
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Implicación de política
            if pred == 1:
                st.warning(
                    f"**⚠️ Perfil de ALTO riesgo** — Probabilidad {prob:.1%}. "
                    "Este trabajador debería ser priorizado en programas del SENA, "
                    "inspecciones del Ministerio del Trabajo o subsidios de formalización."
                )
            else:
                st.success(
                    f"**✅ Perfil FORMAL** — Probabilidad de informalidad {prob:.1%}. "
                    "El trabajador presenta características asociadas al empleo formal."
                )

            with st.expander("📊 Factores de riesgo clave según el modelo SHAP"):
                st.markdown("""
| Factor | Dirección | Impacto relativo |
|---|---|---|
| Microempresa (≤ 10 trabajadores) | ↑ Mayor riesgo | ★★★★★ |
| Contrato verbal | ↑ Mayor riesgo | ★★★★☆ |
| Cuenta propia / Jornalero | ↑ Mayor riesgo | ★★★★☆ |
| Zona rural | ↑ Mayor riesgo (~15 pp) | ★★★☆☆ |
| Sector agricultura / construcción | ↑ Mayor riesgo | ★★★☆☆ |
| Años de educación | ↓ Reduce riesgo | ★★★★☆ |
| Empresa grande (> 30 empleados) | ↓ Reduce riesgo | ★★★★☆ |
| Contrato escrito / indefinido | ↓ Reduce riesgo | ★★★★★ |
                """)

        except Exception as ex:
            st.error(f"Error en predicción: {ex}")
            st.info("Verifica que el preprocessor y el modelo sean del mismo pipeline.")

# ══════════════════════════════════════════════════════════════════════════
# FOOTER — Hallazgos y recomendaciones
# ══════════════════════════════════════════════════════════════════════════
st.divider()

with st.expander("📌 Hallazgos clave y recomendaciones de política pública", expanded=False):
    hall_col, rec_col = st.columns(2)

    with hall_col:
        st.markdown("#### 🔍 Hallazgos principales")
        st.markdown("""
**1. Magnitud estructural:**
El 56% de los trabajadores colombianos son informales (tasa ponderada DANE 2024). La informalidad afecta de manera desproporcionada a zonas rurales (~75%) vs cabeceras (~52%).

**2. Predictores dominantes (SHAP):**
- Trabajar en microempresa (≤10 empleados) es el factor #1, con el mayor impacto individual.
- El contrato verbal duplica el riesgo; el escrito lo reduce drásticamente.
- Cada año adicional de educación reduce ~3 pp la probabilidad de informalidad.

**3. Geografía:**
Departamentos como Vichada, Guainía y Vaupés superan el 85% de informalidad; Bogotá y Atlántico están por debajo del 45%.

**4. Sectores críticos:**
Agricultura, construcción, hogares con servicio doméstico y comercio informal concentran las tasas más altas (>70%).

**5. Rendimiento del modelo:**
LightGBM logra AUC-ROC = 0.985 y F1 = 0.958 — muy por encima de las metas del proyecto (AUC ≥ 0.80, F1 ≥ 0.75).
        """)

    with rec_col:
        st.markdown("#### 💡 Recomendaciones accionables")
        st.markdown("""
**Para el Ministerio del Trabajo:**
- Usar el score individual del modelo en operativos de inspección, priorizando microempresas rurales de los sectores agrícola y construcción.
- Crear umbrales de score para escalonar el nivel de intervención: notificación (>40%), visita (>65%), sanción preventiva (>80%).

**Para el SENA:**
- Focalizar programas de capacitación y formalización en departamentos con tasa > 70% y en jóvenes (15-24 años) de zonas rurales.
- Incentivar la transición de contratos verbales a escritos mediante asesoría gratuita en los municipios con mayor índice de informalidad.

**Para Planeación Nacional:**
- Integrar el modelo en los sistemas de información del FILCO (Ministerio del Trabajo) para que los indicadores sean individuales, no solo agregados departamentales.
- Re-entrenar el modelo con cada nueva ola anual de la GEIH para mantener vigencia.

**Para la academia / investigación:**
- Ampliar el análisis con datos longitudinales para medir el impacto de intervenciones de formalización sobre el score individual a lo largo del tiempo.
        """)

st.caption(
    "Fuente: DANE – Gran Encuesta Integrada de Hogares 2024. "
    "El DANE no avala los resultados del análisis. "
    "Maestría en Ciencia de Datos y Analítica · EAFIT · 2025."
)
