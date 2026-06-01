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
from groq import Groq

st.set_page_config(
    page_title="Informalidad Laboral · Colombia 2025",
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
    1:  "Agricultura/Pesca",            5:  "Minería",
    10: "Industria manufacturera",      14: "Confección/Textil",
    25: "Productos metálicos",          36: "Agua/Servicios",
    41: "Construcción",                 42: "Obras civiles",
    43: "Construcción especializada",   45: "Comercio vehículos",
    46: "Comercio al por mayor",        47: "Comercio al por menor",
    49: "Transporte",                   52: "Almacenamiento/Apoyo transp.",
    55: "Alojamiento",                  56: "Restaurantes/Comidas",
    58: "Comunicaciones",               64: "Financiero",
    68: "Inmobiliario",                 69: "Jurídico/Contabilidad",
    75: "Veterinaria",                  78: "Servicios adm.",
    81: "Servicios a edificios",        82: "Servicios de oficina",
    84: "Administración pública",       85: "Educación",
    86: "Salud",                        90: "Artes/Entretenimiento",
    95: "Reparación equipos",           96: "Otros servicios personales",
    97: "Hogares con servicio dom.",    99: "Otro/NS",
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
        df = pd.read_parquet(DATA_PATH)
        if "P6430" in df.columns:
            df = df[df["P6430"] == 4].copy()
        return df
    return None

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def compute_shap_beeswarm():
    """Carga modelo + test desde disco, calcula SHAP para 800 muestras.
    Devuelve tipos serializables para el caché de Streamlit."""
    try:
        import shap as _shap, pickle as _pk
        _mp = BASE / "outputs" / "champion_geih.pkl"
        _tp = BASE / "parquet"  / "test.parquet"
        if not _mp.exists() or not _tp.exists():
            return None
        with open(_mp, "rb") as fh:
            _mdl = _pk.load(fh)
        df_t = pd.read_parquet(_tp)
        X_s  = df_t.drop("INFORMAL", axis=1).sample(
            min(800, len(df_t)), random_state=42
        )
        sv = _shap.TreeExplainer(_mdl).shap_values(X_s)
        sv = sv[1] if isinstance(sv, list) else sv
        return {
            "sv":   sv.tolist(),
            "X":    X_s.values.tolist(),
            "cols": X_s.columns.tolist(),
        }
    except Exception:
        return None

# ── Helper: tasa ponderada ───────────────────────────────────────────────
def tasa_pond_grp(g):
    return (g["INFORMAL"] * g["FEX_C18"]).sum() / g["FEX_C18"].sum() * 100

# ── Helper: construir input para el preprocessor ─────────────────────────
def construir_input(p6040, p3271, p6070, clase, dpto,
                    p6800, p3069, rama,
                    anios_edu, pluriempleo):
    """Construye el vector de entrada para el modelo de trabajadores independientes.
    El modelo fue entrenado exclusivamente sobre P6430 == 4 (Cuenta propia).
    P6450 (tipo de contrato) excluida: proxy del target para trabajadores independientes.
    """
    mapa_edu = {1:0, 2:1, 3:3, 4:5, 5:7, 6:9, 7:10,
                8:11, 9:14, 10:16, 11:17, 12:18, 13:21}
    bins   = [14, 24, 34, 44, 54, 64, 120]
    labels = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    edad_grupo = pd.cut([p6040], bins=bins, labels=labels)[0]
    row = {
        "P3271":        p3271,
        "P6040":        p6040,
        "EDAD_GRUPO":   str(edad_grupo),
        "ANIOS_EDU":    anios_edu,
        "P6070":        p6070,
        "CLASE":        clase,
        "DPTO":         dpto,
        "P6800":        p6800,
        "P6450":        9,       # NS/NR — oculto; excluida de próxima versión del modelo
        "P3069":        p3069,
        "RAMA2D_R4":    rama,
        "MICROEMPRESA": int(p3069 in [1, 2, 3]),
        "SUBEMPLEADO":  int(p6800 < 32),
        "PLURIEMPLEO":  pluriempleo,
    }
    return pd.DataFrame([row])

# ── Recomendaciones de política pública con Groq AI ───────────────────────
def generar_recomendaciones_ia(api_key: str, perfil: dict, prob: float, riesgo: str) -> str:
    """Llama a Groq (Llama 3.3 70B) para generar recomendaciones de política pública."""
    client = Groq(api_key=api_key)

    perfil_texto = "\n".join(f"- {k}: {v}" for k, v in perfil.items())

    prompt = f"""Eres un asesor estratégico creativo del Ministerio de Trabajo de Colombia.
Tu misión es diseñar intervenciones de política pública personalizadas y realizables
para reducir la informalidad laboral de trabajadores independientes (cuenta propia).

PERFIL DEL TRABAJADOR ANALIZADO:
{perfil_texto}

DIAGNÓSTICO DEL MODELO:
- Probabilidad de informalidad: {prob:.1%}
- Nivel de riesgo: {riesgo}

CONTEXTO: Este trabajador fue clasificado como INFORMAL por un modelo LightGBM
entrenado sobre la Gran Encuesta Integrada de Hogares (GEIH 2025) del DANE,
con AUC-ROC de 0.95 y F1 de 0.96. La informalidad aquí significa no cotizar
a pensión (P6920), lo que implica desprotección ante vejez, enfermedad y muerte.

INSTRUCCIONES:
1. Analiza cuál o cuáles factores del perfil son los determinantes principales de informalidad.
2. Genera entre 5 y 7 recomendaciones concretas, innovadoras y accionables.
3. Para cada recomendación incluye:
   - Un título con impacto (usa lenguaje persuasivo y directo)
   - La acción específica que debe tomar el Ministerio
   - Por qué esta acción impacta directamente el perfil de este trabajador
   - Una métrica de éxito que se pueda medir en 12 meses
4. Cierra con una "Palanca crítica": el único cambio que, si se logra, tiene mayor probabilidad
   de quebrar la cadena de informalidad para este perfil específico.

Usa formato markdown con encabezados claros. Sé creativo pero realista con el contexto colombiano.
No repitas el perfil en la respuesta. Ve directo a las recomendaciones. Responde en español."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content




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

st.title("🇨🇴 Predicción de Informalidad Laboral · Colombia 2025")
st.caption(
    "Proyecto Final de Maestría · Juan Andrés Montoya · Julián David Mejía · "
    "GEIH 2025 – DANE  |  SI7006 · SI7007 · SI7009"
)

st.markdown("""
<div class="pregunta-oro">
<b>🏆 Pregunta de Oro:</b> ¿Por qué los trabajadores independientes en Colombia permanecen informales,
y qué condiciones deben cambiar para que la formalización sea una opción viable para ellos?<br><br>
<span style='font-size:0.95rem; color:#b3cde3;'>
<b>Hipótesis:</b> La informalidad en trabajadores independientes no es una elección aleatoria sino un resultado
predecible de condiciones estructurales — sector, tamaño del entorno laboral, zona geográfica y nivel educativo.
Intervenir sobre esas condiciones, en lugar de ofrecer programas universales, aumentaría significativamente
la efectividad de los programas de formalización existentes.
</span>
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

# ── Sidebar: configuración API + contexto de referencia ───────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración IA")
    st.caption(
        "Ingresa tu API Key de Groq para habilitar las "
        "recomendaciones de política pública generadas por IA (Llama 3.3 70B). "
        "Gratis en console.groq.com."
    )
    anthropic_api_key = st.text_input(
        "API Key de Groq",
        type="password",
        placeholder="gsk_...",
        help="Obtén tu clave gratis en console.groq.com. No se almacena en ningún servidor.",
    )
    if anthropic_api_key:
        st.success("API Key cargada — recomendaciones IA activas")
    else:
        st.info("Sin API Key — recomendaciones IA desactivadas")

    # ── Contexto de referencia ─────────────────────────────────────────
    _df_ref = load_data()
    if _df_ref is not None:
        st.markdown("---")
        st.markdown("### 📊 Referencia: % informalidad por educación")
        st.caption("Tasa ponderada GEIH 2024 · cuenta propia")
        _labels_edu_ref = {
            1:"Ninguno", 2:"Preescolar", 3:"Primaria inc.", 4:"Primaria",
            5:"Sec. inc.", 6:"Secundaria", 7:"Media inc.", 8:"Media",
            9:"Técnica", 10:"Universidad", 11:"Especializ.", 12:"Maestría", 13:"Doctorado",
        }
        if "P3042" in _df_ref.columns and "FEX_C18" in _df_ref.columns:
            _tasa_edu_ref = (
                _df_ref[_df_ref["P3042"].notna()]
                .groupby("P3042", observed=True)
                .apply(tasa_pond_grp, include_groups=False)
                .reset_index().rename(columns={0: "tasa"})
            )
            _tasa_edu_ref["nivel"] = _tasa_edu_ref["P3042"].map(_labels_edu_ref)
            _tasa_edu_ref["pct"]   = _tasa_edu_ref["tasa"].round(1)
            _tasa_edu_ref = _tasa_edu_ref[_tasa_edu_ref["P3042"] <= 13].copy()
            st.dataframe(
                _tasa_edu_ref[["nivel", "pct"]]
                    .rename(columns={"nivel": "Nivel educativo", "pct": "% Informal"})
                    .reset_index(drop=True),
                use_container_width=True, hide_index=True, height=290,
            )

        st.markdown("### 🌍 Referencia: % informalidad por demografía")
        st.caption("Tasa ponderada · cuenta propia")

        # Zona
        if "CLASE" in _df_ref.columns:
            _t_zona = (
                _df_ref[_df_ref["CLASE"].notna()]
                .groupby("CLASE", observed=True)
                .apply(tasa_pond_grp, include_groups=False)
                .reset_index().rename(columns={0: "tasa"})
            )
            _t_zona["zona"] = _t_zona["CLASE"].map({1: "Cabecera", 2: "Rural"})
            _t_zona["pct"]  = _t_zona["tasa"].round(1)
            st.markdown("**Por zona**")
            st.dataframe(
                _t_zona[["zona", "pct"]]
                    .rename(columns={"zona": "Zona", "pct": "% Informal"})
                    .reset_index(drop=True),
                use_container_width=True, hide_index=True, height=90,
            )

        # Sexo
        if "P3271" in _df_ref.columns:
            _t_sexo = (
                _df_ref[_df_ref["P3271"].notna()]
                .groupby("P3271", observed=True)
                .apply(tasa_pond_grp, include_groups=False)
                .reset_index().rename(columns={0: "tasa"})
            )
            _t_sexo["sexo"] = _t_sexo["P3271"].map({1: "Hombre", 2: "Mujer"})
            _t_sexo["pct"]  = _t_sexo["tasa"].round(1)
            st.markdown("**Por sexo**")
            st.dataframe(
                _t_sexo[["sexo", "pct"]]
                    .rename(columns={"sexo": "Sexo", "pct": "% Informal"})
                    .reset_index(drop=True),
                use_container_width=True, hide_index=True, height=90,
            )

        # Grupo de edad
        if "P6040" in _df_ref.columns:
            _df_ref2 = _df_ref[_df_ref["P6040"].notna()].copy()
            _df_ref2["EDAD_G"] = pd.cut(
                _df_ref2["P6040"],
                bins=[14,24,34,44,54,64,120],
                labels=["15-24","25-34","35-44","45-54","55-64","65+"]
            )
            _t_edad = (
                _df_ref2[_df_ref2["EDAD_G"].notna()]
                .groupby("EDAD_G", observed=True)
                .apply(tasa_pond_grp, include_groups=False)
                .reset_index().rename(columns={0: "tasa"})
            )
            _t_edad["pct"] = _t_edad["tasa"].round(1)
            st.markdown("**Por grupo de edad**")
            st.dataframe(
                _t_edad[["EDAD_G", "pct"]]
                    .rename(columns={"EDAD_G": "Edad", "pct": "% Informal"})
                    .reset_index(drop=True),
                use_container_width=True, hide_index=True, height=210,
            )

    st.markdown("---")
    st.caption("Proyecto Final · Maestría · GEIH 2025")

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab_arq, tab_dash, tab_dpto, tab_modelo, tab_pred = st.tabs([
    "🏛️ Arquitectura & Pipeline",
    "📊 ¿Por qué permanecen informales?",
    "🗺️ Informalidad por departamento",
    "📈 Desempeño del Modelo",
    "🔮 Predicción Individual",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Arquitectura & Pipeline  (SI7006)
# ══════════════════════════════════════════════════════════════════════════
with tab_arq:
    col_pipe, col_tech = st.columns([1, 1], gap="large")

    with col_pipe:
        st.markdown("#### Pipeline de datos — Batch")
        st.code("""
ORIGEN
  DANE – GEIH 2025 · 24 CSV · 817 550 registros
  Ingesta batch mensual (Pandas)
        │
        ▼
ALMACENAMIENTO CRUDO
  geih_2025_crudo.parquet  [6.2 MB]
  geih_2025.duckdb         [358K filas · SQL analítico]
  JOIN: DIRECTORIO + SECUENCIA_P + ORDEN
        │
        ▼
PREPROCESAMIENTO
  train.parquet  [121K filas · 13 features]
  test.parquet   [ 30K filas · 13 features]
  preprocessor.joblib  [ColumnTransformer]
        │
        ▼
MODELADO
  LightGBM · RF · XGBoost · Regresión Logística
  Early stopping · Threshold tuning (0.41)
        │
        ▼
DESPLIEGUE
  champion_geih.pkl  [11.5 MB]
  app.py → Streamlit Community Cloud
        """, language="text")

    with col_tech:
        st.markdown("#### Stack tecnológico")
        st.markdown("""
| Capa | Tecnología |
|------|-----------|
| **Ingesta** | Python · Pandas |
| **Almacenamiento** | Parquet (PyArrow) · DuckDB |
| **Procesamiento** | scikit-learn · ColumnTransformer |
| **Modelado** | LightGBM · XGBoost · scikit-learn |
| **Interpretabilidad** | SHAP (TreeExplainer) |
| **Visualización** | Plotly · Streamlit |
| **Despliegue** | Streamlit Community Cloud · GitHub |
        """)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — ¿Por qué permanecen informales?
# ══════════════════════════════════════════════════════════════════════════
with tab_dash:
    if df is None:
        st.info("Ejecuta los notebooks para cargar datos reales.")
    else:
        tasa_global = (df["INFORMAL"] * df["FEX_C18"]).sum() / df["FEX_C18"].sum() * 100
        n_total     = len(df)
        n_informal  = (df["INFORMAL"] == 1).sum()

        # ── Banner ──────────────────────────────────────────────────────────────────────────────────────
        st.markdown(
            f'''<div style="background:linear-gradient(135deg,#6B0F1A,#A93226);
                    color:white;padding:1.4rem 1.8rem;border-radius:10px;margin-bottom:1rem;">
            <div style="font-size:3rem;font-weight:900;line-height:1;">{tasa_global:.0f}%</div>
            <div style="font-size:1.15rem;margin-top:0.4rem;">
                de los trabajadores independientes colombianos son informales
            </div>
            <div style="font-size:0.95rem;color:#f8c0c0;margin-top:0.5rem;">
                No es una elección — es el resultado de condiciones estructurales concretas.
            </div></div>''',
            unsafe_allow_html=True,
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Trabajadores en muestra", f"{n_total:,}")
        k2.metric("Informales", f"{n_informal:,}",
                  f"{n_informal/n_total*100:.1f}% del total", delta_color="off")
        k3.metric("Formales",   f"{n_total-n_informal:,}",
                  f"{(n_total-n_informal)/n_total*100:.1f}% del total", delta_color="off")

        def _semaforo(pct):
            if pct >= tasa_global + 10: return "#C0392B"
            elif pct >= tasa_global:    return "#E67E22"
            elif pct >= tasa_global - 15: return "#F1C40F"
            else:                       return "#27AE60"

        # ═══════════════════════════════════════════════════════════
        # ② EDUCACIÓN  +  ③ ZONA (como KPIs)
        # ═══════════════════════════════════════════════════════════
        st.divider()
        col_edu, col_zona = st.columns([3, 2])

        with col_edu:
            tasa_edu = (
                df[df["P3042"].notna()]
                .groupby("P3042", observed=True)
                .apply(tasa_pond_grp, include_groups=False)
                .reset_index().rename(columns={0: "tasa"}).sort_values("P3042")
            )
            tasa_edu["etiqueta"] = tasa_edu["P3042"].map(LABELS_EDU)
            tasa_edu["pct"]      = tasa_edu["tasa"].round(1)
            _pct_ningu = float(tasa_edu.iloc[0]["pct"])
            _univ_v    = tasa_edu.loc[tasa_edu["P3042"]==10, "pct"].values
            _pct_univ  = float(_univ_v[0]) if len(_univ_v) else float(tasa_edu.iloc[-1]["pct"])
            st.markdown(
                f"### Sin educación: {_pct_ningu:.0f}% informal. "
                f"Con universidad: {_pct_univ:.0f}% — "
                f"{_pct_ningu - _pct_univ:.0f} pp de diferencia"
            )
            st.caption(
                "🔴 Rojo = >10 pp sobre el promedio · "
                "🟠 Naranja = sobre el promedio · "
                "🟡 Amarillo = cerca del promedio · "
                "🟢 Verde = bajo riesgo"
            )

            edu_texts = [f"<b>{v:.0f}%</b>" for v in tasa_edu["pct"]]
            fig_edu = go.Figure(go.Bar(
                x=tasa_edu["etiqueta"], y=tasa_edu["pct"],
                marker_color=[_semaforo(v) for v in tasa_edu["pct"]],
                text=edu_texts, textposition="outside",
                textfont=dict(size=14, family="Arial Black, Arial Bold, Arial", color="#111111"),
            ))
            fig_edu.add_hline(y=tasa_global, line_dash="dot",
                              line_color="#888888", line_width=1.5,
                              annotation_text=f"  promedio: {tasa_global:.0f}%",
                              annotation_position="top left",
                              annotation_font=dict(size=11, color="#666666"),
                              annotation_bgcolor="white")
            fig_edu.update_layout(
                xaxis=dict(
                    tickangle=-45,
                    tickfont=dict(size=11, family="Arial", color="#111111"),
                    showgrid=False,
                ),
                yaxis=dict(
                    title=dict(
                        text="% de trabajadores informales<br>según nivel educativo",
                        font=dict(size=11, color="#555555"),
                    ),
                    range=[0, 122], ticksuffix="%",
                    showticklabels=False,
                    gridcolor="#F4F4F4", gridwidth=1, zeroline=False,
                    tickfont=dict(size=12, color="#777777"),
                ),
                plot_bgcolor="white", paper_bgcolor="white",
                height=410, showlegend=False, margin=dict(t=25, b=90, l=55, r=15),
            )
            st.plotly_chart(fig_edu, use_container_width=True, key="dash_edu")

            st.markdown(
                f"> **Hallazgo:** Sin educación: **{_pct_ningu:.0f}%** informal. "
                f"Con universidad: **{_pct_univ:.0f}%** — "
                f"**{_pct_ningu - _pct_univ:.0f} pp** de diferencia."
            )

        with col_zona:
            t_zona = (
                df[df["CLASE"].notna()]
                .groupby("CLASE", observed=True)
                .apply(tasa_pond_grp, include_groups=False)
                .reset_index().rename(columns={0: "tasa"})
            )
            t_zona["pct"] = t_zona["tasa"].round(1)
            pct_cab = float(t_zona.loc[t_zona["CLASE"]==1, "pct"].values[0])
            pct_rur = float(t_zona.loc[t_zona["CLASE"]==2, "pct"].values[0])
            diferencia = pct_rur - pct_cab
            st.markdown(
                f"### Rural: {pct_rur:.0f}% informal. "
                f"Cabecera: {pct_cab:.0f}% — {diferencia:.0f} pp de diferencia"
            )
            st.caption("La zona geográfica amplifica todos los demás factores de riesgo.")

            z1, z2 = st.columns(2)
            z1.metric("Cabecera Municipal", f"{pct_cab:.1f}%")
            z2.metric("Rural", f"{pct_rur:.1f}%",
                      delta=f"+{diferencia:.0f} pp", delta_color="inverse")

            fig_zona = go.Figure(go.Bar(
                x=["Cabecera", "Rural"],
                y=[pct_cab, pct_rur],
                marker_color=[_semaforo(pct_cab), _semaforo(pct_rur)],
                text=[f"<b>{pct_cab:.1f}%</b>", f"<b>{pct_rur:.1f}%</b>"],
                textposition="outside",
                textfont=dict(size=18, family="Arial Black", color="#111111"),
                width=[0.45, 0.45],
            ))
            fig_zona.add_hline(
                y=tasa_global, line_dash="dot", line_color="#888888", line_width=1.5,
                annotation_text=f"  promedio: {tasa_global:.0f}%",
                annotation_position="top left",
                annotation_font=dict(size=11, color="#666666"),
                annotation_bgcolor="white",
            )
            fig_zona.update_layout(
                yaxis=dict(range=[0, pct_rur + 20], ticksuffix="%",
                           showticklabels=False,
                           gridcolor="#F4F4F4", zeroline=False,
                           tickfont=dict(size=12, color="#777777")),
                xaxis=dict(showgrid=False, tickfont=dict(size=14, color="#111111")),
                plot_bgcolor="white", paper_bgcolor="white",
                height=260, showlegend=False,
                margin=dict(t=15, b=20, l=55, r=15),
            )
            st.plotly_chart(fig_zona, use_container_width=True, key="dash_zona")
            st.markdown(
                f"> **Hallazgo:** Vivir en zona rural incrementa la informalidad "
                f"en **{diferencia:.0f} puntos porcentuales**, independientemente "
                f"del sector y el nivel educativo."
            )

        # ═══════════════════════════════════════════════════════════
        # ④ SECTOR ECONÓMICO
        # ═══════════════════════════════════════════════════════════
        st.divider()
        st.markdown("### En agricultura y hogares, ser informal es la norma — no la excepción")
        st.caption(
            "Solo los 8 sectores con mayor presencia de independientes. "
            "Los tres más informales resaltados en negro. "
            "La línea vertical es el promedio nacional."
        )

        LABELS_RAMA_CORTO = {
            1:  "Agricultura / Pesca",       5:  "Minería",
            10: "Manufactura",               14: "Confección / Textil",
            25: "Productos metálicos",       36: "Agua y servicios",
            41: "Construcción",              42: "Obras civiles",
            43: "Constr. especializada",     45: "Comercio vehículos",
            46: "Comercio al por mayor",     47: "Comercio al por menor",
            49: "Transporte",                52: "Almacenamiento",
            55: "Alojamiento",               56: "Restaurantes / Comidas",
            58: "Comunicaciones",            64: "Finanzas",
            68: "Inmobiliario",              69: "Jurídico / Contabilidad",
            75: "Veterinaria",               78: "Servicios adm.",
            81: "Servicios a edificios",     82: "Servicios de oficina",
            84: "Adm. pública",              85: "Educación",
            86: "Salud",                     90: "Artes / Entretenimiento",
            95: "Reparación equipos",        96: "Otros serv. personales",
            97: "Hogares c/serv. dom.",      99: "Otro / NS",
        }
        top_ramas = df["RAMA2D_R4"].value_counts().head(8).index
        tasa_rama = (
            df[df["RAMA2D_R4"].isin(top_ramas)]
            .groupby("RAMA2D_R4", observed=True)
            .apply(tasa_pond_grp, include_groups=False)
            .reset_index().rename(columns={0: "tasa"}).sort_values("tasa")
        )
        tasa_rama["etiqueta"] = tasa_rama["RAMA2D_R4"].map(LABELS_RAMA_CORTO).fillna(
            "Sector " + tasa_rama["RAMA2D_R4"].astype(str)
        )
        tasa_rama["pct"] = tasa_rama["tasa"].round(1)
        n_r = len(tasa_rama)

        rama_cols  = [_semaforo(v) for v in tasa_rama["pct"]]
        rama_texts = [
            f"<b>{v:.0f}%</b>" if _semaforo(v) in ("#C0392B", "#E67E22") else ""
            for v in tasa_rama["pct"]
        ]

        fig_rama = go.Figure(go.Bar(
            x=tasa_rama["pct"], y=tasa_rama["etiqueta"],
            orientation="h",
            marker_color=rama_cols,
            text=rama_texts, textposition="outside",
            textfont=dict(size=18, family="Arial Black, Arial Bold, Arial", color="#111111"),
        ))
        fig_rama.add_vline(x=tasa_global, line_dash="dot",
                           line_color="#555555", line_width=2,
                           annotation_text=f"promedio: {tasa_global:.0f}%",
                           annotation_position="top right",
                           annotation_font=dict(size=12, color="#555555"),
                           annotation_bgcolor="white")
        fig_rama.update_layout(
            xaxis=dict(
                range=[0, 122], ticksuffix="%",
                showgrid=False, zeroline=False,
                tickfont=dict(size=12, color="#777777"),
            ),
            yaxis=dict(
                tickfont=dict(size=15, family="Arial", color="#111111"),
                showgrid=False,
            ),
            plot_bgcolor="white", paper_bgcolor="white",
            height=420, showlegend=False,
            margin=dict(t=20, b=20, l=220, r=95),
        )
        st.plotly_chart(fig_rama, use_container_width=True, key="dash_rama")

        top_sector = tasa_rama.iloc[-1]
        st.markdown(
            f"> **Hallazgo:** **{top_sector['etiqueta']}** tiene la tasa más alta con "
            f"**{top_sector['pct']:.0f}%** de informalidad. "
            f"Los tres sectores críticos superan el promedio nacional en más de 5 puntos."
        )

        # ═══════════════════════════════════════════════════════════
        # ⑤ EDAD — curva en U
        # ═══════════════════════════════════════════════════════════
        st.divider()
        _bins_e   = [14, 24, 34, 44, 54, 64, 120]
        _labs_e   = ["15–24", "25–34", "35–44", "45–54", "55–64", "65+"]
        df_edad   = df[df["P6040"].notna()].copy()
        df_edad["EDAD_GRUPO"] = pd.cut(df_edad["P6040"], bins=_bins_e, labels=_labs_e)
        tasa_edad = (
            df_edad[df_edad["EDAD_GRUPO"].notna()]
            .groupby("EDAD_GRUPO", observed=True)
            .apply(tasa_pond_grp, include_groups=False)
            .reset_index().rename(columns={0: "tasa"})
        )
        tasa_edad["pct"]   = tasa_edad["tasa"].round(1)
        tasa_edad["grupo"] = tasa_edad["EDAD_GRUPO"].astype(str)
        _min_grupo = tasa_edad.loc[tasa_edad["pct"].idxmin(), "grupo"]
        _max_grupo = tasa_edad.loc[tasa_edad["pct"].idxmax(), "grupo"]
        _min_pct   = tasa_edad["pct"].min()
        _max_pct   = tasa_edad["pct"].max()
        st.markdown(
            f"### {_max_grupo} años: {_max_pct:.0f}% informal. "
            f"Menor riesgo en {_min_grupo} años con {_min_pct:.0f}%"
        )
        st.caption(
            "Tasa de informalidad ponderada por grupo de edad. "
            "El riesgo es mínimo en el grupo 25–34 y sube en los extremos — "
            "los jóvenes por falta de experiencia y los mayores por desprotección estructural."
        )

        _marker_colors_edad = [_semaforo(v) for v in tasa_edad["pct"]]

        fig_edad = go.Figure()
        fig_edad.add_trace(go.Scatter(
            x=tasa_edad["grupo"], y=tasa_edad["pct"],
            mode="lines+markers+text",
            line=dict(color="#888888", width=2.5),
            marker=dict(
                size=16, color=_marker_colors_edad,
                line=dict(color="white", width=2),
            ),
            text=[f"<b>{v:.0f}%</b>" for v in tasa_edad["pct"]],
            textposition="top center",
            textfont=dict(size=13, family="Arial Black", color="#111111"),
        ))
        fig_edad.add_hline(
            y=tasa_global, line_dash="dot", line_color="#888888", line_width=1.5,
            annotation_text=f"  promedio: {tasa_global:.0f}%",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#666666"),
            annotation_bgcolor="white",
        )
        fig_edad.update_layout(
            xaxis=dict(
                title="<b>Grupo de edad</b>",
                showgrid=False,
                tickfont=dict(size=13, color="#111111"),
            ),
            yaxis=dict(
                range=[0, max(tasa_edad["pct"]) + 18],
                ticksuffix="%",
                showticklabels=False,
                gridcolor="#F4F4F4", zeroline=False,
                tickfont=dict(size=12, color="#777777"),
            ),
            plot_bgcolor="white", paper_bgcolor="white",
            height=340, showlegend=False,
            margin=dict(t=30, b=50, l=55, r=20),
        )
        st.plotly_chart(fig_edad, use_container_width=True, key="dash_edad")

        st.markdown(
            f"> **Hallazgo:** El grupo de menor riesgo es **{_min_grupo}** ({_min_pct:.0f}%), "
            f"mientras que **{_max_grupo}** alcanza **{_max_pct:.0f}%** — "
            f"una diferencia de **{_max_pct - _min_pct:.0f} pp**. "
            "Los programas del SENA deberían priorizar los extremos de la curva."
        )

        # ═══════════════════════════════════════════════════════════
        # ⑥ CONCLUSIÓN — Estructura del riesgo
        # ═══════════════════════════════════════════════════════════
        st.divider()
        st.markdown("### El riesgo de informalidad está determinado por condiciones estructurales")
        st.markdown(
            "> La educación, la zona geográfica y el sector económico explican sistemáticamente "
            "quién es informal y quién no. La informalidad entre trabajadores independientes "
            "**no es una elección**: es el resultado predecible de las condiciones en que se trabaja. "
            "Intervenir sobre esas condiciones, en lugar de diseñar programas universales, "
            "aumentaría significativamente la efectividad de los programas de formalización."
        )


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — Informalidad por departamento
# ════════════════════════════════════════════════════════════════════════
with tab_dpto:
    st.markdown(
        "### Los departamentos del Caribe y la Amazonia concentran "
        "la informalidad entre trabajadores independientes"
    )
    st.caption(
        "Tasa ponderada por FEX_C18 · GEIH 2025 · Trabajadores por cuenta propia (P6430 = 4). "
        "**Rojo** = sobre el promedio · **Amarillo** = cerca del promedio · **Azul** = bajo el promedio. "
        "Borde negro = Top 5 más y menos informales."
    )

    if df is None:
        st.info("Ejecuta los notebooks para cargar datos reales.")
    else:
        zona_sel = st.radio(
            "Filtrar por zona", ["Todas", "Cabecera municipal", "Rural"],
            horizontal=True,
        )
        df_d = df.copy()
        if zona_sel == "Cabecera municipal":
            df_d = df_d[df_d["CLASE"] == 1]
        elif zona_sel == "Rural":
            df_d = df_d[df_d["CLASE"] == 2]

        tasa_dpto = (
            df_d[df_d["DPTO"].notna()]
            .groupby("DPTO", observed=True)
            .apply(tasa_pond_grp, include_groups=False)
            .reset_index().rename(columns={0: "tasa"})
        )
        tasa_dpto["DPTO"]   = tasa_dpto["DPTO"].astype(int)
        tasa_dpto["nombre"] = tasa_dpto["DPTO"].map(
            lambda d: DPTO_INFO.get(d, (str(d), 0, 0))[0]
        )
        tasa_dpto["pct"] = tasa_dpto["tasa"].round(1)
        tasa_dpto        = tasa_dpto.sort_values("pct", ascending=True).reset_index(drop=True)

        tasa_nac = (
            df_d["INFORMAL"] * df_d["FEX_C18"]
        ).sum() / df_d["FEX_C18"].sum() * 100

        n = len(tasa_dpto)
        TOP5_ALTO = 5   # los más informales (últimas filas → arriba del chart)
        TOP5_BAJO = 5   # los menos informales (primeras filas → abajo del chart)

        # ── 3 colores según distancia al promedio ─────────────────────────
        ROJO     = "#C0392B"   # sobre promedio
        AMARILLO = "#E8A838"   # cerca del promedio
        AZUL     = "#2980B9"   # bajo promedio

        def _col_dpto(pct):
            if pct >= tasa_nac + 4:   return ROJO
            elif pct >= tasa_nac - 4: return AMARILLO
            else:                     return AZUL

        colores = [_col_dpto(v) for v in tasa_dpto["pct"]]

        # ── Bordes: Top 5 alto (últimos N) y Top 5 bajo (primeros N) ──────
        borde_color = [
            "#111111" if i >= n - TOP5_ALTO or i < TOP5_BAJO else "rgba(0,0,0,0)"
            for i in range(n)
        ]
        borde_ancho = [
            2.5 if i >= n - TOP5_ALTO or i < TOP5_BAJO else 0
            for i in range(n)
        ]

        # ── Etiquetas: negrita en Top/Bottom 5, normal en el resto ────────
        labels = [
            f"<b>{v:.1f}%</b>" if i >= n - TOP5_ALTO or i < TOP5_BAJO else f"{v:.0f}%"
            for i, v in enumerate(tasa_dpto["pct"])
        ]

        fig_d = go.Figure(go.Bar(
            x=tasa_dpto["pct"],
            y=tasa_dpto["nombre"],
            orientation="h",
            marker=dict(
                color=colores,
                line=dict(color=borde_color, width=borde_ancho),
            ),
            text=labels,
            textposition="outside",
            textfont=dict(size=12, family="Arial Bold, Arial", color="#111111"),
            hovertemplate="<b>%{y}</b><br>Informalidad: <b>%{x:.1f}%</b><extra></extra>",
            cliponaxis=False,
        ))

        # ── Línea de promedio — prominente ────────────────────────────────
        fig_d.add_vline(
            x=tasa_nac,
            line_dash="solid", line_color="#333333", line_width=2.5,
        )
        fig_d.add_annotation(
            x=tasa_nac, y=1.01, xref="x", yref="paper",
            text=f"<b>Promedio nacional<br>{tasa_nac:.1f}%</b>",
            showarrow=False, xanchor="center",
            font=dict(size=13, color="#111111"),
            bgcolor="white",
            bordercolor="#333333", borderwidth=1.5, borderpad=5,
        )

        # ── Etiquetas de grupo Top/Bottom ──────────────────────────────────
        fig_d.add_annotation(
            x=tasa_dpto.iloc[-1]["pct"] + 1,
            y=tasa_dpto.iloc[-3]["nombre"],
            text="<b>⬆ 5 más<br>informales</b>",
            showarrow=False, xanchor="left",
            font=dict(size=11, color=ROJO),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=ROJO, borderwidth=1.5, borderpad=4,
        )
        fig_d.add_annotation(
            x=tasa_dpto.iloc[0]["pct"] + 1,
            y=tasa_dpto.iloc[2]["nombre"],
            text="<b>⬇ 5 menos<br>informales</b>",
            showarrow=False, xanchor="left",
            font=dict(size=11, color=AZUL),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=AZUL, borderwidth=1.5, borderpad=4,
        )

        fig_d.update_layout(
            xaxis=dict(
                title="<b>Tasa de informalidad (%)</b>",
                range=[0, 122],
                ticksuffix="%",
                gridcolor="#F0F0F0", gridwidth=1,
                zeroline=False,
                tickfont=dict(size=12, color="#555555"),
            ),
            yaxis=dict(
                tickfont=dict(size=13, family="Arial", color="#111111"),
                showgrid=False,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=1000,
            margin=dict(l=160, r=140, t=60, b=30),
            font=dict(family="Arial, sans-serif"),
            bargap=0.22,
        )
        st.plotly_chart(fig_d, use_container_width=True, key="dpto_bar")

        # ── Lectura narrativa ──────────────────────────────────────────────
        top5    = tasa_dpto.tail(5)["nombre"].tolist()
        bottom5 = tasa_dpto.head(5)["nombre"].tolist()
        st.markdown(
            f"> **Lectura:** Los departamentos con mayor informalidad entre independientes son "
            f"**{', '.join(top5[:-1])} y {top5[-1]}**, todos superando el promedio nacional. "
            f"Los de menor informalidad son **{', '.join(bottom5[:-1])} y {bottom5[-1]}**, "
            f"posiblemente por mayor densidad urbana y acceso a programas de formalización."
        )

        with st.expander("Ver tablas de ranking completo"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Top 10 — Mayor informalidad**")
                st.dataframe(
                    tasa_dpto[["nombre", "pct"]]
                        .sort_values("pct", ascending=False).head(10)
                        .rename(columns={"nombre": "Departamento", "pct": "% Informal"})
                        .reset_index(drop=True),
                    use_container_width=True, height=310,
                )
            with d2:
                st.markdown("**Top 10 — Menor informalidad**")
                st.dataframe(
                    tasa_dpto[["nombre", "pct"]]
                        .sort_values("pct", ascending=True).head(10)
                        .rename(columns={"nombre": "Departamento", "pct": "% Informal"})
                        .reset_index(drop=True),
                    use_container_width=True, height=310,
                )

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
        st.markdown("#### Importancia de variables del modelo (gain)")
        if model is not None and meta is not None:
            # Mapa completo de nombres legibles para todas las features OHE
            _NOM = {
                "P6040": "Edad",
                "ANIOS_EDU": "Años de educación",
                "P6800": "Horas / semana",
                "P3271": "Sexo",
                "CLASE": "Zona (urbana/rural)",
                "MICROEMPRESA": "Microempresa (≤10 p.)",
                "SUBEMPLEADO": "Subempleado (<32 h)",
                "PLURIEMPLEO": "Pluriempleo",
                "0": "Rama de actividad",
                "1": "Departamento",
                "EDAD_GRUPO_15-24": "Edad: 15–24",
                "EDAD_GRUPO_25-34": "Edad: 25–34",
                "EDAD_GRUPO_35-44": "Edad: 35–44",
                "EDAD_GRUPO_45-54": "Edad: 45–54",
                "EDAD_GRUPO_55-64": "Edad: 55–64",
                "EDAD_GRUPO_65+":   "Edad: 65+",
                "P6070_1.0": "Civil: No unido/a",
                "P6070_2.0": "Civil: Unión libre",
                "P6070_3.0": "Civil: Casado/a",
                "P6070_4.0": "Civil: Separado/a",
                "P6070_5.0": "Civil: Viudo/a",
                "P6070_6.0": "Civil: NS/NR",
                "P6450_1.0": "Contrato: Verbal",
                "P6450_2.0": "Contrato: Escrito",
                "P6450_9.0": "Contrato: NS/NR",
                "P3069_1":  "Establ. 1 persona",
                "P3069_2":  "Establ. 2–5 p.",
                "P3069_3":  "Establ. 6–10 p.",
                "P3069_4":  "Establ. 11–19 p.",
                "P3069_5":  "Establ. 20–30 p.",
                "P3069_6":  "Establ. 31–50 p.",
                "P3069_7":  "Establ. 51–100 p.",
                "P3069_8":  "Establ. 101–200 p.",
                "P3069_9":  "Establ. 201+ p.",
                "P3069_10": "Establ. tamaño NS",
            }
            feat_raw   = meta.get("feature_names", [])
            feat_label = [_NOM.get(n, n) for n in feat_raw]
            importances = model.feature_importances_

            df_imp = (
                pd.DataFrame({"Variable": feat_label, "Importancia": importances})
                .sort_values("Importancia")
                .tail(15)
            )
            fig_imp = go.Figure(go.Bar(
                x=df_imp["Importancia"],
                y=df_imp["Variable"],
                orientation="h",
                marker_color="#2196F3",
                text=df_imp["Importancia"].round(0).astype(int).astype(str),
                textposition="outside",
                textfont=dict(size=12, color="#111111"),
            ))
            fig_imp.update_layout(
                xaxis=dict(title="Importancia (gain)", showgrid=False, zeroline=False,
                           tickfont=dict(size=11, color="#555555")),
                yaxis=dict(tickfont=dict(size=13, family="Arial", color="#111111"),
                           showgrid=False),
                plot_bgcolor="white", paper_bgcolor="white",
                height=480, showlegend=False,
                margin=dict(l=185, r=60, t=20, b=30),
            )
            st.plotly_chart(fig_imp, use_container_width=True, key="shap_imp_live")
        elif IMG_SHAP_BAR.exists():
            st.image(str(IMG_SHAP_BAR), use_container_width=True)

        st.markdown("#### SHAP beeswarm — distribución de impactos por variable")
        _NOMB = {
            "P6040": "Edad", "ANIOS_EDU": "Años de educación",
            "P6800": "Horas / semana", "P3271": "Sexo",
            "CLASE": "Zona (urbana/rural)", "MICROEMPRESA": "Microempresa (≤10 p.)",
            "SUBEMPLEADO": "Subempleado (<32 h)", "PLURIEMPLEO": "Pluriempleo",
            "0": "Rama de actividad", "1": "Departamento",
            "EDAD_GRUPO_15-24": "Edad: 15–24", "EDAD_GRUPO_25-34": "Edad: 25–34",
            "EDAD_GRUPO_35-44": "Edad: 35–44", "EDAD_GRUPO_45-54": "Edad: 45–54",
            "EDAD_GRUPO_55-64": "Edad: 55–64", "EDAD_GRUPO_65+": "Edad: 65+",
            "P6070_1.0": "Civil: No unido/a", "P6070_2.0": "Civil: Unión libre",
            "P6070_3.0": "Civil: Casado/a",   "P6070_4.0": "Civil: Separado/a",
            "P6070_5.0": "Civil: Viudo/a",     "P6070_6.0": "Civil: NS/NR",
            "P6450_1.0": "Contrato: Verbal",   "P6450_2.0": "Contrato: Escrito",
            "P6450_9.0": "Contrato: NS/NR",
            "P3069_1": "Establ. 1 persona",    "P3069_2":  "Establ. 2–5 p.",
            "P3069_3": "Establ. 6–10 p.",      "P3069_4":  "Establ. 11–19 p.",
            "P3069_5": "Establ. 20–30 p.",     "P3069_6":  "Establ. 31–50 p.",
            "P3069_7": "Establ. 51–100 p.",    "P3069_8":  "Establ. 101–200 p.",
            "P3069_9": "Establ. 201+ p.",      "P3069_10": "Establ. tamaño NS",
        }
        _bee = compute_shap_beeswarm()
        if _bee is not None:
            _sv   = np.array(_bee["sv"])
            _X    = np.array(_bee["X"])
            _cols = _bee["cols"]
            _labels = [_NOMB.get(c, c) for c in _cols]

            # Top 15 por |SHAP| medio
            _mean_abs = np.abs(_sv).mean(axis=0)
            _top_idx  = np.argsort(_mean_abs)[-15:]

            _sorted_labels = [_labels[i] for i in _top_idx]
            _pos = {idx: pos for pos, idx in enumerate(_top_idx)}
            _rng = np.random.default_rng(42)
            _traces = []
            for _i in _top_idx:
                _sv_i   = _sv[:, _i]
                _fv_i   = _X[:, _i]
                _fmin, _fmax = _fv_i.min(), _fv_i.max()
                _fnorm  = (_fv_i - _fmin) / (_fmax - _fmin + 1e-9)
                _colors = [
                    f"rgb({int(220*v+35)},{int(40*(1-v))},{int(220*(1-v)+35)})"
                    for v in _fnorm
                ]
                _jit   = _rng.uniform(-0.28, 0.28, len(_sv_i))
                _y_num = float(_pos[_i]) + _jit
                _traces.append(go.Scatter(
                    x=_sv_i,
                    y=_y_num,
                    mode="markers",
                    marker=dict(color=_colors, size=3.5, opacity=0.45),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{_labels[_i]}</b><br>"
                        "SHAP: %{x:.3f}<extra></extra>"
                    ),
                ))

            _fig_bee = go.Figure(_traces)
            _fig_bee.add_vline(x=0, line_color="#333333", line_width=1.2)
            _fig_bee.update_layout(
                xaxis=dict(
                    title="<b>Valor SHAP</b>  (→ aumenta probabilidad informal · ← disminuye)",
                    zeroline=False, showgrid=True, gridcolor="#F0F0F0",
                    tickfont=dict(size=11, color="#555555"),
                ),
                yaxis=dict(
                    tickvals=list(range(len(_sorted_labels))),
                    ticktext=_sorted_labels,
                    tickfont=dict(size=13, family="Arial", color="#111111"),
                    showgrid=False,
                    range=[-0.6, len(_sorted_labels) - 0.4],
                ),
                plot_bgcolor="white", paper_bgcolor="white",
                height=500, showlegend=False,
                margin=dict(l=185, r=20, t=15, b=55),
            )
            st.plotly_chart(_fig_bee, use_container_width=True, key="shap_bee_live")
            st.caption(
                "Cada punto = un trabajador de la muestra. "
                "Color: **rojo** = valor alto de la variable · **azul** = valor bajo."
            )
        elif IMG_SHAP_BEE.exists():
            st.image(str(IMG_SHAP_BEE), use_container_width=True)

    if not any([IMG_CONFUSION.exists(), IMG_COMP.exists(),
                IMG_SHAP_BAR.exists(), IMG_SHAP_BEE.exists()]):
        st.info("Ejecuta `4.Modelamiento.ipynb` y `5.Visualizacion.ipynb` para generar las imágenes.")

    st.divider()
    st.markdown("""
    **Interpretación del modelo campeón — Trabajadores independientes (LightGBM):**

    | Variable | Dirección del impacto | Importancia SHAP (referencial) |
    |---|---|---|
    | Años de educación | ↓ Reduce informalidad | **#1** |
    | Departamento (enc.) | Varía por región | **#2** |
    | Rama de actividad (enc.) | Varía por sector | **#3** |
    | Edad | Curva no lineal (jóvenes y mayores más vulnerables) | **#4** |
    | Horas / semana | ↓ Más horas = menor riesgo | **#5** |
    | Zona (urbano/rural) | ↑ Rural = mayor riesgo | **#6** |
    | Estado civil | Efecto social / responsabilidad familiar | **#7** |

    > Los valores SHAP exactos se actualizan al reentrenar el modelo con la nueva muestra de trabajadores independientes.
    """)

# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — Predicción Individual
# ══════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("Estima la probabilidad de informalidad de un trabajador independiente")
    st.info(
        "**Modelo entrenado exclusivamente sobre trabajadores independientes (Cuenta propia · P6430 = 4).** "
        "Ingresa las características del trabajador para obtener su score de riesgo de informalidad "
        "según LightGBM entrenado sobre GEIH 2025."
    )

    if model is None or preprocessor is None:
        st.error("Carga el modelo ejecutando `4.Modelamiento.ipynb` primero.")
    else:
        col_l, col_r, col_ref = st.columns([4, 4, 5])

        with col_l:
            st.markdown("**Datos demográficos**")
            edad      = st.slider("Edad (años)", 15, 75, 35)
            anios_edu = st.slider("Años de educación acumulados", 0, 25, 9)
            # Labels de referencia debajo del slider
            st.markdown(
                """<div style="display:flex;justify-content:space-between;
                            font-size:0.72rem;color:#888;margin-top:-14px;
                            margin-bottom:6px;padding:0 2px">
                    <span>Ninguno<br><b style='color:#555'>0</b></span>
                    <span style="text-align:center">Primaria<br><b style='color:#555'>5</b></span>
                    <span style="text-align:center">Bachillerato<br><b style='color:#555'>11</b></span>
                    <span style="text-align:center">Técnica<br><b style='color:#555'>14</b></span>
                    <span style="text-align:center">Univer.<br><b style='color:#555'>16</b></span>
                    <span style="text-align:right">Doctorado<br><b style='color:#555'>21</b></span>
                </div>""",
                unsafe_allow_html=True,
            )
            # Nivel dinámico según valor actual
            _edu_ref = [(0,"Ninguno"),(1,"Preescolar"),(3,"Primaria inc."),
                        (5,"Primaria"),(7,"Sec. incompleta"),(9,"Secundaria"),
                        (10,"Media inc."),(11,"Bachillerato"),(14,"Técnica/Tecnológica"),
                        (16,"Universidad"),(17,"Especialización"),(18,"Maestría"),(21,"Doctorado")]
            _nivel_edu = next(n for u, n in reversed(_edu_ref) if anios_edu >= u)
            st.caption(f"Nivel: **{_nivel_edu}**")
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
            p6800     = st.slider("Horas trabajadas / semana", 1, 100, 40)
            rama_lbl  = st.selectbox("Rama de actividad (CIIU)", list(RAMA_CIIU.values()))
            rama      = {v: k for k, v in RAMA_CIIU.items()}[rama_lbl]
            pluriemp  = st.checkbox("¿Tiene otro trabajo adicional (pluriempleo)?", value=False)
            p7040_val = 1 if pluriemp else 0

        with col_ref:
            if df is not None:
                _ref_semaforo = lambda v, ref: (
                    "#C0392B" if v >= ref + 10 else
                    "#E67E22" if v >= ref else
                    "#F1C40F" if v >= ref - 15 else "#27AE60"
                )
                _tasa_ref = (df["INFORMAL"] * df["FEX_C18"]).sum() / df["FEX_C18"].sum() * 100

                # ── Gráfica 1: % informalidad según educación ────────────
                st.markdown("**% informalidad según educación**")
                _lbl_edu = {
                    1:"Ninguno", 2:"Preescolar", 3:"Primaria inc.", 4:"Primaria",
                    5:"Sec. inc.", 6:"Secundaria", 7:"Media inc.", 8:"Media",
                    9:"Técnica", 10:"Universidad", 11:"Especializ.",
                    12:"Maestría", 13:"Doctorado",
                }
                if "P3042" in df.columns:
                    _te = (
                        df[df["P3042"].notna()]
                        .groupby("P3042", observed=True)
                        .apply(tasa_pond_grp, include_groups=False)
                        .reset_index().rename(columns={0: "tasa"})
                    )
                    _te["etiq"] = _te["P3042"].map(_lbl_edu)
                    _te["pct"]  = _te["tasa"].round(1)
                    _te = _te[_te["P3042"] <= 13].copy()

                    _fig_edu_ref = go.Figure(go.Bar(
                        y=_te["etiq"], x=_te["pct"],
                        orientation="h",
                        marker_color=[_ref_semaforo(v, _tasa_ref) for v in _te["pct"]],
                        text=[f"<b>{v:.0f}%</b>" for v in _te["pct"]],
                        textposition="outside",
                        textfont=dict(size=10, color="#111111"),
                    ))
                    _fig_edu_ref.add_vline(
                        x=_tasa_ref, line_dash="dot", line_color="#888",
                        line_width=1.5,
                    )
                    _fig_edu_ref.update_layout(
                        xaxis=dict(range=[0, 120], ticksuffix="%",
                                   showgrid=False, zeroline=False,
                                   tickfont=dict(size=9, color="#777")),
                        yaxis=dict(tickfont=dict(size=10, color="#111"),
                                   showgrid=False),
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=320, showlegend=False,
                        margin=dict(t=5, b=10, l=5, r=50),
                    )
                    st.plotly_chart(_fig_edu_ref, use_container_width=True,
                                    key="ref_edu_pred")

                # ── Gráfica 2: % informalidad según demografía ───────────
                st.markdown("**% informalidad según demografía**")
                _demog_labels, _demog_vals = [], []

                if "CLASE" in df.columns:
                    for _c, _n in [(1, "Cabecera"), (2, "Rural")]:
                        _sub = df[df["CLASE"] == _c]
                        if len(_sub):
                            _demog_labels.append(_n)
                            _demog_vals.append(
                                (_sub["INFORMAL"]*_sub["FEX_C18"]).sum()
                                / _sub["FEX_C18"].sum() * 100
                            )

                if "P3271" in df.columns:
                    for _s, _n in [(1, "Hombre"), (2, "Mujer")]:
                        _sub = df[df["P3271"] == _s]
                        if len(_sub):
                            _demog_labels.append(_n)
                            _demog_vals.append(
                                (_sub["INFORMAL"]*_sub["FEX_C18"]).sum()
                                / _sub["FEX_C18"].sum() * 100
                            )

                if "P6040" in df.columns:
                    _df_e = df[df["P6040"].notna()].copy()
                    _df_e["EG"] = pd.cut(_df_e["P6040"],
                                         bins=[14,24,34,44,54,64,120],
                                         labels=["15-24","25-34","35-44",
                                                 "45-54","55-64","65+"])
                    for _g in ["15-24","25-34","35-44","45-54","55-64","65+"]:
                        _sub = _df_e[_df_e["EG"].astype(str) == _g]
                        if len(_sub):
                            _demog_labels.append(_g)
                            _demog_vals.append(
                                (_sub["INFORMAL"]*_sub["FEX_C18"]).sum()
                                / _sub["FEX_C18"].sum() * 100
                            )

                if _demog_labels:
                    _fig_dem = go.Figure(go.Bar(
                        y=_demog_labels, x=[round(v, 1) for v in _demog_vals],
                        orientation="h",
                        marker_color=[_ref_semaforo(v, _tasa_ref) for v in _demog_vals],
                        text=[f"<b>{v:.0f}%</b>" for v in _demog_vals],
                        textposition="outside",
                        textfont=dict(size=10, color="#111111"),
                    ))
                    _fig_dem.add_vline(
                        x=_tasa_ref, line_dash="dot", line_color="#888",
                        line_width=1.5,
                    )
                    _fig_dem.update_layout(
                        xaxis=dict(range=[0, 120], ticksuffix="%",
                                   showgrid=False, zeroline=False,
                                   tickfont=dict(size=9, color="#777")),
                        yaxis=dict(tickfont=dict(size=10, color="#111"),
                                   showgrid=False),
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=280, showlegend=False,
                        margin=dict(t=5, b=10, l=5, r=50),
                    )
                    st.plotly_chart(_fig_dem, use_container_width=True,
                                    key="ref_dem_pred")

        # Trabajadores independientes (cuenta propia) se consideran empleados únicos
        p3069 = 1

        input_raw = construir_input(
            p6040=edad,  p3271=p3271,  p6070=p6070,  clase=clase,  dpto=dpto,
            p6800=p6800, p3069=p3069,  rama=rama,
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

            # ── Diagnóstico ────────────────────────────────────────────
            if pred == 1:
                st.warning(
                    f"**⚠️ Perfil INFORMAL — Probabilidad {prob:.1%}.** "
                    "Este trabajador debería ser priorizado en programas del SENA, "
                    "inspecciones del Ministerio del Trabajo o subsidios de formalización."
                )

                # ── Recomendaciones IA ─────────────────────────────────
                st.markdown("---")
                st.markdown("#### 🤖 Recomendaciones de política pública generadas por IA")

                if not anthropic_api_key:
                    st.info(
                        "Ingresa tu API Key de Claude en el panel lateral izquierdo "
                        "para generar recomendaciones personalizadas con IA."
                    )
                else:
                    perfil_legible = {
                        "Edad":              f"{edad} años",
                        "Sexo":              sexo_lbl,
                        "Estado civil":      ecivil_lbl,
                        "Años de educación": f"{anios_edu} años",
                        "Zona":              zona_lbl,
                        "Departamento":      dpto_lbl,
                        "Horas / semana":    f"{p6800} h",
                        "Rama de actividad": rama_lbl,
                        "Subempleado":       "Sí (< 32 h/sem)" if p6800 < 32 else "No",
                        "Pluriempleo":       "Sí" if pluriemp else "No",
                    }

                    if st.button(
                        "Generar recomendaciones con IA",
                        key="btn_ia_recom",
                        type="primary",
                    ):
                        with st.spinner("Analizando el perfil y generando recomendaciones..."):
                            try:
                                texto_ia = generar_recomendaciones_ia(
                                    anthropic_api_key, perfil_legible, prob, riesgo
                                )
                                st.session_state["recomendaciones_ia"] = texto_ia
                            except Exception as e_ia:
                                if "auth" in str(e_ia).lower() or "api key" in str(e_ia).lower() or "401" in str(e_ia):
                                    st.error("API Key inválida. Verifica la clave de Groq en el panel lateral.")
                                else:
                                    st.error(f"Error al llamar a Groq API: {e_ia}")

                    if "recomendaciones_ia" in st.session_state:
                        st.markdown(st.session_state["recomendaciones_ia"])
                        st.caption(
                            "Recomendaciones generadas por Llama 3.3 70B (Groq) "
                            "basadas en el perfil del trabajador y el contexto GEIH 2025. "
                            "No constituyen asesoría oficial del Ministerio de Trabajo."
                        )

            else:
                st.success(
                    f"**✅ Perfil FORMAL — Probabilidad de informalidad {prob:.1%}.** "
                    "El trabajador presenta características asociadas al empleo formal."
                )

            with st.expander("📊 Factores de riesgo clave según el modelo SHAP"):
                st.markdown("""
| Factor | Dirección | Impacto relativo |
|---|---|---|
| Años de educación bajos | ↑ Mayor riesgo | ★★★★★ |
| Departamentos con alta informalidad estructural | ↑ Mayor riesgo | ★★★★☆ |
| Sector agricultura / construcción / hogares | ↑ Mayor riesgo | ★★★☆☆ |
| Zona rural | ↑ Mayor riesgo | ★★★☆☆ |
| Pocas horas semanales (subempleo) | ↑ Mayor riesgo | ★★★☆☆ |
| Educación técnica / universitaria o más | ↓ Reduce riesgo | ★★★★★ |
| Zona cabecera municipal | ↓ Reduce riesgo | ★★★☆☆ |
| Más horas semanales trabajadas | ↓ Reduce riesgo | ★★☆☆☆ |
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
**1. Magnitud:**
~85% de los trabajadores por cuenta propia en Colombia son informales (tasa ponderada DANE 2025).
La formalidad entre independientes no es la norma — es la excepción.

**2. La informalidad no es una elección:**
El modelo (AUC-ROC 0.9853, F1 0.9576) muestra que el sector, la zona geográfica y la educación
predicen la informalidad con alta precisión. La condición de informal está estructuralmente determinada.

**3. Predictores dominantes (SHAP):**
- La educación es el factor modificable de mayor impacto: a más años, menor riesgo.
- La geografía importa: departamentos como Vaupés y Chocó superan el 85%.
- La rama de actividad define el contexto: agricultura, construcción y hogares concentran los perfiles más vulnerables.

**4. Sectores críticos:**
Agricultura, construcción y hogares con servicio doméstico
concentran las tasas más altas entre independientes (>85%).

**5. Rendimiento del modelo:**
LightGBM logra AUC-ROC 0.9853 y F1 0.9576 — muy por encima de las metas (AUC ≥ 0.80, F1 ≥ 0.75).
        """)

    with rec_col:
        st.markdown("#### 💡 Recomendaciones accionables")
        st.markdown("""
**Argumento central:**
Los programas de formalización fallan cuando son universales.
Un independiente rural en el sector agrícola tiene un perfil completamente diferente
al de uno urbano con educación técnica. Diseñar intervenciones según ese perfil
aumentaría significativamente su efectividad.

**Para el Ministerio del Trabajo:**
- Usar el score de este modelo para priorizar inspecciones laborales en zonas rurales
  de los departamentos con tasa > 75%.
- Escalar intervención por score: orientación (>40%), visita (>65%), acompañamiento activo (>80%).

**Para el SENA:**
- Focalizar programas de formación técnica en independientes jóvenes (15-24 años) de zonas rurales.
- Diseñar rutas específicas por sector (construcción, comercio, agricultura)
  en lugar de convocatorias abiertas genéricas.

**Para Planeación Nacional:**
- Integrar el modelo en el FILCO para pasar de indicadores agregados
  departamentales a scores individuales de riesgo.
- Reentrenar con cada nueva ola anual de la GEIH para mantener vigencia.
        """)

st.caption(
    "Fuente: DANE – Gran Encuesta Integrada de Hogares 2025. "
    "El DANE no avala los resultados del análisis. "
    "Maestría en Ciencia de Datos y Analítica · EAFIT · 2025."
)
