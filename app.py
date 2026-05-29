import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="Informalidad Laboral · Colombia 2024",
    page_icon="🇨🇴",
    layout="wide",
)

BASE          = Path(__file__).parent
MODEL_PATH    = BASE / "outputs" / "champion_geih.pkl"
META_PATH     = BASE / "outputs" / "champion_geih_meta.json"
DATA_PATH     = BASE / "outputs" / "datos_procesados.parquet"
PREPROC_PATH  = BASE / "parquet"  / "preprocessor.joblib"

# ── Catálogos ──────────────────────────────────────────────────────────
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

POSICION = {1: "Empleado particular", 2: "Empleado del gobierno",
            3: "Empleado doméstico",  4: "Cuenta propia",
            5: "Empleador / Patrón",  6: "Familiar sin remuneración",
            7: "Jornalero / Peón",    8: "Otro"}

ESTADO_CIVIL = {1: "No unido/a", 2: "Unión libre", 3: "Casado/a",
                4: "Separado/a", 5: "Viudo/a", 6: "NS/NR"}

CONTRATO_TIPO = {1: "Verbal", 2: "Escrito", 9: "No sabe / No aplica"}

TAMANO_EMP = {
    1: "1 persona",     2: "2–5 personas",  3: "6–10 personas",
    4: "11–19 personas",5: "20–30 personas", 6: "31–50 personas",
    7: "51–100",        8: "101–200",        9: "201 o más",
    10: "No sabe",
}

RAMA_CIIU = {
    1: "Agricultura/Pesca", 5: "Minería", 10: "Industria manufacturera",
    36: "Agua/Servicios", 41: "Construcción", 45: "Comercio/Vehículos",
    49: "Transporte", 55: "Alojamiento/Restaurantes",
    58: "Comunicaciones", 64: "Financiero", 68: "Inmobiliario",
    75: "Veterinaria", 78: "Servicios adm.", 84: "Administración pública",
    85: "Educación", 86: "Salud", 90: "Artes/Entretenimiento",
    97: "Hogares con servicio dom.", 99: "Otro/NS",
}

# ── Carga de recursos ───────────────────────────────────────────────────
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

# ── Función de predicción ───────────────────────────────────────────────
def construir_input(p6040, p3271, p6070, clase, dpto,
                    p6430, p6450, p6800, p3069, rama,
                    anios_edu, pluriempleo):
    """Construye el DataFrame de features raw que espera el preprocessor."""
    mapa_edu = {1:0,2:1,3:3,4:5,5:7,6:9,7:10,8:11,9:14,10:16,11:17,12:18,13:21}
    # EDAD_GRUPO derivado igual que en Fase 3
    bins   = [14, 24, 34, 44, 54, 64, 120]
    labels = ["15-24","25-34","35-44","45-54","55-64","65+"]
    edad_grupo = pd.cut([p6040], bins=bins, labels=labels)[0]

    row = {
        "P3271":          p3271,
        "P6040":          p6040,
        "EDAD_GRUPO":     str(edad_grupo),
        "ANIOS_EDU":      anios_edu,
        "P6070":          p6070,
        "CLASE":          clase,
        "DPTO":           dpto,
        "P6430":          p6430,
        "P6450":          p6450,
        "P6800":          p6800,
        "P3069":          p3069,
        "RAMA2D_R4":      rama,
        "CUENTA_PROPIA":  int(p6430 == 4),
        "MICROEMPRESA":   int(p3069 in [1, 2, 3]),
        "SUBEMPLEADO":    int(p6800 < 32),
        "PLURIEMPLEO":    pluriempleo,
        "CONTRATO_VERBAL": int(p6450 == 1),
    }
    return pd.DataFrame([row])

# ── Header ──────────────────────────────────────────────────────────────
st.title("🇨🇴 Predicción de Informalidad Laboral · Colombia 2024")
st.caption("Proyecto Final de Maestría · Juan Andrés Montoya · Julián David Mejía · GEIH 2024 – DANE")

model = preprocessor = meta = None
try:
    model        = load_model()
    preprocessor = load_preprocessor()
    meta         = load_meta()
except Exception as e:
    st.warning(f"⚠️ Modelo no cargado — ejecuta primero `4.Modelamiento.ipynb`. ({e})")

df = load_data()

# ── KPIs ────────────────────────────────────────────────────────────────
if meta:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Modelo campeón",  meta.get("model", "—"))
    k2.metric("F1-Score (test)", f"{meta.get('test_f1',  0):.4f}")
    k3.metric("AUC-ROC (test)",  f"{meta.get('test_auc', 0):.4f}")
    k4.metric("Muestra entrén.", f"{meta.get('n_train',  0):,}")

st.divider()

tab1, tab2, tab3 = st.tabs(["🗺️ Mapa departamental", "📊 EDA Interactivo", "🔮 Predicción individual"])

# ══════════════════════════════════════════════
# TAB 1 — Mapa departamental
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Tasa de informalidad por departamento · GEIH 2024")

    if df is not None:
        def tasa_pond_grp(g):
            return (g["INFORMAL"] * g["FEX_C18"]).sum() / g["FEX_C18"].sum() * 100

        inf_dpto = (
            df[df["DPTO"].notna()]
            .groupby("DPTO", observed=True)
            .apply(tasa_pond_grp, include_groups=False)
            .reset_index()
            .rename(columns={0: "tasa"})
            .query("tasa > 0")
        )
        inf_dpto["DPTO"] = inf_dpto["DPTO"].astype(int)
    else:
        np.random.seed(42)
        rows = [(k, round(np.random.uniform(0.35, 0.82), 3)) for k in DPTO_INFO]
        inf_dpto = pd.DataFrame(rows, columns=["DPTO", "tasa"])
        st.info("Datos de ejemplo — ejecuta los notebooks para datos reales.")

    inf_dpto["nombre"]   = inf_dpto["DPTO"].map(lambda d: DPTO_INFO.get(d, (str(d),0,0))[0])
    inf_dpto["lat"]      = inf_dpto["DPTO"].map(lambda d: DPTO_INFO.get(d, ("",4.5,-74))[1])
    inf_dpto["lon"]      = inf_dpto["DPTO"].map(lambda d: DPTO_INFO.get(d, ("",4.5,-74))[2])
    inf_dpto["tasa_pct"] = inf_dpto["tasa"].round(1)

    fig_map = px.scatter_geo(
        inf_dpto, lat="lat", lon="lon",
        size="tasa_pct", color="tasa_pct",
        hover_name="nombre",
        hover_data={"tasa_pct": True, "lat": False, "lon": False},
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

    st.dataframe(
        inf_dpto[["nombre", "tasa_pct"]]
            .sort_values("tasa_pct", ascending=False)
            .rename(columns={"nombre": "Departamento", "tasa_pct": "% Informal"})
            .reset_index(drop=True),
        use_container_width=True, height=320,
    )

# ══════════════════════════════════════════════
# TAB 2 — EDA Interactivo
# ══════════════════════════════════════════════
with tab2:
    if df is None:
        st.info("Ejecuta los notebooks para cargar datos reales.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            counts = df["INFORMAL"].value_counts().reset_index()
            counts.columns = ["Condición", "N"]
            counts["Condición"] = counts["Condición"].map({0: "Formal", 1: "Informal"})
            fig_pie = px.pie(counts, values="N", names="Condición",
                             color_discrete_sequence=["#2196F3", "#FF5722"],
                             title="Distribución Formal / Informal")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            t_zona = (
                df[df["CLASE"].notna()]
                .groupby("CLASE", observed=True)["INFORMAL"]
                .mean().reset_index()
            )
            t_zona["pct"]      = (t_zona["INFORMAL"] * 100).round(1)
            t_zona["etiqueta"] = t_zona["CLASE"].map({1: "Cabecera", 2: "Rural"})
            fig_zona = px.bar(t_zona, x="etiqueta", y="pct",
                              color="etiqueta",
                              color_discrete_sequence=["#5DCAA5", "#EF9F27"],
                              title="Informalidad por zona (%)",
                              labels={"pct": "% Informal", "etiqueta": "Zona"})
            st.plotly_chart(fig_zona, use_container_width=True)

        # Por posición ocupacional
        LABELS_POS = {1:"Empleado particular", 2:"Empleado gobierno",
                      3:"Doméstico", 4:"Cuenta propia", 5:"Empleador",
                      6:"Familiar s/rem.", 7:"Jornalero", 8:"Otro"}
        t_pos = (
            df[df["P6430"].notna()]
            .groupby("P6430", observed=True)["INFORMAL"]
            .mean().reset_index()
            .sort_values("INFORMAL")
        )
        t_pos["pct"]      = (t_pos["INFORMAL"] * 100).round(1)
        t_pos["etiqueta"] = t_pos["P6430"].map(LABELS_POS)
        fig_pos = px.bar(t_pos, x="pct", y="etiqueta", orientation="h",
                         color="pct", color_continuous_scale="RdYlGn_r",
                         title="Tasa de informalidad por posición ocupacional (%)",
                         labels={"pct": "% Informal", "etiqueta": ""})
        fig_pos.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_pos, use_container_width=True)

        # Distribución de ingresos
        if "INGLABO" in df.columns:
            df_ing = df[df["INGLABO"].notna() & (df["INGLABO"] > 0)].copy()
            df_samp = df_ing.sample(min(6000, len(df_ing)), random_state=42)
            df_samp["Condición"] = df_samp["INFORMAL"].map({0: "Formal", 1: "Informal"})
            df_samp["log_ing"]   = np.log1p(df_samp["INGLABO"])
            fig_violin = px.violin(
                df_samp, x="Condición", y="log_ing",
                color="Condición",
                color_discrete_map={"Formal": "#2196F3", "Informal": "#FF5722"},
                box=True,
                title="Distribución de log(Ingreso laboral) por condición",
                labels={"log_ing": "log(INGLABO + 1)"},
            )
            st.plotly_chart(fig_violin, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — Predicción individual
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Estima la probabilidad de informalidad de un trabajador")

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
            p6070     = {v:k for k,v in ESTADO_CIVIL.items()}[ecivil_lbl]
            zona_lbl  = st.selectbox("Zona", ["Cabecera municipal", "Rural"])
            clase     = 1 if zona_lbl == "Cabecera municipal" else 2
            dpto_lbl  = st.selectbox("Departamento", [v[0] for v in DPTO_INFO.values()])
            dpto      = {v[0]:k for k,v in DPTO_INFO.items()}.get(dpto_lbl, 11)

        with col_r:
            st.markdown("**Datos laborales**")
            pos_lbl   = st.selectbox("Posición ocupacional", list(POSICION.values()))
            p6430     = {v:k for k,v in POSICION.items()}[pos_lbl]
            cont_lbl  = st.selectbox("Tipo de contrato", list(CONTRATO_TIPO.values()))
            p6450     = {v:k for k,v in CONTRATO_TIPO.items()}[cont_lbl]
            p6800     = st.slider("Horas trabajadas / semana", 1, 100, 48)
            tam_lbl   = st.selectbox("Tamaño del establecimiento", list(TAMANO_EMP.values()))
            p3069     = {v:k for k,v in TAMANO_EMP.items()}[tam_lbl]
            rama_lbl  = st.selectbox("Rama de actividad (CIIU)", list(RAMA_CIIU.values()))
            rama      = {v:k for k,v in RAMA_CIIU.items()}[rama_lbl]
            pluriemp  = st.checkbox("¿Tiene otro empleo adicional (pluriempleo)?", value=False)
            p7040_val = 1 if pluriemp else 0

        input_raw = construir_input(
            p6040=edad, p3271=p3271, p6070=p6070, clase=clase, dpto=dpto,
            p6430=p6430, p6450=p6450, p6800=p6800, p3069=p3069, rama=rama,
            anios_edu=anios_edu, pluriempleo=p7040_val
        )

        try:
            X_input = preprocessor.transform(input_raw)
            thresh  = meta.get("threshold", 0.5) if meta else 0.5
            prob    = model.predict_proba(X_input)[0, 1]
            pred    = int(prob >= thresh)

            st.divider()
            m1, m2, m3 = st.columns(3)
            riesgo = "Alto" if prob > 0.65 else "Medio" if prob > 0.4 else "Bajo"
            m1.metric("Probabilidad de informalidad", f"{prob:.1%}")
            m2.metric("Nivel de riesgo", riesgo)
            m3.metric("Clasificación", "INFORMAL" if pred == 1 else "FORMAL")

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
                    "threshold": {"line": {"color": "black", "width": 3},
                                  "thickness": 0.8, "value": 50},
                },
                title={"text": "Score de informalidad"},
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            with st.expander("Factores de riesgo principales según el modelo"):
                st.markdown("""
| Factor | Dirección |
|---|---|
| Cuenta propia · Jornalero · Doméstico | ↑ Mayor riesgo |
| Sector agropecuario y construcción | ↑ Mayor riesgo |
| Zona rural | ↑ Mayor riesgo (~10-15 pp) |
| Microempresa (≤ 10 trabajadores) | ↑ Mayor riesgo |
| Contrato verbal | ↑ Mayor riesgo |
| Años de educación | ↓ Reduce riesgo |
| Empresa grande (> 30 trabajadores) | ↓ Reduce riesgo |
""")

        except Exception as ex:
            st.error(f"Error en predicción: {ex}")
            st.info("Verifica que el preprocessor y el modelo sean del mismo pipeline.")

st.divider()
st.caption("Fuente: DANE – Gran Encuesta Integrada de Hogares 2024. El DANE no avala los resultados del análisis.")
