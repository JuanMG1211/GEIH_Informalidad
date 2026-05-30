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

@st.cache_data(ttl=86400)
def load_colombia_geojson():
    """Descarga GeoJSON de departamentos de Colombia e inyecta códigos DANE en properties.DANE."""
    import unicodedata, requests

    def _norm(s):
        return "".join(
            c for c in unicodedata.normalize("NFD", str(s).lower())
            if unicodedata.category(c) != "Mn"
        ).strip()

    # Lookup normalizado nombre → código DANE
    lookup = {_norm(v[0]): k for k, v in DPTO_INFO.items()}
    lookup.update({
        "bogota d.c.": 11, "bogota dc": 11, "bogota": 11,
        "norte de santander": 54, "n. de santander": 54, "n de santander": 54,
        "san andres": 88, "san andres providencia y santa catalina": 88,
        "narino": 52, "choco": 27, "vaupes": 97, "guainia": 94,
        "valle del cauca": 76, "valle": 76,
    })

    urls = [
        "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/colombia-departments.geojson",
        "https://raw.githubusercontent.com/marcovega/colombia-geojson/master/Colombia.geo.json",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=12)
            if r.status_code != 200:
                continue
            gj = r.json()
            if len(gj.get("features", [])) < 20:
                continue
            for feat in gj["features"]:
                props = feat.get("properties", {})
                # Intenta con campos comunes de nombre
                raw = (props.get("name") or props.get("NOMBRE_DPT")
                       or props.get("NOM_DEP") or props.get("departamento") or "")
                dane = lookup.get(_norm(raw))
                if dane is None:
                    # Intenta primer token del nombre
                    first = _norm(raw).split()[0] if raw else ""
                    dane = lookup.get(first)
                feat["properties"]["DANE"] = dane
            return gj
        except Exception:
            continue
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

# ── Gráfico: top 5 variables SHAP — cómo dividen la informalidad ─────────
def build_shap_division_chart(df, tasa_global):
    """5 subplots de línea que muestran la tasa de informalidad por categoría
    de cada uno de los 5 predictores SHAP más importantes del modelo."""
    LABELS_TAM = {1:"1", 2:"2-5", 3:"6-10", 4:"11-19", 5:"20-30",
                  6:"31-50", 7:"51-100", 8:"101-200", 9:"201+", 10:"NS"}
    LABELS_EDU = {1:"Ninguno", 2:"Preescol.", 3:"Prim.inc.", 4:"Primaria",
                  5:"Sec.inc.", 6:"Sec.", 7:"Med.inc.", 8:"Media",
                  9:"Técnica", 10:"Univers.", 11:"Espec.", 12:"Maestría", 13:"Doctor."}
    LABELS_POS = {1:"Emp. particular", 2:"Emp. gobierno", 3:"Doméstico",
                  4:"Cuenta propia", 5:"Empleador", 6:"Familiar s/rem.",
                  7:"Jornalero", 8:"Otro"}

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "① Tamaño del establecimiento",
            "② Tipo de contrato",
            "③ Posición ocupacional",
            "④ Nivel educativo",
            "⑤ Grupo de edad",
            "",
        ],
        vertical_spacing=0.30,
        horizontal_spacing=0.12,
    )

    def color_punto(v):
        if v >= tasa_global + 8:
            return "#C0392B"   # rojo — muy por encima del promedio
        elif v >= tasa_global:
            return "#E67E22"   # naranja — sobre el promedio
        elif v >= tasa_global - 10:
            return "#F1C40F"   # amarillo — cerca del promedio
        else:
            return "#27AE60"   # verde — bajo informalidad

    def trazar(x_vals, y_vals, row, col):
        y_r  = [round(v, 1) for v in y_vals]
        x_s  = [str(v) for v in x_vals]
        cols = [color_punto(v) for v in y_r]

        fig.add_trace(go.Scatter(
            x=x_s, y=y_r,
            mode="lines+markers+text",
            line=dict(color="#2C3E50", width=3),
            marker=dict(
                size=16,
                color=cols,
                line=dict(color="white", width=2.5),
            ),
            text=[f"<b>{v:.0f}%</b>" for v in y_r],
            textposition="top center",
            textfont=dict(size=12, color="#1a1a2e", family="Arial"),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Informalidad: <b>%{y:.1f}%</b><extra></extra>",
        ), row=row, col=col)

    # ① Tamaño del establecimiento — de 1 persona a 201+
    t1 = (
        df.assign(_t=df["P3069"].map(LABELS_TAM))
        .dropna(subset=["_t"])
        .groupby(["P3069", "_t"], observed=True)
        .apply(tasa_pond_grp, include_groups=False)
        .reset_index()
        .sort_values("P3069")
    )
    t1.columns = ["P3069", "cat", "tasa"]
    trazar(t1["cat"], t1["tasa"], 1, 1)

    # ② Tipo de contrato
    t2 = (
        df.assign(_c=df["P6450"].map({1: "Verbal", 2: "Escrito", 9: "NS/NR"}))
        .dropna(subset=["_c"])
        .groupby("_c", observed=True)
        .apply(tasa_pond_grp, include_groups=False)
        .reset_index()
    )
    t2.columns = ["cat", "tasa"]
    t2 = t2.sort_values("tasa")
    trazar(t2["cat"], t2["tasa"], 1, 2)

    # ③ Posición ocupacional — ordenada de menor a mayor informalidad
    t3 = (
        df.assign(_p=df["P6430"].map(LABELS_POS))
        .dropna(subset=["_p"])
        .groupby("_p", observed=True)
        .apply(tasa_pond_grp, include_groups=False)
        .reset_index()
    )
    t3.columns = ["cat", "tasa"]
    t3 = t3.sort_values("tasa")
    trazar(t3["cat"], t3["tasa"], 1, 3)

    # ④ Nivel educativo — del más bajo al más alto
    t4 = (
        df.assign(_e=df["P3042"].map(LABELS_EDU))
        .dropna(subset=["_e"])
        .groupby(["P3042", "_e"], observed=True)
        .apply(tasa_pond_grp, include_groups=False)
        .reset_index()
        .sort_values("P3042")
    )
    t4.columns = ["P3042", "cat", "tasa"]
    trazar(t4["cat"], t4["tasa"], 2, 1)

    # ⑤ Grupo de edad — quinquenios 15-74
    df_age = df[df["P6040"].between(15, 74)].copy()
    df_age["_g"] = pd.cut(df_age["P6040"], bins=range(14, 76, 5),
                           labels=[f"{i}-{i+4}" for i in range(15, 75, 5)])
    t5 = (
        df_age.dropna(subset=["_g"])
        .groupby("_g", observed=True)
        .apply(tasa_pond_grp, include_groups=False)
        .reset_index()
    )
    t5.columns = ["cat", "tasa"]
    trazar(t5["cat"].astype(str), t5["tasa"], 2, 2)

    # Línea de referencia global
    fig.add_hline(
        y=tasa_global,
        line_dash="dash",
        line_color="rgba(44,62,80,0.55)",
        line_width=2,
        annotation_text=f"<b>Promedio nacional: {tasa_global:.1f}%</b>",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#2C3E50"),
        annotation_bgcolor="rgba(255,255,255,0.85)",
    )

    # Leyenda de colores (como texto en la figura)
    fig.add_annotation(
        text=(
            "<b>Color del punto:</b>  "
            "<span style='color:#C0392B'>●</span> Alto riesgo  "
            "<span style='color:#E67E22'>●</span> Sobre promedio  "
            "<span style='color:#F1C40F'>●</span> Cerca del promedio  "
            "<span style='color:#27AE60'>●</span> Bajo riesgo"
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.04,
        showarrow=False,
        font=dict(size=11),
        align="center",
    )

    fig.update_yaxes(
        range=[0, 118],
        ticksuffix="%",
        tickfont=dict(size=11, color="#333"),
        gridcolor="#E8ECF0",
        gridwidth=1,
        zeroline=False,
    )
    fig.update_xaxes(
        tickangle=-35,
        tickfont=dict(size=10, color="#333"),
    )
    fig.update_annotations(font=dict(size=13, family="Arial"))
    fig.update_layout(
        height=680,
        showlegend=False,
        margin=dict(t=80, b=70, l=55, r=20),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="#F4F6F9",
        paper_bgcolor="white",
    )
    return fig


def build_interaction_chart(df, tasa_global):
    """Interacción: tamaño de empresa × tipo de contrato → tasa de informalidad ponderada."""
    df_int = df[df["P6450"].isin([1, 2, 9]) & df["P3069"].notna()].copy()
    df_int["MICRO"]    = df_int["P3069"].isin([1, 2, 3]).astype(int)
    df_int["Contrato"] = df_int["P6450"].map({1: "Verbal", 2: "Escrito", 9: "NS/NR"})
    df_int["Empresa"]  = df_int["MICRO"].map({
        1: "Microempresa  (1–10 personas)",
        0: "Gran empresa  (11+ personas)",
    })

    tasa_int = (
        df_int
        .groupby(["Empresa", "Contrato"], observed=True)
        .apply(tasa_pond_grp, include_groups=False)
        .reset_index()
        .rename(columns={0: "tasa"})
    )
    tasa_int["pct"] = tasa_int["tasa"].round(1)
    orden = {"Verbal": 0, "Escrito": 1, "NS/NR": 2}
    tasa_int = tasa_int.assign(_ord=tasa_int["Contrato"].map(orden)).sort_values(["_ord", "Empresa"])

    # Color semántico por nivel de riesgo (no por empresa)
    def bar_color(pct, is_micro):
        if pct >= 70:   return "#A93226" if is_micro else "#E74C3C"
        elif pct >= 40: return "#CA6F1E" if is_micro else "#E67E22"
        elif pct >= 20: return "#B7950B" if is_micro else "#F4D03F"
        else:           return "#1A5276" if is_micro else "#2980B9"

    micro_v = tasa_int.query("Empresa.str.startswith('Micro') and Contrato=='Verbal'")["pct"].values[0]
    micro_e = tasa_int.query("Empresa.str.startswith('Micro') and Contrato=='Escrito'")["pct"].values[0]
    delta   = micro_v - micro_e

    fig = go.Figure()

    # Zonas de fondo
    fig.add_hrect(y0=70, y1=120, fillcolor="rgba(169,50,38,0.07)",  line_width=0, layer="below")
    fig.add_hrect(y0=0,  y1=25,  fillcolor="rgba(26,82,118,0.07)",  line_width=0, layer="below")

    for empresa in ["Microempresa  (1–10 personas)", "Gran empresa  (11+ personas)"]:
        sub      = tasa_int[tasa_int["Empresa"] == empresa]
        is_micro = empresa.startswith("Micro")
        cols     = [bar_color(v, is_micro) for v in sub["pct"]]
        fig.add_trace(go.Bar(
            name=("▮ " if is_micro else "▯ ") + empresa.strip(),
            x=sub["Contrato"],
            y=sub["pct"],
            marker=dict(
                color=cols,
                opacity=1.0 if is_micro else 0.60,
                line=dict(color="white", width=2.5),
                pattern_shape="" if is_micro else "/",
            ),
            text=[f"<b>{v:.1f}%</b>" for v in sub["pct"]],
            textposition="outside",
            textfont=dict(size=17, family="Arial Black, Arial Bold, Arial", color="#111111"),
        ))

    # Línea promedio nacional
    fig.add_hline(
        y=tasa_global,
        line_dash="dash", line_color="rgba(44,62,80,0.50)", line_width=2,
        annotation_text=f"<b>Promedio nacional: {tasa_global:.1f}%</b>",
        annotation_position="top right",
        annotation_font=dict(size=12, color="#2C3E50"),
        annotation_bgcolor="rgba(255,255,255,0.88)",
    )

    # Caja del hallazgo principal (top-left)
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98,
        text=(
            f"<b>💡  CONTRATO ESCRITO en microempresa:</b><br>"
            f"<b>    {micro_v:.1f}%  →  {micro_e:.1f}%  &nbsp;(↓ {delta:.0f} puntos porcentuales)</b>"
        ),
        showarrow=False,
        font=dict(size=14, color="#A93226", family="Arial"),
        bgcolor="rgba(255,245,245,0.96)",
        bordercolor="#A93226", borderwidth=2, borderpad=10,
        align="left", xanchor="left", yanchor="top",
    )

    # Etiquetas de zona
    fig.add_annotation(
        xref="paper", yref="y", x=0.995, y=108,
        text="<b>⚠ ZONA CRÍTICA  (&gt;70%)</b>",
        showarrow=False, font=dict(size=11, color="#A93226"), xanchor="right",
    )
    fig.add_annotation(
        xref="paper", yref="y", x=0.995, y=13,
        text="<b>✓ ZONA SEGURA  (&lt;25%)</b>",
        showarrow=False, font=dict(size=11, color="#1A5276"), xanchor="right",
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>¿El contrato escrito protege incluso en microempresas?</b><br>"
                "<sup>Interacción entre las dos variables más importantes del modelo  "
                "(SHAP #1 Microempresa · SHAP #2 Contrato verbal)</sup>"
            ),
            font=dict(size=17, family="Arial Black, Arial Bold, Arial", color="#1a1a2e"),
        ),
        xaxis=dict(
            title="<b>Tipo de contrato</b>",
            title_font=dict(size=14, color="#333"),
            tickfont=dict(size=15, family="Arial Black, Arial Bold", color="#222"),
        ),
        yaxis=dict(
            title="<b>Tasa de informalidad (%)</b>",
            title_font=dict(size=14, color="#333"),
            tickfont=dict(size=13, color="#333"),
            range=[0, 120],
            ticksuffix="%",
            gridcolor="#DCDCDC", gridwidth=1, zeroline=False,
        ),
        barmode="group",
        bargap=0.28,
        bargroupgap=0.05,
        height=510,
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=13),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.04,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#CCCCCC", borderwidth=1,
            font=dict(size=13),
        ),
        margin=dict(t=130, b=55, l=75, r=30),
    )
    fig.update_xaxes(gridcolor="#DCDCDC", zeroline=False)
    return fig


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

            titulo_mapa = f"Tasa de informalidad laboral (%) — {zona_sel} · {sector_sel}"
            geojson = load_colombia_geojson()

            if geojson is not None:
                fig_map = px.choropleth_mapbox(
                    inf_dpto,
                    geojson=geojson,
                    locations="DPTO",
                    featureidkey="properties.DANE",
                    color="tasa_pct",
                    color_continuous_scale="RdYlGn_r",
                    range_color=[20, 90],
                    mapbox_style="carto-positron",
                    zoom=4.55,
                    center={"lat": 4.5, "lon": -74.0},
                    opacity=0.80,
                    hover_name="nombre",
                    hover_data={"tasa_pct": ":.1f", "DPTO": False,
                                "lat": False, "lon": False},
                    labels={"tasa_pct": "% Informal"},
                    title=titulo_mapa,
                )
                fig_map.update_layout(
                    height=600,
                    margin=dict(r=0, t=50, l=0, b=0),
                    coloraxis_colorbar=dict(
                        title="% Informal",
                        ticksuffix="%",
                        len=0.65, thickness=16,
                        title_font_size=13,
                        tickfont_size=12,
                    ),
                )
            else:
                # Fallback: burbujas si el GeoJSON no cargó
                fig_map = px.scatter_geo(
                    inf_dpto, lat="lat", lon="lon",
                    size="tasa_pct", color="tasa_pct",
                    hover_name="nombre",
                    hover_data={"tasa_pct": True, "lat": False, "lon": False},
                    color_continuous_scale="RdYlGn_r",
                    range_color=[20, 90], size_max=45,
                    labels={"tasa_pct": "% Informal"},
                    title=titulo_mapa,
                )
                fig_map.update_geos(
                    visible=False, showcountries=True, countrycolor="#aaa",
                    showsubunits=True, subunitcolor="#ccc",
                    lonaxis_range=[-83, -65], lataxis_range=[-5, 14],
                    bgcolor="#f8f9fa",
                )
                fig_map.update_layout(height=560, margin=dict(r=0, t=40, l=0, b=0))

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
        geojson = load_colombia_geojson()
        if geojson is not None:
            fig_map = px.choropleth_mapbox(
                inf_dpto,
                geojson=geojson,
                locations="DPTO",
                featureidkey="properties.DANE",
                color="tasa_pct",
                color_continuous_scale="RdYlGn_r",
                range_color=[30, 85],
                mapbox_style="carto-positron",
                zoom=4.55,
                center={"lat": 4.5, "lon": -74.0},
                opacity=0.80,
                hover_name="nombre",
                hover_data={"tasa_pct": ":.1f", "DPTO": False,
                            "lat": False, "lon": False},
                labels={"tasa_pct": "% Informal"},
                title="Tasa de informalidad laboral (%) por departamento",
            )
            fig_map.update_layout(
                height=600, margin=dict(r=0, t=50, l=0, b=0),
                coloraxis_colorbar=dict(title="% Informal", ticksuffix="%",
                                        len=0.65, thickness=16),
            )
        else:
            fig_map = px.scatter_geo(
                inf_dpto, lat="lat", lon="lon",
                size="tasa_pct", color="tasa_pct",
                hover_name="nombre",
                color_continuous_scale="RdYlGn_r",
                range_color=[30, 85], size_max=40,
                labels={"tasa_pct": "% Informal"},
                title="Tasa de informalidad laboral (%) por departamento",
            )
            fig_map.update_geos(
                visible=False, showcountries=True, countrycolor="#aaa",
                lonaxis_range=[-83, -65], lataxis_range=[-5, 14],
                bgcolor="#f8f9fa",
            )
            fig_map.update_layout(height=560, margin=dict(r=0, t=40, l=0, b=0))
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

        # ── Scatter bivariado: separación Formal vs Informal ──────────────
        if "P6040" in df.columns and "P3042" in df.columns:
            st.divider()
            st.markdown("#### 📈 Separación bivariada — Formal vs Informal")
            st.caption(
                "Muestra de 6 000 trabajadores. **Rojo** = Informal (arriba) · **Azul** = Formal (abajo). "
                "La línea oscura muestra cómo cambia la probabilidad de informalidad a lo largo de cada variable."
            )

            MAPA_EDU_SC = {1:0, 2:1, 3:3, 4:5, 5:7, 6:9, 7:10,
                           8:11, 9:14, 10:16, 11:17, 12:18, 13:21}

            df_sc = df[df["INFORMAL"].notna()].copy()
            df_sc = df_sc.sample(min(6000, len(df_sc)), random_state=42)
            df_sc["Condición"] = df_sc["INFORMAL"].map({0: "Formal", 1: "Informal"})
            df_sc["anios_edu"] = df_sc["P3042"].map(MAPA_EDU_SC)

            rng = np.random.default_rng(42)
            df_sc["y_jit"] = (
                df_sc["INFORMAL"].astype(float)
                + rng.uniform(-0.07, 0.07, len(df_sc))
            )

            COLORES_SC = {"Formal": "#2196F3", "Informal": "#E05C5C"}

            def tendencia_biv(sub, x_col, n_bins=18):
                clean = sub.dropna(subset=[x_col, "INFORMAL"])
                bins  = pd.cut(clean[x_col], bins=n_bins)
                prop  = (
                    clean.groupby(bins, observed=True)["INFORMAL"]
                    .agg(["mean", "count"])
                )
                prop  = prop[prop["count"] >= 10]
                x_c   = np.array([iv.mid for iv in prop.index])
                y_p   = prop["mean"].values
                orden = np.argsort(x_c)
                return x_c[orden], y_p[orden]

            def scatter_biv(x_col, x_label, key):
                fig = go.Figure()
                sub = df_sc.dropna(subset=[x_col])

                for cond, color in COLORES_SC.items():
                    mask = sub["Condición"] == cond
                    fig.add_trace(go.Scatter(
                        x=sub.loc[mask, x_col],
                        y=sub.loc[mask, "y_jit"],
                        mode="markers",
                        name=cond,
                        marker=dict(
                            color=color, size=5, opacity=0.38,
                            line=dict(color="white", width=0.3),
                        ),
                        hovertemplate=(
                            f"<b>{cond}</b><br>{x_label}: %{{x:.1f}}<extra></extra>"
                        ),
                    ))

                x_c, y_p = tendencia_biv(sub, x_col)
                if len(x_c) > 1:
                    fig.add_trace(go.Scatter(
                        x=x_c, y=y_p,
                        mode="lines",
                        line=dict(color="#2C3E50", width=3.5),
                        showlegend=False,
                        hovertemplate=(
                            f"{x_label}: %{{x:.1f}}<br>"
                            "P(Informal): %{y:.2f}<extra></extra>"
                        ),
                    ))

                fig.update_layout(
                    xaxis_title=x_label,
                    yaxis=dict(
                        title="Condición",
                        tickvals=[0, 1],
                        ticktext=["Formal", "Informal"],
                        range=[-0.2, 1.2],
                        gridcolor="white",
                        gridwidth=1.5,
                        zeroline=False,
                        tickfont=dict(size=12, color="#333"),
                    ),
                    plot_bgcolor="#EBEBEB",
                    paper_bgcolor="white",
                    font=dict(family="Arial, sans-serif", size=11, color="#333"),
                    legend=dict(
                        bgcolor="rgba(255,255,255,0.88)",
                        bordercolor="#BBBBBB", borderwidth=1,
                        font=dict(size=11),
                        orientation="h",
                        x=0.5, xanchor="center", y=1.10,
                    ),
                    height=390,
                    margin=dict(t=40, b=45, l=70, r=15),
                    hovermode="closest",
                )
                fig.update_xaxes(
                    gridcolor="white", gridwidth=1.5,
                    zeroline=False, tickfont=dict(size=10),
                )
                return fig

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown("**A · Educación**")
                st.plotly_chart(
                    scatter_biv("anios_edu", "Años de educación", "a"),
                    use_container_width=True, key="scatter_edu_ing",
                )
            with sc2:
                st.markdown("**B · Edad**")
                st.plotly_chart(
                    scatter_biv("P6040", "Edad (años)", "b"),
                    use_container_width=True, key="scatter_edad_ing",
                )
            with sc3:
                st.markdown("**C · Horas trabajadas**")
                st.plotly_chart(
                    scatter_biv("P6800", "Horas / semana", "c"),
                    use_container_width=True, key="scatter_horas_ing",
                )

        # ── Top 5 SHAP: cómo dividen la informalidad ─────────────────────
        st.divider()
        st.markdown("#### 🎯 Top 5 variables SHAP — ¿Cómo dividen la informalidad?")
        st.caption(
            "Cada línea conecta la tasa de informalidad ponderada (%) para cada categoría del predictor. "
            "La línea punteada es el promedio nacional de referencia. "
            "Cuanto más pronunciada la curva, mayor poder de separación tiene esa variable."
        )
        tasa_global_eda = (df["INFORMAL"] * df["FEX_C18"]).sum() / df["FEX_C18"].sum() * 100
        st.plotly_chart(
            build_shap_division_chart(df, tasa_global_eda),
            use_container_width=True,
            key="shap_division_eda",
        )

        # ── Efecto de interacción: Microempresa × Tipo de contrato ───────
        if "P3069" in df.columns and "P6450" in df.columns:
            st.divider()
            st.markdown("#### 🔗 Efecto de interacción — Microempresa × Tipo de contrato")
            st.caption(
                "Las dos variables más importantes del modelo (SHAP #1 y #2) **no son independientes**: "
                "dentro de una microempresa, pasar de contrato verbal a escrito reduce la informalidad "
                "de forma drástica. La intervención de mayor impacto es **formalizar el contrato** "
                "en empresas pequeñas, no solo fiscalizar el tamaño."
            )
            st.plotly_chart(
                build_interaction_chart(df, tasa_global_eda),
                use_container_width=True,
                key="interaction_micro_contrato",
            )
            with st.expander("¿Qué significa esto para política pública?"):
                st.markdown("""
**Lectura del gráfico:**
- Una barra **roja alta** con contrato verbal en microempresa confirma el perfil clásico de informalidad.
- La barra **roja más baja** con contrato escrito muestra que la formalización contractual *dentro de la misma microempresa* protege significativamente.
- Las barras **azules** (gran empresa) son consistentemente más bajas, pero también se benefician del contrato escrito.

**Implicación directa para el Ministerio del Trabajo:**
> Focalizar las inspecciones laborales y los incentivos de contractualización escrita en **microempresas del sector informal**
> produce mayor impacto que cualquier otra intervención, porque combina el efecto del predictor #1 (tamaño)
> con el predictor #2 (tipo de contrato).
""")

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
