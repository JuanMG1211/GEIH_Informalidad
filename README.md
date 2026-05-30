# Predicción de Informalidad Laboral en Colombia · GEIH 2024

**Maestría en Ciencia de Datos y Analítica** · SI7006 · SI7007 · SI7009  
Juan Andrés Montoya · Julián David Mejía

---

## Requisitos del sistema

| Requisito | Versión mínima |
|-----------|---------------|
| Python | 3.11 |
| RAM | 4 GB |
| Espacio en disco | ~2 GB (datos crudos + artefactos) |
| Conexión a internet | Solo para el mapa departamental (GeoJSON) |

---

## Estructura del proyecto

```
Proyecto_Final_maestria/
│
├── 2024_data/                          ← Datos crudos GEIH (NO subir a GitHub)
│   ├── Caracteristicas_Generales/
│   │   ├── 1.Caracteristicas_Generales_Enero.csv
│   │   └── ...  (12 archivos CSV)
│   └── Ocupados/
│       ├── 1.Ocupados_Enero.csv
│       └── ...  (12 archivos CSV)
│
├── parquet/                            ← Generado por los notebooks
│   ├── geih_2024_crudo.parquet
│   ├── train.parquet
│   ├── test.parquet
│   └── preprocessor.joblib
│
├── outputs/                            ← Generado por los notebooks
│   ├── champion_geih.pkl               ← Modelo LightGBM serializado
│   ├── champion_geih_meta.json         ← Métricas y umbral óptimo
│   ├── datos_procesados.parquet        ← Dataset para el dashboard
│   ├── metrics_comparison.csv
│   ├── shap_summary.png
│   ├── shap_importance.png
│   ├── confusion_matrix.png
│   └── viz_01_target.html … viz_06_shap.html
│
├── 1.Ingesta.ipynb                     ← Fase 1: carga y unión de módulos GEIH
├── 2.EDA.ipynb                         ← Fase 2: análisis exploratorio
├── 3.Preparacion.ipynb                 ← Fase 3: feature engineering + pipeline
├── 4.Modelamiento.ipynb                ← Fase 4: entrenamiento y selección
├── 5.Visualizacion.ipynb               ← Fase 5: gráficas Plotly interactivas
│
├── app.py                              ← Dashboard Streamlit (5 tabs)
├── requirements.txt                    ← Dependencias Python
├── geih_2024.duckdb                    ← Base de datos DuckDB
├── GUIA_ESTUDIO_PRESENTACION.md        ← Guía de estudio para la defensa
└── README.md                           ← Este archivo
```

---

## Instalación

### 1. Clonar o descargar el repositorio

```bash
git clone <url-del-repositorio>
cd Proyecto_Final_maestria
```

### 2. Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` incluye:

```
streamlit>=1.35.0
pandas>=2.2.0
numpy>=1.26.0
plotly>=5.18.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
joblib>=1.3.0
pyarrow>=15.0.0
requests>=2.31.0
```

> Si también vas a ejecutar los notebooks necesitas instalar adicionalmente:
> ```bash
> pip install duckdb shap xgboost category-encoders jupyter
> ```

---

## Opción A — Solo el dashboard (sin re-ejecutar notebooks)

Si ya tienes las carpetas `parquet/` y `outputs/` con todos los archivos generados, puedes lanzar el dashboard directamente:

```bash
python -m streamlit run app.py
```

Se abre automáticamente en el navegador: **http://localhost:8501**

> El mapa de departamentos descarga un GeoJSON de Colombia la primera vez que abres el tab. Requiere conexión a internet. Queda cacheado por 24 horas.

---

## Opción B — Reproducir el proyecto completo desde cero

Ejecuta los notebooks **en orden**. Cada uno depende del anterior.

### Fase 1 — Ingesta

```bash
jupyter notebook 1.Ingesta.ipynb
```

**Requiere:** carpeta `2024_data/` con los 24 archivos CSV del GEIH 2024 (descargados desde [microdatos.dane.gov.co](https://microdatos.dane.gov.co)).

**Genera:**
- `parquet/geih_2024_crudo.parquet`
- `geih_2024.duckdb`

Tiempo estimado: **2–5 minutos** (depende del hardware).

---

### Fase 2 — EDA

```bash
jupyter notebook 2.EDA.ipynb
```

**Requiere:** `parquet/geih_2024_crudo.parquet`

**Genera:** visualizaciones exploratorias (solo en el notebook, no escribe archivos).

---

### Fase 3 — Preparación

```bash
jupyter notebook 3.Preparacion.ipynb
```

**Requiere:** `parquet/geih_2024_crudo.parquet`

**Genera:**
- `parquet/train.parquet`
- `parquet/test.parquet`
- `parquet/preprocessor.joblib`

---

### Fase 4 — Modelamiento

```bash
jupyter notebook 4.Modelamiento.ipynb
```

**Requiere:** `parquet/train.parquet`, `parquet/test.parquet`

**Genera:**
- `outputs/champion_geih.pkl` (modelo LightGBM, ~11 MB)
- `outputs/champion_geih_meta.json`
- `outputs/datos_procesados.parquet`
- `outputs/metrics_comparison.csv`
- `outputs/shap_summary.png`, `shap_importance.png`, `confusion_matrix.png`

Tiempo estimado: **5–15 minutos** (early stopping en ~858 iteraciones).

---

### Fase 5 — Visualización

```bash
jupyter notebook 5.Visualizacion.ipynb
```

**Requiere:** `outputs/champion_geih.pkl`, `outputs/datos_procesados.parquet`, `parquet/test.parquet`

**Genera:**
- `outputs/viz_01_target.html` … `viz_06_shap.html`

---

### Lanzar el dashboard

```bash
python -m streamlit run app.py
```

---

## Despliegue en Streamlit Cloud

El dashboard está desplegado públicamente en **Streamlit Community Cloud** (gratuito).

### Pasos para re-desplegar

1. Asegúrate de que el repositorio en GitHub contiene:
   - `app.py`
   - `requirements.txt`
   - `outputs/` (modelo + datos + imágenes)
   - `parquet/preprocessor.joblib`

   > Los archivos en `2024_data/` **no** deben subirse (están en `.gitignore`).

2. Ingresa a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.

3. Clic en **New app** → selecciona el repositorio → selecciona `app.py` como archivo principal → **Deploy**.

4. Streamlit Cloud instala automáticamente las dependencias de `requirements.txt`.

---

## Resumen de resultados del modelo

| Modelo | F1-Score | AUC-ROC | Precision | Recall |
|--------|----------|---------|-----------|--------|
| **LightGBM (campeón)** | **0.9576** | **0.9853** | 0.9399 | 0.9760 |
| Random Forest | 0.9553 | 0.9818 | 0.9378 | 0.9734 |
| XGBoost | 0.9546 | 0.9825 | 0.9423 | 0.9673 |
| Logistic Regression | 0.9507 | 0.9738 | 0.9347 | 0.9673 |

- **Umbral de decisión:** 0.41 (calibrado por threshold tuning)
- **Variable más importante (SHAP):** Microempresa (1.40)
- **Tasa de informalidad ponderada GEIH 2024:** 56.0%

---

## Solución de problemas frecuentes

| Error | Causa | Solución |
|-------|-------|----------|
| `ModuleNotFoundError: streamlit` | Streamlit no instalado | `pip install -r requirements.txt` |
| `FileNotFoundError: champion_geih.pkl` | Falta la carpeta `outputs/` | Ejecuta `4.Modelamiento.ipynb` primero |
| `FileNotFoundError: datos_procesados.parquet` | Falta `outputs/` | Ejecuta `4.Modelamiento.ipynb` primero |
| `FileNotFoundError: preprocessor.joblib` | Falta `parquet/` | Ejecuta `3.Preparacion.ipynb` primero |
| El mapa aparece como burbujas (no polígonos) | Sin conexión a internet | Conéctate y recarga la página |
| Notebook tarda mucho en Fase 4 | Normal — early stopping en ~858 iter | Espera 5–15 min según hardware |
| `UnicodeDecodeError` en Fase 1 | Problema de encoding en CSV GEIH | Los CSV usan `latin-1` — no cambiar el parámetro `encoding` |

---

## Datos fuente

Los archivos CSV crudos **no están incluidos** en el repositorio por su tamaño (~180 MB sin comprimir).

Descárgalos desde el portal oficial del DANE:

**[microdatos.dane.gov.co → GEIH 2024](https://microdatos.dane.gov.co/index.php/catalog/819)**

Módulos necesarios (12 archivos cada uno):
- **Características Generales** → carpeta `2024_data/Caracteristicas_Generales/`
- **Ocupados** → carpeta `2024_data/Ocupados/`

Nomenclatura esperada:
```
1.Caracteristicas_Generales_Enero.csv
2.Caracteristicas_Generales_Febrero.csv
...
1.Ocupados_Enero.csv
2.Ocupados_Febrero.csv
...
```
