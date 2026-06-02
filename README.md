# Predicción de Informalidad Laboral en Colombia · GEIH 2025

**Maestría en Ciencia de Datos y Analítica** · SI7006 · SI7007 · SI7009  
Juan Andrés Montoya · Julián David Mejía

> **Pregunta de investigación:** ¿Por qué los trabajadores independientes en Colombia permanecen informales, y qué condiciones deben cambiar para que la formalización sea una opción viable para ellos?

---

## Dashboard en producción

El dashboard está desplegado públicamente en Streamlit Community Cloud — no requiere instalar nada:

**[→ Abrir dashboard](https://geih-informalidad.streamlit.app)**

---

## Resultados del modelo

| Modelo | F1-Score | AUC-ROC | Precisión | Recall |
|--------|----------|---------|-----------|--------|
| **LightGBM (campeón)** | **0.9576** | **0.9853** | 0.9399 | 0.9760 |
| Random Forest | 0.9553 | 0.9818 | 0.9378 | 0.9734 |
| XGBoost | 0.9546 | 0.9825 | 0.9423 | 0.9673 |
| Logistic Regression | 0.9507 | 0.9738 | 0.9347 | 0.9673 |

- **Umbral de decisión:** 0.41 (calibrado por threshold tuning)
- **Predictor más importante (SHAP):** Años de educación
- **Población objetivo:** Trabajadores independientes (cuenta propia · P6430 = 4)
- **Tasa de informalidad ponderada GEIH 2024:** 84.5% (entre independientes)

---

## Estructura del proyecto

```
Proyecto_Final_maestria/
│
├── app.py                              ← Dashboard Streamlit (5 tabs)
├── requirements.txt                    ← Dependencias Python
├── informe_final.tex                   ← Documento de entrega (LaTeX)
├── geih_2024.duckdb                    ← Base de datos DuckDB
│
├── 1.Ingesta.ipynb                     ← Fase 1: carga y unión de módulos GEIH
├── 2.EDA.ipynb                         ← Fase 2: análisis exploratorio
├── 3.Preparacion.ipynb                 ← Fase 3: feature engineering + pipeline
├── 4.Modelamiento.ipynb                ← Fase 4: entrenamiento y selección
├── 5.Visualizacion.ipynb               ← Fase 5: gráficas Plotly complementarias
│
├── parquet/                            ← Artefactos del pipeline
│   ├── preprocessor.joblib             ← Pipeline ColumnTransformer serializado
│   └── geih_2024_crudo.parquet         ← Dataset crudo (generado por notebook 1)
│
├── outputs/                            ← Artefactos del modelo (en el repo)
│   ├── champion_geih.pkl               ← Modelo LightGBM serializado (48 MB)
│   ├── champion_geih_meta.json         ← Métricas y umbral óptimo
│   ├── datos_procesados.parquet        ← Dataset para el dashboard
│   ├── metrics_comparison.csv          ← Comparativa de los 4 modelos
│   ├── confusion_matrix.png
│   ├── model_comparativa.png
│   ├── shap_summary.png
│   └── shap_importance.png
│
└── 2024_data/                          ← Datos crudos GEIH (NO en GitHub)
    ├── Caracteristicas_Generales/      ← 12 CSV mensuales
    └── Ocupados/                       ← 12 CSV mensuales
```

---

## Correr el dashboard localmente

### 1. Clonar el repositorio

```bash
git clone https://github.com/JuanMG1211/GEIH_Informalidad.git
cd GEIH_Informalidad
```

### 2. Crear entorno virtual e instalar dependencias

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Lanzar el dashboard

```bash
python -m streamlit run app.py
```

Se abre en **http://localhost:8501**

> Los artefactos del modelo (`outputs/` y `parquet/preprocessor.joblib`) ya están en el repositorio — no es necesario ejecutar ningún notebook para ver el dashboard.

---

## Recomendaciones IA (opcional)

El tab **Predicción Individual** puede generar recomendaciones de política pública personalizadas usando Llama 3.3 70B vía Groq. Para habilitarlo:

1. Crea una cuenta gratuita en [console.groq.com](https://console.groq.com)
2. Genera una API Key (`gsk_...`)
3. Pégala en el panel lateral del dashboard

---

## Reproducir el pipeline completo desde cero

Solo necesario si quieres reentrenar el modelo con nuevos datos.

**Requiere:** carpeta `2024_data/` con los 24 CSV del GEIH 2024 descargados desde [microdatos.dane.gov.co](https://microdatos.dane.gov.co/index.php/catalog/819).

Ejecuta los notebooks **en orden**:

| Notebook | Genera | Tiempo estimado |
|----------|--------|-----------------|
| `1.Ingesta.ipynb` | `parquet/geih_2024_crudo.parquet` · `geih_2024.duckdb` | 2–5 min |
| `2.EDA.ipynb` | Solo visualizaciones (no escribe archivos) | 1–2 min |
| `3.Preparacion.ipynb` | `parquet/train.parquet` · `test.parquet` · `preprocessor.joblib` | 2–5 min |
| `4.Modelamiento.ipynb` | `outputs/champion_geih.pkl` · métricas · imágenes SHAP | 5–15 min |
| `5.Visualizacion.ipynb` | Gráficas HTML complementarias (no requeridas por el dashboard) | 1–3 min |

```bash
jupyter notebook
```

Luego reinicia el dashboard para que tome los artefactos actualizados.

---

## Dependencias principales

```
streamlit>=1.35.0
pandas>=2.2.0
numpy>=1.26.0
plotly>=5.18.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
joblib>=1.3.0
pyarrow>=15.0.0
shap>=0.44.0
category-encoders>=2.6.0
groq>=0.9.0
```

---

## Solución de problemas

| Error | Causa | Solución |
|-------|-------|----------|
| `ModuleNotFoundError: category_encoders` | Paquete no instalado | `pip install category-encoders` |
| `ModuleNotFoundError: shap` | Paquete no instalado | `pip install shap` |
| `FileNotFoundError: champion_geih.pkl` | Clonaste sin LFS / falta `outputs/` | `git pull` o ejecuta `4.Modelamiento.ipynb` |
| `FileNotFoundError: preprocessor.joblib` | Falta `parquet/` | `git pull` o ejecuta `3.Preparacion.ipynb` |
| `UnicodeDecodeError` en Fase 1 | Encoding CSV GEIH | Los CSV usan `latin-1` — no modificar el parámetro `encoding` |
| Notebook Fase 4 tarda mucho | Normal — early stopping ~858 iter | Espera 5–15 min según hardware |

---

## Fuente de datos

**[DANE — Gran Encuesta Integrada de Hogares 2024](https://microdatos.dane.gov.co/index.php/catalog/819)**

Los archivos CSV crudos no están en el repositorio (~180 MB sin comprimir). Módulos necesarios:
- **Características Generales** → `2024_data/Caracteristicas_Generales/`
- **Ocupados** → `2024_data/Ocupados/`

> El DANE no avala los resultados del análisis.
