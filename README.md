# Predicción de Informalidad Laboral en Colombia

Modelo de clasificación binaria que estima la probabilidad de informalidad laboral de un trabajador colombiano a partir de sus características socioeconómicas, laborales y geográficas, entrenado sobre los microdatos de la Gran Encuesta Integrada de Hogares (GEIH) 2025 del DANE.

**Proyecto integrador** — Ingeniería de Sistemas — EAFIT — 2025

---

## Descripción del problema

El 58% de los ocupados en Colombia trabaja en condiciones de informalidad. Los programas de formalización laboral del Estado se diseñan con estadísticas agregadas a nivel departamental o sectorial, lo que limita su focalización hacia los perfiles de mayor riesgo. Este proyecto construye un modelo predictivo a nivel de individuo para apoyar esa focalización.

**Variable objetivo:** `INFORMAL` — 1 si el trabajador no cotiza a salud por su relación laboral (P6920 = 2), 0 si es cotizante (P6920 = 1).

---

## Datos

### Descarga

Los microdatos de la GEIH 2025 son públicos y de descarga gratuita:

1. Ir a: https://microdatos.dane.gov.co/index.php/catalog/GEIH2025
2. Descargar los módulos **Características Generales** y **Ocupados** (12 archivos mensuales cada uno)
3. Organizarlos en Google Drive con esta estructura:

```
Mi unidad/
└── GEIH_2025/
    ├── Caracteristicas_Generales/
    │   ├── 1_Caracteristicas_Generales_Enero.csv
    │   └── ...
    └── Ocupados/
        ├── 1_Ocupados_Enero.csv
        └── ...
```

### Variables principales

| Variable | Módulo | Descripción |
|---|---|---|
| `P6920` | Ocupados | Variable objetivo: 1=Cotizante, 2=No cotizante |
| `P3271` | Generales | Sexo (1=Hombre, 2=Mujer) |
| `P6040` | Generales | Edad en años cumplidos |
| `P3042` | Generales | Nivel educativo (1=Ninguno a 13=Doctorado) |
| `DPTO` | Generales | Código de departamento (33 dominios DANE) |
| `CLASE` | Generales | Zona (1=Cabecera, 2=Rural) |
| `P6430` | Ocupados | Posición ocupacional |
| `P3069` | Ocupados | Tamaño del establecimiento |
| `RAMA2D_R4` | Ocupados | Rama de actividad (CIIU 2 dígitos) |
| `FEX_C18` | Generales | Factor de expansión estadística |

---