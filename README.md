# Análisis de Susceptibilidad por Deslizamientos usando Procesos Gaussianos

Este repositorio contiene el código, datos y el manuscrito de investigación sobre el modelado de susceptibilidad por deslizamientos en la cuenca de la quebrada La Iguaná (Medellín, Colombia), utilizando **Regresión por Procesos Gaussianos (GPR)**.

## Estructura del Repositorio

- **`CODE/`**: Contiene los scripts de procesamiento y el notebook principal.
  - `landslide_susceptibility_GP.ipynb`: Notebook con el flujo de trabajo completo.
  - `run_analysis.py`: Script para ejecución del modelo.
  - `regenerate_figures.py`: Utilidad para recrear los gráficos del manuscrito.
- **`DATA/`**: Datos geoespaciales utilizados en el modelo.
  - Incluye archivos `.shp`, `.gpkg` y rásteres de covariables (pendiente, aspecto, TWI).
  - *Nota: El archivo `DEM.tif` original ha sido excluido debido a su tamaño.*
- **`FIGURAS/`**: Mapas y gráficos de evaluación del modelo generados durante el análisis.
- **`manuscrito.tex`**: Archivo fuente en LaTeX del artículo científico en preparación.
- **`referencias.bib`**: Base de datos de bibliografía para el manuscrito.

## Metodología

El estudio compara el desempeño de los Procesos Gaussianos frente a métodos tradicionales como la Regresión Logística, destacando la capacidad de GPR para capturar incertidumbre espacial en las predicciones de susceptibilidad.

## Requisitos

- Python 3.8+
- Librerías: `scikit-learn`, `geopandas`, `rasterio`, `matplotlib`, `numpy`.
