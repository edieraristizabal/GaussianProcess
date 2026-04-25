"""
resample_categorical_5m.py
==========================
Rasteriza los vectores de geología y cobertura del suelo (GPKG con 5 clases
cada uno) directamente al grid exacto de DEM_5m.tif (MAGNA-SIRGAS @ 5m).

Fuentes:
  geosurface_map.gpkg  campo 'Codigo'  → geology_5m.tif   (clases 1-5)
  landcover_map.gpkg   campo 'code'    → landcover_5m.tif  (clases 1-5)

Las leyendas están en geosurface_legend.txt y landcover_lookup.txt.
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.crs import CRS
import geopandas as gpd
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
DATA = Path('/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/'
            'GaussianProcess/DATA')

REF_GRID  = DATA / 'DEM_5m.tif'
GEO_GPKG  = DATA / 'geosurface_map.gpkg'
LC_GPKG   = DATA / 'landcover_map.gpkg'
OUT_GEO   = DATA / 'geology_5m.tif'
OUT_LC    = DATA / 'landcover_5m.tif'

NODATA_OUT = -9999

# ── Leer parámetros del grid de referencia ────────────────────────────────────
with rasterio.open(REF_GRID) as ref:
    dst_crs       = ref.crs
    dst_transform = ref.transform
    dst_shape     = ref.shape          # (n_rows, n_cols)
    ws_mask       = ref.read(1) != ref.nodata

print(f'Grid de referencia: {dst_shape[0]}r × {dst_shape[1]}c  res=5m\n')

profile = {
    'driver': 'GTiff',
    'dtype': 'int16',
    'width': dst_shape[1],
    'height': dst_shape[0],
    'count': 1,
    'crs': dst_crs,
    'transform': dst_transform,
    'nodata': NODATA_OUT,
    'compress': 'lzw',
    'tiled': True,
    'blockxsize': 256,
    'blockysize': 256,
}

# ── Función de rasterización ──────────────────────────────────────────────────
def rasterize_vector(gpkg_path, field, dst_path, label):
    gdf = gpd.read_file(gpkg_path)
    print(f'[{label}]')
    print(f'  Fuente: {gpkg_path.name}  campo="{field}"  CRS={gdf.crs}')
    print(f'  Clases en vector: {sorted(gdf[field].unique())}')

    # Reprojectar al CRS destino
    gdf_proj = gdf.to_crs(dst_crs)

    # Crear pares (geometría, valor)
    shapes = [(geom, int(val)) for geom, val in zip(gdf_proj.geometry, gdf_proj[field])]

    # Rasterizar (fill=NODATA donde no hay polígono)
    arr = rasterize(
        shapes,
        out_shape=dst_shape,
        transform=dst_transform,
        fill=NODATA_OUT,
        dtype=np.int16,
        all_touched=False,
    )

    # Aplicar máscara de cuenca
    arr[~ws_mask] = NODATA_OUT

    valid   = arr[arr != NODATA_OUT]
    classes = np.unique(valid)
    print(f'  Celdas válidas: {len(valid):,}  Clases rasterizadas: {classes}')

    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(arr, 1)
    print(f'  Guardado: {dst_path}\n')


# ── Ejecutar ──────────────────────────────────────────────────────────────────
rasterize_vector(GEO_GPKG, 'Codigo', OUT_GEO, 'geology')
rasterize_vector(LC_GPKG,  'code',   OUT_LC,  'landcover')

# ── Verificación final ────────────────────────────────────────────────────────
print('=== Verificación final ===')
for f in [OUT_GEO, OUT_LC]:
    with rasterio.open(f) as src:
        arr  = src.read(1)
        mask = arr != src.nodata
        print(f'  {f.name:20s}  shape={src.shape}  res={src.res}  '
              f'válidas={mask.sum():,}  clases={np.unique(arr[mask])}')

print('\nListo.')
