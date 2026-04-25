"""
prepare_5m_covariates.py
========================
Prepara todas las variables continuas del DEM a 5m de resolución
para la Cuenca Iguana, partiendo del DEM de 1m de Medellín.

Salidas en DATA/:
  DEM_5m.tif        – DEM recortado y remuestreado a 5m
  slope_5m.tif      – Pendiente (grados) a 5m
  aspect_5m.tif     – Aspecto (grados, 0-360) a 5m
  twi_5m.tif        – Índice Topográfico de Humedad (TWI) a 5m

Las variables categóricas (geology.tif, landcover.tif) NO se tocan;
se remuestrearán automáticamente al grid de 5m en run_analysis.py.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from rasterio.crs import CRS
import geopandas as gpd
from pathlib import Path
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ────────────────────────────────────────────────────────────────────
DEM_1M   = Path('/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/'
                'NHPP-movimientos-masa-VdeA/DATA/dem_medellin_1m.tif')
GP_DATA  = Path('/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/'
                'GaussianProcess/DATA')
WS_PATH  = GP_DATA / 'Cuenca_Iguana.shp'

OUT_DEM  = GP_DATA / 'DEM_5m.tif'
OUT_SLP  = GP_DATA / 'slope_5m.tif'
OUT_ASP  = GP_DATA / 'aspect_5m.tif'
OUT_TWI  = GP_DATA / 'twi_5m.tif'

TARGET_RES = 5.0  # metros

# ── CRS de referencia (del DEM de 10m existente) ────────────────────────────
with rasterio.open(GP_DATA / 'DEM_10m.tif') as src10:
    REF_CRS = src10.crs          # MAGNA-SIRGAS 2018 Origen Nacional
    ref_bounds = src10.bounds

print(f'CRS de referencia: {REF_CRS.to_string()[:60]}')
print(f'Extent cuenca 10m: {ref_bounds}')

# ── PASO 1: Crear DEM_5m clipeado a la cuenca ────────────────────────────────
print('\nPASO 1: Creando DEM_5m.tif...')

ws = gpd.read_file(WS_PATH)
ws_proj = ws.to_crs(REF_CRS)

# Calcular transform de salida a 5m usando el mismo extent del DEM 10m
left, bottom, right, top = ref_bounds
n_cols = int(np.ceil((right - left) / TARGET_RES))
n_rows = int(np.ceil((top - bottom) / TARGET_RES))
# Ajustar bounds para que sean múltiplos exactos de 5m
right_adj = left + n_cols * TARGET_RES
top_adj   = bottom + n_rows * TARGET_RES
dst_transform = rasterio.transform.from_bounds(left, bottom, right_adj, top_adj,
                                               n_cols, n_rows)
dst_shape = (n_rows, n_cols)

print(f'  Grid 5m: {n_rows} filas × {n_cols} columnas ({n_rows*n_cols:,} celdas)')

# Remuestrear 1m DEM → 5m con promedio (average)
with rasterio.open(DEM_1M) as src:
    src_crs = src.crs
    dem_5m = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=dem_5m,
        src_transform=src.transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=REF_CRS,
        resampling=Resampling.average,
        src_nodata=src.nodata,
        dst_nodata=-9999.0
    )

# Aplicar máscara de cuenca
ws_mask = geometry_mask(ws_proj.geometry, transform=dst_transform,
                        invert=True, out_shape=dst_shape)
dem_5m[~ws_mask] = -9999.0
# Reemplazar nodata por nan para cálculos
dem_work = dem_5m.copy().astype(np.float64)
dem_work[dem_work == -9999.0] = np.nan

valid_cells = np.sum(~np.isnan(dem_work))
print(f'  Celdas válidas en cuenca: {valid_cells:,}')

# Guardar DEM_5m
profile = {
    'driver': 'GTiff', 'dtype': 'float32', 'width': n_cols, 'height': n_rows,
    'count': 1, 'crs': REF_CRS, 'transform': dst_transform,
    'nodata': -9999.0, 'compress': 'lzw', 'tiled': True, 'blockxsize': 256, 'blockysize': 256
}
with rasterio.open(OUT_DEM, 'w', **profile) as dst:
    dst.write(dem_5m, 1)
print(f'  Guardado: {OUT_DEM}')

# ── PASO 2: Pendiente (slope) a 5m ──────────────────────────────────────────
print('\nPASO 2: Calculando pendiente (slope_5m)...')

# Usar gdaldem para cálculo estándar (Horn's formula)
ret = subprocess.run([
    '/usr/bin/gdaldem', 'slope',
    str(OUT_DEM), str(OUT_SLP),
    '-compute_edges',
    '-co', 'COMPRESS=LZW',
    '-co', 'TILED=YES',
    '-co', 'BLOCKXSIZE=256',
    '-co', 'BLOCKYSIZE=256',
    '-of', 'GTiff'
], capture_output=True, text=True)

if ret.returncode != 0:
    # Fallback: cálculo manual con numpy
    print('  gdaldem falló, usando cálculo manual...')
    grad_y, grad_x = np.gradient(
        np.where(np.isnan(dem_work), 0.0, dem_work), TARGET_RES, TARGET_RES)
    # Evitar gradientes espurios en bordes nodata
    nan_pad = np.isnan(dem_work)
    grad_y[nan_pad] = np.nan
    grad_x[nan_pad] = np.nan
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    slope_deg[~ws_mask] = -9999.0
    with rasterio.open(OUT_SLP, 'w', **profile) as dst:
        dst.write(slope_deg, 1)
else:
    # Aplicar máscara de cuenca al resultado de gdaldem
    with rasterio.open(OUT_SLP) as src:
        slp = src.read(1).astype(np.float32)
        slp_nodata = src.nodata
    slp[~ws_mask] = -9999.0
    with rasterio.open(OUT_SLP, 'w', **profile) as dst:
        dst.write(slp, 1)

print(f'  Guardado: {OUT_SLP}')

# ── PASO 3: Aspecto (aspect) a 5m ───────────────────────────────────────────
print('\nPASO 3: Calculando aspecto (aspect_5m)...')

ret = subprocess.run([
    '/usr/bin/gdaldem', 'aspect',
    str(OUT_DEM), str(OUT_ASP),
    '-compute_edges',
    '-zero_for_flat',
    '-co', 'COMPRESS=LZW',
    '-co', 'TILED=YES',
    '-co', 'BLOCKXSIZE=256',
    '-co', 'BLOCKYSIZE=256',
    '-of', 'GTiff'
], capture_output=True, text=True)

if ret.returncode != 0:
    print('  gdaldem falló, usando cálculo manual...')
    grad_y, grad_x = np.gradient(
        np.where(np.isnan(dem_work), 0.0, dem_work), TARGET_RES, TARGET_RES)
    aspect_rad = np.arctan2(-grad_y, grad_x)
    aspect_deg = (90.0 - np.degrees(aspect_rad)) % 360
    aspect_deg = aspect_deg.astype(np.float32)
    aspect_deg[~ws_mask] = -9999.0
    with rasterio.open(OUT_ASP, 'w', **profile) as dst:
        dst.write(aspect_deg, 1)
else:
    with rasterio.open(OUT_ASP) as src:
        asp = src.read(1).astype(np.float32)
    asp[~ws_mask] = -9999.0
    with rasterio.open(OUT_ASP, 'w', **profile) as dst:
        dst.write(asp, 1)

print(f'  Guardado: {OUT_ASP}')

# ── PASO 4: TWI (Topographic Wetness Index) a 5m ────────────────────────────
print('\nPASO 4: Calculando TWI (twi_5m)...')
print('  Usando pysheds para dirección de flujo y área acumulada...')

try:
    from pysheds.grid import Grid
    import tempfile, os

    # pysheds necesita un archivo limpio (sin nodata en bordes)
    # Preparar DEM rellenando nodata con un valor bajo para que el flujo salga de la cuenca
    dem_twi = dem_work.copy()
    # Rellenar áreas fuera de la cuenca con el mínimo - 1 para forzar salida de flujo
    min_val = np.nanmin(dem_twi) - 100.0
    dem_twi[np.isnan(dem_twi)] = min_val

    # Guardar DEM temporal para pysheds
    tmp_dem = str(GP_DATA / '_tmp_dem_pysheds.tif')
    tmp_prof = profile.copy()
    tmp_prof['nodata'] = min_val - 1
    dem_twi_f32 = dem_twi.astype(np.float32)
    dem_twi_f32[~ws_mask] = float(tmp_prof['nodata'])
    with rasterio.open(tmp_dem, 'w', **tmp_prof) as dst:
        dst.write(dem_twi_f32, 1)

    # Cargar con pysheds
    grid = Grid.from_raster(tmp_dem)
    dem_ps = grid.read_raster(tmp_dem)

    # Acondicionar DEM
    pit_filled = grid.fill_pits(dem_ps)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)

    # Dirección de flujo D8
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated, dirmap=dirmap)

    # Acumulación de flujo
    acc = grid.accumulation(fdir, dirmap=dirmap)
    acc_arr = np.array(acc).astype(np.float64)

    # Pendiente en radianes desde DEM
    dem_smooth = np.array(inflated).astype(np.float64)
    dem_smooth[~ws_mask] = np.nan
    grad_y, grad_x = np.gradient(dem_smooth, TARGET_RES, TARGET_RES)
    slope_rad_twi = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    # Mínimo de 0.0001 rad (~0.006°) para evitar log(inf)
    slope_rad_twi = np.where(ws_mask, np.maximum(slope_rad_twi, 0.0001), np.nan)

    # TWI = ln(a / tan(β))  donde a = área específica = (acc+1) * celda
    sca = (acc_arr + 1.0) * TARGET_RES   # specific catchment area (m)
    sca[~ws_mask] = np.nan
    twi_arr = np.log(sca / np.tan(slope_rad_twi))
    twi_arr[~ws_mask] = -9999.0
    twi_arr = twi_arr.astype(np.float32)

    with rasterio.open(OUT_TWI, 'w', **profile) as dst:
        dst.write(twi_arr, 1)

    # Limpiar temporal
    os.remove(tmp_dem)
    print(f'  TWI rango: [{np.nanmin(twi_arr[ws_mask]):.2f}, {np.nanmax(twi_arr[ws_mask]):.2f}]')
    print(f'  Guardado: {OUT_TWI}')

except Exception as e:
    print(f'  pysheds error: {e}')
    print('  Fallback: TWI aproximado con acumulación de flujo numpy...')

    # Fallback: TWI simple con gradiente de pendiente
    grad_y, grad_x = np.gradient(
        np.where(np.isnan(dem_work), 0.0, dem_work), TARGET_RES, TARGET_RES)
    slope_rad_fb = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_rad_fb[~ws_mask] = np.nan
    slope_rad_fb = np.maximum(slope_rad_fb, 0.0001)

    # Sin acumulación real, usar TWI simplificado (índice de pendiente)
    # TWI_approx = -ln(tan(slope)) (representación escalar, monotónicamente relacionada)
    twi_fb = -np.log(np.tan(slope_rad_fb))
    twi_fb[~ws_mask] = -9999.0
    twi_fb = twi_fb.astype(np.float32)
    with rasterio.open(OUT_TWI, 'w', **profile) as dst:
        dst.write(twi_fb, 1)
    print(f'  Guardado (TWI aproximado): {OUT_TWI}')

# ── Verificación final ───────────────────────────────────────────────────────
print('\n=== Verificación de salidas ===')
for f in [OUT_DEM, OUT_SLP, OUT_ASP, OUT_TWI]:
    with rasterio.open(f) as src:
        arr = src.read(1)
        mask = arr != src.nodata
        print(f'  {f.name:18s}  shape={src.shape}  res={src.res}  '
              f'válidas={mask.sum():,}  '
              f'rango=[{arr[mask].min():.2f}, {arr[mask].max():.2f}]')

print('\nListo. Ahora ejecuta run_analysis.py para el análisis con 5m.')
