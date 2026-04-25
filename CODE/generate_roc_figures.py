"""
Genera Fig6 (ROC PGR-KDE) y Fig8 (ROC PGC) con todas las curvas correctas.
Ejecutar con: /home/edier/miniconda3/bin/python CODE/generate_roc_figures.py
"""
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import roc_curve, roc_auc_score

DATA_DIR = Path('DATA')
FIG_DIR  = Path('FIGURAS')
chunk    = 50_000
RNG      = np.random.default_rng(42)

# ------------------------------------------------------------------ #
# 1. Carga de datos — misma lógica que run_analysis.py STEP 1
# ------------------------------------------------------------------ #
print('Cargando rasters...')
WS_PATH  = str(DATA_DIR / 'Cuenca_Iguana.shp')
INV_PATH = str(DATA_DIR / 'Deslizamientos_Iguana.gpkg')

ws = gpd.read_file(WS_PATH)

with rasterio.open(DATA_DIR / 'DEM_5m.tif') as src:
    dem_crs   = src.crs
    dem_trans = src.transform
    dem_shape = src.shape
    dem_arr   = src.read(1).astype(np.float32)
    dem_arr[dem_arr == src.nodata] = np.nan

ws_proj = ws.to_crs(dem_crs)
ws_mask = geometry_mask(ws_proj.geometry, transform=dem_trans,
                         invert=True, out_shape=dem_shape)

def load_resample(name, categorical=False):
    with rasterio.open(str(DATA_DIR / f'{name}.tif')) as src:
        dest = np.zeros(dem_shape, np.float32)
        reproject(rasterio.band(src, 1), dest,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dem_trans, dst_crs=dem_crs,
                  resampling=Resampling.nearest if categorical else Resampling.bilinear)
        dest[dest == src.nodata] = np.nan
    return dest

def rasterize_gpkg(path, column):
    gdf     = gpd.read_file(path).to_crs(dem_crs)
    ws_geom = ws.union_all()
    gdf_in  = gdf[gdf.intersects(ws_geom)]
    shapes  = ((geom, val) for geom, val in zip(gdf_in.geometry, gdf_in[column]))
    raster  = rasterize(shapes, out_shape=dem_shape, transform=dem_trans,
                        fill=0, all_touched=True, dtype=np.float32)
    raster[~ws_mask] = np.nan
    return raster

slope     = load_resample('slope_5m')
aspect    = load_resample('aspect_5m')
twi       = load_resample('twi_5m')
geology   = rasterize_gpkg(DATA_DIR / 'geosurface_map.gpkg', column='Codigo')
landcover = rasterize_gpkg(DATA_DIR / 'landcover_map.gpkg',  column='code')

valid_mask = ws_mask & ~np.isnan(slope) & ~np.isnan(geology) & ~np.isnan(landcover)
r_valid, c_valid = np.where(valid_mask)
n_valid = valid_mask.sum()
print(f'  Celdas válidas: {n_valid:,}')

# ------------------------------------------------------------------ #
# 2. Inventario → etiquetas binarias
# ------------------------------------------------------------------ #
inv      = gpd.read_file(INV_PATH).to_crs(dem_crs)
ls_r, ls_c = [], []
for geom in inv.geometry:
    if geom is None: continue
    r, c = rowcol(dem_trans, geom.x, geom.y)
    if 0 <= r < dem_shape[0] and 0 <= c < dem_shape[1] and valid_mask[r, c]:
        ls_r.append(r); ls_c.append(c)
label_map = np.zeros(dem_shape, np.int8)
label_map[ls_r, ls_c] = 1
y_all = label_map[r_valid, c_valid]
print(f'  Deslizamientos mapeados: {y_all.sum()}')

# ------------------------------------------------------------------ #
# 3. Cargar KDE (ya calculado)
# ------------------------------------------------------------------ #
with rasterio.open(DATA_DIR / 'kde_density.tif') as src:
    kde_map = src.read(1).astype(np.float32)
    kde_map[kde_map == src.nodata] = np.nan
kde_all = kde_map[r_valid, c_valid]

# ------------------------------------------------------------------ #
# 4. Target encoding de variables categóricas (misma lógica)
# ------------------------------------------------------------------ #
import pandas as pd
df_enc    = pd.DataFrame({'cat': geology[valid_mask],  'ls': y_all})
df_enc_lc = pd.DataFrame({'cat': landcover[valid_mask], 'ls': y_all})

def target_encode(raster, df):
    enc    = df.groupby('cat')['ls'].mean().to_dict()
    mapped = np.vectorize(lambda x: enc.get(x, np.nan))(raster.flatten())
    return mapped.reshape(raster.shape).astype(np.float32)

geo_e = target_encode(geology,   df_enc)
lc_e  = target_encode(landcover, df_enc_lc)

feat_dict = {
    'Pendiente': slope,
    'Aspecto':   aspect,
    'ITH':       twi,
    'Geologia':  geo_e,
    'Cobertura': lc_e,
}

# ------------------------------------------------------------------ #
# 5. Conjunto de entrenamiento balanceado (250+250, seed=42)
# ------------------------------------------------------------------ #
idx_ls    = np.where(y_all == 1)[0]
idx_bg    = np.where(y_all == 0)[0]
idx_train = np.concatenate([RNG.choice(idx_ls, min(250, len(idx_ls)), replace=False),
                             RNG.choice(idx_bg, 250, replace=False)])
y_tr    = y_all[idx_train]
kde_tr  = kde_all[idx_train]
print(f'  Entrenamiento: {len(idx_train)} puntos ({y_tr.sum()} positivos)')

# ------------------------------------------------------------------ #
# 6. PGR-KDE: una curva por variable + benchmark
# ------------------------------------------------------------------ #
def roc_pgr(var_names, label):
    X = np.column_stack([feat_dict[n][r_valid, c_valid] for n in var_names])
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X[idx_train])
    Xf  = sc.transform(X)
    gpr = GaussianProcessRegressor(
        kernel=1.0**2 * Matern(length_scale=[1.0]*len(var_names), nu=1.5),
        alpha=0.01, n_restarts_optimizer=1, normalize_y=True)
    gpr.fit(Xtr, kde_tr)
    pred = np.zeros(len(Xf), dtype=np.float32)
    for i in range(0, len(Xf), chunk):
        pred[i:i+chunk] = gpr.predict(Xf[i:i+chunk])
    pc  = np.clip(pred, 0, None)
    pn  = (pc - pc.min()) / (pc.max() - pc.min() + 1e-12)
    auc = roc_auc_score(y_all, pn)
    fpr, tpr, _ = roc_curve(y_all, pn)
    print(f'  PGR-KDE {label:30s}: AUC={auc:.3f}')
    return fpr, tpr, auc

def roc_pgc(var_names, label):
    X = np.column_stack([feat_dict[n][r_valid, c_valid] for n in var_names])
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X[idx_train])
    Xf  = sc.transform(X)
    gpc = GaussianProcessClassifier(
        kernel=1.0**2 * Matern(length_scale=[1.0]*len(var_names), nu=1.5),
        random_state=42)
    gpc.fit(Xtr, y_tr)
    probs = np.zeros(len(Xf), dtype=np.float32)
    for i in range(0, len(Xf), chunk):
        probs[i:i+chunk] = gpc.predict_proba(Xf[i:i+chunk])[:, 1]
    auc = roc_auc_score(y_all, probs)
    fpr, tpr, _ = roc_curve(y_all, probs)
    print(f'  PGC {label:33s}: AUC={auc:.3f}')
    return fpr, tpr, auc

# ------------------------------------------------------------------ #
# 7. PGR-KDE: todas las variables individuales + benchmark
# ------------------------------------------------------------------ #
print('\n=== PGR-KDE ===')
all_vars = ['Pendiente','Aspecto','ITH','Geologia','Cobertura']

pgr_curves = []
for v in all_vars:
    fpr, tpr, auc = roc_pgr([v], v)
    pgr_curves.append((fpr, tpr, auc, v))
fpr_b, tpr_b, auc_b = roc_pgr(all_vars, 'Benchmark (todas)')
pgr_curves.append((fpr_b, tpr_b, auc_b, 'Benchmark (todas)'))

# ------------------------------------------------------------------ #
# 8. PGC: cuatro configuraciones
# ------------------------------------------------------------------ #
print('\n=== PGC ===')
gpc_configs = [
    (['Pendiente','Aspecto','ITH','Geologia','Cobertura'], 'Benchmark (todas)'),
    (['Pendiente','Aspecto','ITH'],                        'Solo topografía'),
    (['Geologia','Cobertura'],                             'Solo geoespacial'),
    (['Pendiente','Geologia','Cobertura'],                 'Pendiente + Geología + Cobertura'),
]
gpc_curves = []
for var_names, label in gpc_configs:
    fpr, tpr, auc = roc_pgc(var_names, label)
    gpc_curves.append((fpr, tpr, auc, label))

# ------------------------------------------------------------------ #
# 9. Fig6: Curvas ROC PGR-KDE (sin título)
# ------------------------------------------------------------------ #
print('\nGenerando Fig6...')
colors_pgr = ['#2166ac','#4dac26','#d01c8b','#f1a340','#998ec3','#e08214']
linestyles = ['-','-','-','-','-','--']

fig, ax = plt.subplots(figsize=(9, 7))
for (fpr, tpr, auc, label), col, ls in zip(pgr_curves, colors_pgr, linestyles):
    ax.plot(fpr, tpr, color=col, linewidth=2.0, linestyle=ls,
            label=f'{label} (AUC = {auc:.3f})')
ax.plot([0,1],[0,1], 'gray', linestyle=':', alpha=0.5, linewidth=1)
ax.legend(fontsize=9.5, loc='lower right', framealpha=0.9)
ax.set_xlabel('Tasa de Falsos Positivos', fontsize=14)
ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig6_roc_pgr_kde.png', dpi=300)
plt.close()
print('  Fig6_roc_pgr_kde.png guardada')

# ------------------------------------------------------------------ #
# 10. Fig8: Curvas ROC PGC (sin título)
# ------------------------------------------------------------------ #
print('Generando Fig8...')
colors_gpc    = ['#1b7837','#762a83','#e08214','#d6604d']
linestyles_gpc = ['--','-.', ':', '-']

fig, ax = plt.subplots(figsize=(9, 7))
for (fpr, tpr, auc, label), col, ls in zip(gpc_curves, colors_gpc, linestyles_gpc):
    ax.plot(fpr, tpr, color=col, linewidth=2.0, linestyle=ls,
            label=f'{label} (AUC = {auc:.3f})')
ax.plot([0,1],[0,1], 'gray', linestyle=':', alpha=0.5, linewidth=1)
ax.legend(fontsize=9.5, loc='lower right', framealpha=0.9)
ax.set_xlabel('Tasa de Falsos Positivos', fontsize=14)
ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig8_roc_pgc.png', dpi=300)
plt.close()
print('  Fig8_roc_pgc.png guardada')
print('\nListo.')
