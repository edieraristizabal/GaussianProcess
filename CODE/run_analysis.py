# =====================================================================
# Landslide Susceptibility – La Iguana – FINAL PRODUCTION PIPELINE
# =====================================================================
# Enfoque secuencial:
#   1. KDE del inventario → superficie de densidad espacial
#   2. PGR-KDE: Proceso Gaussiano de Regresión sobre densidad KDE
#   3. PGC: Proceso Gaussiano de Clasificación (modelo final)
# =====================================================================
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, roc_curve)
from scipy.stats import gaussian_kde as scipy_kde, pearsonr
import pyproj
from rasterio.plot import plotting_extent
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Config
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'DATA'
FIG_DIR  = BASE_DIR / 'FIGURAS'
FIG_DIR.mkdir(exist_ok=True)
np.random.seed(42)

def add_panel_label(ax, letter):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

def add_cartography(ax, extent, crs):
    transformer_from_wgs84 = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon_breaks = np.arange(-75.7, -75.5, 0.02)
    lat_breaks = np.arange(6.24, 6.36, 0.02)
    x_ticks, _ = transformer_from_wgs84.transform(lon_breaks, np.full_like(lon_breaks, 6.3))
    _, y_ticks = transformer_from_wgs84.transform(np.full_like(lat_breaks, -75.6), lat_breaks)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f'{abs(l):.2f}°W' for l in lon_breaks], fontsize=12)
    ax.set_yticklabels([f'{l:.2f}°N' for l in lat_breaks], fontsize=12)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88),
                arrowprops=dict(facecolor='black', width=3, headwidth=10),
                ha='center', va='center', fontsize=14, fontweight='bold', xycoords='axes fraction')
    scale_len_m = 2000
    scale_x = extent[0] + (extent[1] - extent[0]) * 0.4
    scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
    ax.plot([scale_x, scale_x + scale_len_m], [scale_y, scale_y], color='black', linewidth=3)
    ax.text(scale_x + scale_len_m/2, scale_y + (extent[3] - extent[2])*0.02, '2 km',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# =====================================================================
# STEP 1: Data Loading & Resampling
# =====================================================================
print('STEP 1: Loading & Resampling data...')
DEM_PATH = str(DATA_DIR / 'DEM_5m.tif')
WS_PATH  = str(DATA_DIR / 'Cuenca_Iguana.shp')
INV_PATH = str(DATA_DIR / 'Deslizamientos_Iguana.gpkg')

ws = gpd.read_file(WS_PATH)
with rasterio.open(DEM_PATH) as src:
    dem_crs = src.crs; dem_trans = src.transform; dem_shape = src.shape; dem_prof = src.profile
    dem_arr = src.read(1).astype(np.float32)
    dem_arr[dem_arr == src.nodata] = np.nan

ws_proj = ws.to_crs(dem_crs)
ws_mask = geometry_mask(ws_proj.geometry, transform=dem_trans, invert=True, out_shape=dem_shape)

def load_resample(name, categorical=False):
    with rasterio.open(str(DATA_DIR / f'{name}.tif')) as src:
        dest = np.zeros(dem_shape, np.float32)
        reproject(rasterio.band(src, 1), dest, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dem_trans, dst_crs=dem_crs,
                  resampling=Resampling.nearest if categorical else Resampling.bilinear)
        dest[dest == src.nodata] = np.nan
        return dest

def rasterize_gpkg(path, column):
    map_gdf = gpd.read_file(path).to_crs(dem_crs)
    ws_geom = ws.union_all()
    map_in_ws = map_gdf[map_gdf.intersects(ws_geom)]
    shapes = ((geom, val) for geom, val in zip(map_in_ws.geometry, map_in_ws[column]))
    raster = rasterize(
        shapes, out_shape=dem_shape, transform=dem_trans,
        fill=0, all_touched=True, dtype=np.float32)
    raster[~ws_mask] = np.nan
    return raster

slope     = load_resample('slope_5m')
aspect    = load_resample('aspect_5m')
twi       = load_resample('twi_5m')
geology   = rasterize_gpkg(DATA_DIR / 'geosurface_map.gpkg', column='Codigo')
landcover = rasterize_gpkg(DATA_DIR / 'landcover_map.gpkg', column='code')

valid_mask = ws_mask & ~np.isnan(slope) & ~np.isnan(geology) & ~np.isnan(landcover)
inv = gpd.read_file(INV_PATH, layer='Deslizamientos_Iguana')
inv_proj = inv.to_crs(dem_crs)
ls_r, ls_c = [], []
for geom in inv_proj.geometry:
    r, c = rowcol(dem_trans, geom.x, geom.y)
    if 0 <= r < dem_shape[0] and 0 <= c < dem_shape[1] and valid_mask[r, c]:
        ls_r.append(r); ls_c.append(c)
label_map = np.zeros(dem_shape, np.int8); label_map[ls_r, ls_c] = 1

# Extent cartografico (requerido por todas las figuras de mapa)
ext = plotting_extent(np.zeros(dem_shape), dem_trans)
# Colormap susceptibilidad
cmap_susc = LinearSegmentedColormap.from_list('susc',
    ['#238b45','#74c476','#ffeda0','#feb24c','#e31a1c'])
chunk = 50000
print(f'  Celdas validas: {valid_mask.sum():,}  |  Deslizamientos: {len(ls_r)}')

# =====================================================================
# STEP 2: Exploratory Analysis (Correlation & Distributions)
# =====================================================================
print('STEP 2: Exploratory Analysis...')
r_valid, c_valid = np.where(valid_mask)
df_all = pd.DataFrame({
    'Slope': slope[r_valid, c_valid], 'Aspect': aspect[r_valid, c_valid],
    'TWI': twi[r_valid, c_valid], 'Geology': geology[r_valid, c_valid],
    'Landcover': landcover[r_valid, c_valid], 'Label': label_map[r_valid, c_valid]
})

plt.figure(figsize=(8, 6))
df_corr = df_all[['Slope','Aspect','TWI']].rename(
    columns={'Slope':'Pendiente','Aspect':'Aspecto','TWI':'ITH'})
corr = df_corr.corr(method='spearman')
ax = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
                 annot_kws={"size": 24}, cbar=False)
cbar = plt.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.04,
                    ticks=[-1, -0.5, 0, 0.5, 1])
cbar.set_label('Coeficiente de Spearman', fontsize=21)
cbar.ax.tick_params(labelsize=18)
plt.xticks(fontsize=21); plt.yticks(fontsize=21)
plt.tight_layout(); plt.savefig(FIG_DIR / 'Fig2_correlation.png', dpi=300); plt.close()

cont_vars = [('Slope','Pendiente'), ('Aspect','Aspecto'), ('TWI','ITH')]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (var, name) in zip(axes, cont_vars):
    sns.kdeplot(data=df_all, x=var, hue='Label', common_norm=False, fill=True,
                palette=['steelblue','firebrick'], alpha=0.5, ax=ax, legend=False)
    ax.set_xlabel(name, fontsize=21); ax.set_ylabel('Indice de Frecuencia', fontsize=21)
    ax.tick_params(labelsize=18)
fig.legend(['Cuenca','Deslizamiento'], loc='upper center',
           bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=18)
plt.tight_layout(); plt.savefig(FIG_DIR / 'Fig3_distributions_continuous.png', dpi=300); plt.close()

cat_vars = [('Geology','Geologia'), ('Landcover','Cobertura')]
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (var, name) in zip(axes, cat_vars):
    counts = df_all.groupby([var,'Label']).size().unstack(fill_value=0)
    counts_prop = counts.div(counts.sum(axis=0), axis=1)
    counts_prop.plot(kind='bar', ax=ax, color=['steelblue','firebrick'], alpha=0.8)
    ax.set_xlabel(name, fontsize=21); ax.set_ylabel('Proporcion (Indice de Frecuencia)', fontsize=21)
    ax.tick_params(labelsize=15); ax.legend(['Cuenca','Deslizamiento'], fontsize=18)
plt.tight_layout(); plt.savefig(FIG_DIR / 'Fig4_distributions_categorical.png', dpi=300); plt.close()
print('  Figs 2-4 guardadas.')

# =====================================================================
# STEP 3: KDE – Mapa de densidad del inventario
# =====================================================================
print('STEP 3: Computing KDE density map...')

# Coordenadas proyectadas de los deslizamientos (centro del pixel)
ls_x = np.array([dem_trans.c + (c + 0.5) * dem_trans.a for c in ls_c])
ls_y = np.array([dem_trans.f + (r + 0.5) * dem_trans.e for r in ls_r])
ls_coords_2d = np.vstack([ls_x, ls_y])

# Coordenadas de todas las celdas validas
all_x = np.array([dem_trans.c + (c + 0.5) * dem_trans.a for c in c_valid])
all_y = np.array([dem_trans.f + (r + 0.5) * dem_trans.e for r in r_valid])
all_coords_2d = np.vstack([all_x, all_y])

print(f'  KDE: {len(ls_x)} puntos, {len(c_valid):,} celdas a evaluar...')
kde_func = scipy_kde(ls_coords_2d, bw_method='silverman')
bw_m = kde_func.factor * np.std(ls_x)
print(f'  Ancho de banda Silverman aprox. {bw_m:.0f} m')

kde_all = np.zeros(len(c_valid), dtype=np.float32)
for i in range(0, len(c_valid), chunk):
    kde_all[i:i+chunk] = kde_func(all_coords_2d[:, i:i+chunk])
    if (i // chunk) % 5 == 0 and i > 0:
        print(f'    KDE {min(i+chunk, len(c_valid)):,}/{len(c_valid):,}...')

kde_all = (kde_all - kde_all.min()) / (kde_all.max() - kde_all.min())
kde_map = np.full(dem_shape, np.nan, np.float32)
kde_map[r_valid, c_valid] = kde_all
print(f'  Rango KDE normalizado: {kde_all.min():.4f} - {kde_all.max():.4f}')

kde_prof = dem_prof.copy(); kde_prof.update(dtype='float32', nodata=-9999)
arr_save = kde_map.copy(); arr_save[np.isnan(arr_save)] = -9999
with rasterio.open(str(DATA_DIR / 'kde_density.tif'), 'w', **kde_prof) as dst:
    dst.write(arr_save.astype(np.float32), 1)

cmap_kde = LinearSegmentedColormap.from_list('kde',
    ['#f7fbff','#c6dbef','#6baed6','#2171b5','#08306b'])
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(kde_map, cmap=cmap_kde, vmin=0, vmax=1, extent=ext)
cbar = plt.colorbar(im, ax=ax, fraction=0.035)
cbar.set_label('Densidad KDE (normalizada)', fontsize=16)
cbar.ax.tick_params(labelsize=14)
inv_px = [dem_trans.c + (c+0.5)*dem_trans.a for c in ls_c]
inv_py = [dem_trans.f + (r+0.5)*dem_trans.e for r in ls_r]
ax.scatter(inv_px, inv_py, c='red', s=12, alpha=0.7, label='Deslizamientos', zorder=5)
ax.legend(fontsize=12, loc='lower right')
add_cartography(ax, ext, dem_crs)
plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig5_kde_map.png', dpi=300); plt.close()
print('  Fig5_kde_map.png guardada.')

# =====================================================================
# STEP 4: Codificacion y muestreo balanceado
# =====================================================================
print('STEP 4: Target encoding & balanced sampling...')

def target_encode(raster, df):
    enc = df.groupby('cat')['ls'].mean().to_dict()
    mapped = np.vectorize(enc.get)(raster.flatten())
    return mapped.reshape(raster.shape).astype(np.float32)

df_enc    = pd.DataFrame({'cat': geology[valid_mask],  'ls': label_map[valid_mask]})
geo_e     = target_encode(geology, df_enc)
df_enc_lc = pd.DataFrame({'cat': landcover[valid_mask], 'ls': label_map[valid_mask]})
lc_e      = target_encode(landcover, df_enc_lc)

feat_dict = {'Slope': slope, 'Aspect': aspect, 'TWI': twi,
             'Geology': geo_e, 'Landcover': lc_e}

idx_ls    = np.where(df_all['Label'] == 1)[0]
idx_bg    = np.where(df_all['Label'] == 0)[0]
idx_train = np.concatenate([np.random.choice(idx_ls, 250, replace=False),
                             np.random.choice(idx_bg, 250, replace=False)])
y_true = df_all['Label'].values

# =====================================================================
# STEP 5: PGR-KDE – Regresion sobre densidad KDE (primer enfoque)
# =====================================================================
print('STEP 5: PGR-KDE – Gaussian Process Regression on KDE density...')
names_gpr  = ['Slope', 'Geology', 'Landcover']
X_gpr      = np.column_stack([feat_dict[n][r_valid, c_valid] for n in names_gpr])
scaler_gpr = StandardScaler()
X_tr_gpr   = scaler_gpr.fit_transform(X_gpr[idx_train])
X_f_gpr    = scaler_gpr.transform(X_gpr)

# Target: densidad KDE normalizada en los puntos de entrenamiento
kde_targets = kde_all[idx_train]

gpr_kde = GaussianProcessRegressor(
    kernel=1.0**2 * Matern(length_scale=[1.0]*3, nu=1.5),
    alpha=0.01, n_restarts_optimizer=3, normalize_y=True)
gpr_kde.fit(X_tr_gpr, kde_targets)
print('  PGR-KDE entrenado. Prediciendo cuenca completa...')

gpr_pred    = np.zeros(len(X_f_gpr), dtype=np.float32)
gpr_std_kde = np.zeros(len(X_f_gpr), dtype=np.float32)
for i in range(0, len(X_f_gpr), chunk):
    p, s = gpr_kde.predict(X_f_gpr[i:i+chunk], return_std=True)
    gpr_pred[i:i+chunk]    = p
    gpr_std_kde[i:i+chunk] = s
    if (i // chunk) % 5 == 0 and i > 0:
        print(f'    PGR-KDE {min(i+chunk, len(X_f_gpr)):,}/{len(X_f_gpr):,}...')

gpr_pred_clipped = np.clip(gpr_pred, 0, None)
vmin_g, vmax_g = gpr_pred_clipped.min(), gpr_pred_clipped.max()
gpr_pred_norm = (gpr_pred_clipped - vmin_g) / (vmax_g - vmin_g + 1e-12)

# Guardar rasters PGR-KDE (susceptibilidad y std dev)
for fname, vals in [('gpr_kde_susceptibility.tif', gpr_pred_norm),
                    ('gpr_kde_std.tif',             gpr_std_kde)]:
    arr_map = np.full(dem_shape, np.nan, np.float32)
    arr_map[r_valid, c_valid] = vals
    tmp = arr_map.copy(); tmp[np.isnan(tmp)] = -9999
    with rasterio.open(str(DATA_DIR / fname), 'w', **kde_prof) as dst:
        dst.write(tmp.astype(np.float32), 1)
print('  Rasters PGR-KDE guardados (susceptibilidad + std dev).')

auc_gpr   = roc_auc_score(y_true, gpr_pred_norm)
ap_gpr    = average_precision_score(y_true, gpr_pred_norm)
brier_gpr = brier_score_loss(y_true, gpr_pred_norm)
r_gpr, _  = pearsonr(gpr_pred.astype(float), kde_all.astype(float))
rmse_gpr  = float(np.sqrt(np.mean((gpr_pred - kde_all)**2)))
print(f'  PGR-KDE -> AUC={auc_gpr:.3f} | AP={ap_gpr:.3f} | '
      f'Brier={brier_gpr:.3f} | r(KDE)={r_gpr:.3f} | RMSE={rmse_gpr:.4f}')

gpr_kde_map = np.full(dem_shape, np.nan, np.float32)
gpr_kde_map[r_valid, c_valid] = gpr_pred_norm
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(gpr_kde_map, cmap=cmap_susc, vmin=0, vmax=1, extent=ext)
cbar = plt.colorbar(im, ax=ax, fraction=0.035)
cbar.set_label('Susceptibilidad PGR-KDE (normalizada)', fontsize=16)
cbar.ax.tick_params(labelsize=14)
add_cartography(ax, ext, dem_crs)
plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig7_gpr_susceptibility.png', dpi=300); plt.close()
print('  Fig7_gpr_susceptibility.png guardada.')

# =====================================================================
# STEP 5b: PGR-KDE univariado (una variable a la vez) + Benchmark (5 var)
# =====================================================================
print('STEP 5b: Univariate PGR-KDE models + Benchmark (all 5 vars)...')

def run_pgr_kde(var_names, label):
    """Entrena PGR-KDE con las variables indicadas y devuelve métricas + ROC."""
    if len(var_names) == 1:
        X_v = feat_dict[var_names[0]][r_valid, c_valid].reshape(-1, 1)
    else:
        X_v = np.column_stack([feat_dict[n][r_valid, c_valid] for n in var_names])
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_v[idx_train])
    Xf  = sc.transform(X_v)
    ls0 = [1.0] * len(var_names)
    gpr = GaussianProcessRegressor(
        kernel=1.0**2 * Matern(length_scale=ls0, nu=1.5),
        alpha=0.01, n_restarts_optimizer=3, normalize_y=True)
    gpr.fit(Xtr, kde_all[idx_train])
    pred = np.zeros(len(Xf), dtype=np.float32)
    for i in range(0, len(Xf), chunk):
        pred[i:i+chunk] = gpr.predict(Xf[i:i+chunk])
    pc   = np.clip(pred, 0, None)
    pn   = (pc - pc.min()) / (pc.max() - pc.min() + 1e-12)
    auc  = roc_auc_score(y_true, pn)
    ap   = average_precision_score(y_true, pn)
    bri  = brier_score_loss(y_true, pn)
    fpr, tpr, _ = roc_curve(y_true, pn)
    print(f'  PGR-KDE ({label:35s}): AUC={auc:.3f} | AP={ap:.3f} | Brier={bri:.3f}')
    return {'Modelo': label, 'AUC': auc, 'AP': ap, 'Brier': bri, 'fpr': fpr, 'tpr': tpr}

var_display = {
    'Slope':     'Pendiente',
    'Aspect':    'Aspecto',
    'TWI':       'ITH',
    'Geology':   'Geologia',
    'Landcover': 'Cobertura'
}
all_var_names = ['Slope', 'Aspect', 'TWI', 'Geology', 'Landcover']

results_pgr = []
# Univariados
for v in all_var_names:
    results_pgr.append(run_pgr_kde([v], var_display[v]))
# Modelo de 3 variables (ganador) – ya calculado, reusar métricas y ROC
fpr_3var, tpr_3var, _ = roc_curve(y_true, gpr_pred_norm)
results_pgr.append({'Modelo': 'Pendiente + Geologia + Cobertura',
                    'AUC': auc_gpr, 'AP': ap_gpr, 'Brier': brier_gpr,
                    'fpr': fpr_3var, 'tpr': tpr_3var})
# Benchmark (todas las variables)
results_pgr.append(run_pgr_kde(all_var_names, 'Benchmark (todas - 5 var)'))

df_pgr = pd.DataFrame([{k: v for k, v in r.items() if k not in ('fpr','tpr')}
                        for r in results_pgr])
print('\n-- PGR-KDE: tabla completa --')
print(df_pgr.to_string(index=False))

# Fig 6: Curvas ROC – todos los modelos PGR-KDE
colors_pgr = ['#2166ac','#4dac26','#d01c8b','#f1a340','#998ec3','#e08214','#762a83']
linestyles = ['-','-','-','-','-','--','--']
fig, ax = plt.subplots(figsize=(9, 7))
for rec, col, ls in zip(results_pgr, colors_pgr, linestyles):
    ax.plot(rec['fpr'], rec['tpr'], color=col, linewidth=2.0, linestyle=ls,
            label=f"{rec['Modelo']} (AUC={rec['AUC']:.3f})")
ax.plot([0,1],[0,1], 'gray', linestyle=':', alpha=0.5, linewidth=1)
ax.legend(fontsize=9, loc='lower right')
ax.set_xlabel('Tasa de Falsos Positivos', fontsize=14)
ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig6_roc_pgr_kde.png', dpi=300); plt.close()
print('  Fig6_roc_pgr_kde.png guardada.')

# =====================================================================
# STEP 6: PGC – Comparacion multi-configuracion
# =====================================================================
print('STEP 6: PGC Multi-Model Comparison...')
gpc_configs = [
    (['Slope','Aspect','TWI','Geology','Landcover'], 'PGC: Benchmark (TODAS)'),
    (['Slope','Aspect','TWI'],                       'PGC: Solo Topografico'),
    (['Geology','Landcover'],                         'PGC: Solo Geoespacial'),
    (['Slope','Geology','Landcover'],                 'PGC: Pendiente+Geo+LC (Optimo)')
]

results_gpc  = []
gpc_roc_data = []
for names_cfg, m_name in gpc_configs:
    X_c    = np.column_stack([feat_dict[n][r_valid, c_valid] for n in names_cfg])
    sc_c   = StandardScaler()
    X_tr_c = sc_c.fit_transform(X_c[idx_train])
    X_f_c  = sc_c.transform(X_c)
    kernel = 1.0**2 * Matern(length_scale=[1.0]*len(names_cfg), nu=1.5)
    gpc_c  = GaussianProcessClassifier(kernel=kernel, random_state=42)
    gpc_c.fit(X_tr_c, df_all['Label'].iloc[idx_train])
    probs = np.zeros(len(X_f_c), dtype=np.float32)
    for i in range(0, len(X_f_c), chunk):
        probs[i:i+chunk] = gpc_c.predict_proba(X_f_c[i:i+chunk])[:, 1]
    auc    = roc_auc_score(y_true, probs)
    ap     = average_precision_score(y_true, probs)
    brier  = brier_score_loss(y_true, probs)
    results_gpc.append({'Modelo': m_name, 'AUC': auc, 'AP': ap, 'Brier': brier})
    fpr, tpr, _ = roc_curve(y_true, probs)
    gpc_roc_data.append((fpr, tpr, m_name, auc))
    print(f'  {m_name}: AUC={auc:.3f}')

# Fig 8: Curvas ROC – solo modelos PGC
colors_gpc = ['#1b7837','#762a83','#e08214','#d6604d']
fig, ax = plt.subplots(figsize=(9, 7))
for (fpr, tpr, name, auc), col in zip(gpc_roc_data, colors_gpc):
    ax.plot(fpr, tpr, color=col, linewidth=2.0, label=f'{name} (AUC={auc:.3f})')
ax.plot([0,1],[0,1], 'gray', linestyle=':', alpha=0.5, linewidth=1)
ax.legend(fontsize=9.5, loc='lower right')
ax.set_xlabel('Tasa de Falsos Positivos', fontsize=14)
ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig8_roc_pgc.png', dpi=300); plt.close()
print('  Fig8_roc_pgc.png guardada.')

# =====================================================================
# STEP 7: Modelo ganador PGC (Pendiente + Geologia + Cobertura)
# =====================================================================
print('STEP 7: Winner PGC Model & Mapping...')
names  = ['Slope', 'Geology', 'Landcover']
X_w    = np.column_stack([feat_dict[n][r_valid, c_valid] for n in names])
sc_w   = StandardScaler()
X_tr_w = sc_w.fit_transform(X_w[idx_train])
X_f_w  = sc_w.transform(X_w)

gpc_w = GaussianProcessClassifier(
    kernel=1.0**2 * Matern(length_scale=[1.0]*3, nu=1.5), random_state=42)
gpc_w.fit(X_tr_w, df_all['Label'].iloc[idx_train])
prob_winner = np.zeros(len(X_f_w), dtype=np.float32)
for i in range(0, len(X_f_w), chunk):
    prob_winner[i:i+chunk] = gpc_w.predict_proba(X_f_w[i:i+chunk])[:, 1]

gpr_std_model = GaussianProcessRegressor(
    kernel=1.0**2 * Matern(length_scale=[1.0]*3, nu=1.5), alpha=0.1)
gpr_std_model.fit(X_tr_w, df_all['Label'].iloc[idx_train])
std_winner = np.zeros(len(X_f_w), dtype=np.float32)
for i in range(0, len(X_f_w), chunk):
    _, s = gpr_std_model.predict(X_f_w[i:i+chunk], return_std=True)
    std_winner[i:i+chunk] = s

lscales = gpc_w.kernel_.k2.length_scale
imp     = 1.0 / lscales
df_imp  = pd.DataFrame({'Variable': names, 'Importance': imp}).sort_values(
    'Importance', ascending=False)
print("\nVARIABLE IMPORTANCE (PGC optimo):")
print(df_imp.to_string(index=False))

auc_winner   = roc_auc_score(y_true, prob_winner)
ap_winner    = average_precision_score(y_true, prob_winner)
brier_winner = brier_score_loss(y_true, prob_winner)
print(f'\nPGC Optimo -> AUC={auc_winner:.3f} | AP={ap_winner:.3f} | Brier={brier_winner:.3f}')

susc_map = np.full(dem_shape, np.nan, np.float32); susc_map[r_valid, c_valid] = prob_winner
std_map  = np.full(dem_shape, np.nan, np.float32); std_map[r_valid,  c_valid] = std_winner

raster_prof = dem_prof.copy(); raster_prof.update(dtype='float32', nodata=-9999)
for fname, arr in [('susceptibility_GP_mean.tif', susc_map),
                   ('susceptibility_GP_std.tif', std_map)]:
    tmp = arr.copy(); tmp[np.isnan(tmp)] = -9999
    with rasterio.open(str(DATA_DIR / fname), 'w', **raster_prof) as dst:
        dst.write(tmp.astype(np.float32), 1)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(susc_map, cmap=cmap_susc, vmin=0, vmax=1, extent=ext)
cbar = plt.colorbar(im, ax=ax, fraction=0.035)
cbar.set_label('Susceptibilidad PGC', fontsize=16)
cbar.ax.tick_params(labelsize=14)
add_cartography(ax, ext, dem_crs)
plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig9_winner_susceptibility.png', dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(std_map, cmap='viridis', extent=ext)
cbar = plt.colorbar(im, ax=ax, fraction=0.035)
cbar.set_label('Desviacion Estandar (Incertidumbre)', fontsize=16)
cbar.ax.tick_params(labelsize=14)
add_cartography(ax, ext, dem_crs)
plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig10_winner_uncertainty.png', dpi=300); plt.close()
print('  Fig9 & Fig10 guardadas.')

# =====================================================================
# STEP 8: Reporte final
# =====================================================================
print('\n' + '='*65)
print('REPORTE FINAL DE RESULTADOS')
print('='*65)
print(f'\n-- PGR-KDE: modelos univariados y multivariados --')
print(df_pgr.to_string(index=False))
print(f'  (Correlacion KDE para modelo 3-var: r={r_gpr:.3f} | RMSE={rmse_gpr:.4f})')
print(f'\n-- PGC: Comparacion multi-configuracion --')
print(pd.DataFrame(results_gpc).to_string(index=False))
print(f'\n-- PGC Ganador (Pendiente + Geologia + Cobertura) --')
print(f'  AUC={auc_winner:.3f} | AP={ap_winner:.3f} | Brier={brier_winner:.3f}')
print(f'\n-- Importancia ARD (PGC optimo) --')
print(df_imp.to_string(index=False))
print('='*65)
