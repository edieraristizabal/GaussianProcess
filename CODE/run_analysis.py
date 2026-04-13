# =====================================================================
# Landslide Susceptibility – La Iguana – Full Analysis
# =====================================================================
import numpy as np
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.features import geometry_mask
from shapely.geometry import Point
from pysheds.grid import Grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss, roc_curve,
    precision_recall_curve, average_precision_score, confusion_matrix)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'DATA'
FIG_DIR  = BASE_DIR / 'FIGURAS'
FIG_DIR.mkdir(exist_ok=True)
np.random.seed(42)

# =====================================================================
print('=' * 60)
print('STEP 1: Loading data')
print('=' * 60)

DEM_PATH = str(DATA_DIR / 'DEM_10m.tif')
WS_PATH  = str(DATA_DIR / 'Cuenca_Iguana.shp')
INV_PATH = str(DATA_DIR / 'MenM_VdeA.gpkg')

ws = gpd.read_file(WS_PATH)

with rasterio.open(DEM_PATH) as src:
    dem_crs       = src.crs
    dem_transform = src.transform
    dem_profile   = src.profile
    dem_nodata    = src.nodata
    dem_shape     = src.shape
    dem_res       = src.res
    dem_array     = src.read(1).astype(np.float32)

dem_array[dem_array == dem_nodata] = np.nan

ws_proj = ws.to_crs(dem_crs)
ws_mask = geometry_mask(ws_proj.geometry, transform=dem_transform,
                        invert=True, out_shape=dem_shape)
dem_ws = dem_array.copy()
dem_ws[~ws_mask] = np.nan
valid_cells = int(np.sum(ws_mask))
area_km2 = valid_cells * dem_res[0] * dem_res[1] / 1e6
print(f'Watershed: {valid_cells:,} cells | {area_km2:.2f} km2')

inv = gpd.read_file(INV_PATH, layer='merged')
inv['geometry'] = inv.geometry.apply(lambda g: Point(g.x, g.y))
inv_proj = inv.set_crs('EPSG:4326', allow_override=True).to_crs(dem_crs)
inv_ws = gpd.sjoin(inv_proj, ws_proj[['geometry']],
                   how='inner', predicate='within').drop_duplicates(subset='id')
print(f'Landslides in watershed: {len(inv_ws)}')

# =====================================================================
print('\n' + '=' * 60)
print('STEP 2: Terrain covariates')
print('=' * 60)

def compute_slope_aspect(dem, res_x, res_y):
    d = dem.copy()
    d[np.isnan(d)] = np.nanmean(d)
    dy, dx = np.gradient(d, res_y, res_x)
    slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)
    aspect = (np.degrees(np.arctan2(-dx, dy)) % 360).astype(np.float32)
    return slope, aspect

slope, aspect = compute_slope_aspect(dem_ws, dem_res[0], dem_res[1])
slope[~ws_mask]  = np.nan
aspect[~ws_mask] = np.nan
print(f'Slope: {np.nanmin(slope):.1f}-{np.nanmax(slope):.1f} deg')

print('Computing TWI (pysheds)...')
grid = Grid.from_raster(DEM_PATH)
dem_g = grid.read_raster(DEM_PATH)
pit_filled = grid.fill_pits(dem_g)
flooded    = grid.fill_depressions(pit_filled)
inflated   = grid.resolve_flats(flooded)
fdir = grid.flowdir(inflated)
acc  = grid.accumulation(fdir).astype(np.float64)

rx = abs(grid.affine.a)
ry = abs(grid.affine.e)
da = np.array(dem_g).astype(np.float64)
da[da == dem_g.nodata] = np.nan
df_dem = da.copy()
df_dem[np.isnan(df_dem)] = np.nanmean(df_dem)
dy_g, dx_g = np.gradient(df_dem, ry, rx)
sr = np.arctan(np.sqrt(dx_g**2 + dy_g**2))
sr[sr < 0.001] = 0.001
a_sp = (np.array(acc) + 1) * rx * ry / rx
twi = np.log(a_sp / np.tan(sr)).astype(np.float32)

if twi.shape != dem_shape:
    from scipy.ndimage import zoom
    twi = zoom(twi, (dem_shape[0]/twi.shape[0], dem_shape[1]/twi.shape[1]), order=1)

twi[~ws_mask] = np.nan
print(f'TWI: {np.nanmin(twi):.2f}-{np.nanmax(twi):.2f}')

def save_raster(arr, path, profile):
    p = profile.copy()
    p.update(dtype='float32', count=1, nodata=-9999.0, compress='lzw')
    a = arr.copy().astype(np.float32)
    a[np.isnan(a)] = -9999.0
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(a, 1)
    print(f'  Saved: {path}')

save_raster(slope,  str(DATA_DIR / 'slope.tif'),  dem_profile)
save_raster(aspect, str(DATA_DIR / 'aspect.tif'), dem_profile)
save_raster(twi,    str(DATA_DIR / 'twi.tif'),    dem_profile)

# =====================================================================
print('\n' + '=' * 60)
print('STEP 3: Sample points')
print('=' * 60)

def extract_at_points(geoms, arrays, transform, shape):
    rows = []
    data = {k: [] for k in arrays}
    for geom in geoms:
        r, c = rowcol(transform, geom.x, geom.y)
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            vals = {k: arr[r, c] for k, arr in arrays.items()}
            if not any(np.isnan(v) for v in vals.values()):
                for k, v in vals.items():
                    data[k].append(v)
                rows.append((r, c))
    return pd.DataFrame(data), rows

arrs = {'slope': slope, 'aspect': aspect, 'twi': twi}
ls_df, ls_cells = extract_at_points(inv_ws.geometry, arrs, dem_transform, dem_shape)
ls_df['label'] = 1
ls_cells_set = set(map(tuple, ls_cells))
n_ls = len(ls_df)
print(f'Landslide samples: {n_ls}')

valid_r, valid_c = np.where(ws_mask & ~np.isnan(slope) & ~np.isnan(twi))
idx = np.random.choice(len(valid_r), size=min(len(valid_r), n_ls * 5), replace=False)
bg = []
for i in idx:
    r, c = int(valid_r[i]), int(valid_c[i])
    if (r, c) not in ls_cells_set:
        bg.append({'slope': slope[r,c], 'aspect': aspect[r,c], 'twi': twi[r,c], 'label': 0})
    if len(bg) >= n_ls:
        break

bg_df = pd.DataFrame(bg[:n_ls])
print(f'Background samples: {len(bg_df)}')

dataset = pd.concat([ls_df[['slope','aspect','twi','label']], bg_df], ignore_index=True)
print(f'Total: {len(dataset)} | Balance: {dataset.label.value_counts().to_dict()}')

# =====================================================================
print('\n' + '=' * 60)
print('STEP 4: Logistic Regression')
print('=' * 60)

feats = ['slope', 'aspect', 'twi']
X = dataset[feats].values
y = dataset['label'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
lr.fit(X_tr_s, y_tr)

y_prob_te = lr.predict_proba(X_te_s)[:, 1]
y_pred_te = (y_prob_te >= 0.5).astype(int)
cv_auc = cross_val_score(lr, scaler.transform(X), y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring='roc_auc')
print(f'LR Test AUC:  {roc_auc_score(y_te, y_prob_te):.4f}')
print(f'LR CV AUC:    {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}')
print(f'LR Accuracy:  {accuracy_score(y_te, y_pred_te):.4f}')
print(f'LR F1:        {f1_score(y_te, y_pred_te):.4f}')
coef = lr.coef_[0]
print(f'Coef: slope={coef[0]:+.3f}  aspect={coef[1]:+.3f}  twi={coef[2]:+.3f}')

# =====================================================================
print('\n' + '=' * 60)
print('STEP 5: LR prediction all cells')
print('=' * 60)

valid_mask = ws_mask & ~np.isnan(slope) & ~np.isnan(aspect) & ~np.isnan(twi)
r_all, c_all = np.where(valid_mask)
X_all = np.column_stack([slope[r_all, c_all], aspect[r_all, c_all], twi[r_all, c_all]])
X_all_s = scaler.transform(X_all)
print(f'Predicting LR for {len(X_all):,} cells...')
prob_lr_all = lr.predict_proba(X_all_s)[:, 1]
prob_map_lr = np.full(dem_shape, np.nan, np.float32)
prob_map_lr[r_all, c_all] = prob_lr_all
print(f'LR prob: {np.nanmin(prob_map_lr):.4f}-{np.nanmax(prob_map_lr):.4f}')
save_raster(prob_map_lr, str(DATA_DIR / 'susceptibility_LR.tif'), dem_profile)

# =====================================================================
print('\n' + '=' * 60)
print('STEP 6: Gaussian Process')
print('=' * 60)

ls_r_list, ls_c_list = [], []
for geom in inv_ws.geometry:
    r, c = rowcol(dem_transform, geom.x, geom.y)
    ls_r_list.append(r)
    ls_c_list.append(c)

gp_r, gp_c, gp_y = [], [], []
seen = set()
for r, c in zip(ls_r_list, ls_c_list):
    if 0 <= r < dem_shape[0] and 0 <= c < dem_shape[1] and valid_mask[r, c] and (r,c) not in seen:
        seen.add((r, c))
        gp_r.append(r); gp_c.append(c); gp_y.append(float(prob_map_lr[r, c]))

gp_r = np.array(gp_r)
gp_c = np.array(gp_c)
gp_y = np.array(gp_y)

X_gp = np.column_stack([slope[gp_r, gp_c], aspect[gp_r, gp_c], twi[gp_r, gp_c]])
ok = ~np.any(np.isnan(X_gp), axis=1)
X_gp = X_gp[ok]; gp_y = gp_y[ok]
print(f'GP training points: {len(X_gp)}')

scaler_gp = StandardScaler()
X_gp_s = scaler_gp.fit_transform(X_gp)

MAX_GP = 400
if len(X_gp_s) > MAX_GP:
    idx_sub = np.random.choice(len(X_gp_s), MAX_GP, replace=False)
    X_fit = X_gp_s[idx_sub]; y_fit = gp_y[idx_sub]
    print(f'Subsampled to {MAX_GP} points for GP fitting')
else:
    X_fit = X_gp_s; y_fit = gp_y

kernel = (ConstantKernel(1.0, (0.01, 10.0))
          * Matern(length_scale=1.0, length_scale_bounds=(0.01, 10.0), nu=1.5)
          + WhiteKernel(0.01, (1e-4, 0.5)))

print('Fitting GP...')
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                               normalize_y=True, random_state=42)
gp.fit(X_fit, y_fit)
print('Optimized kernel:', gp.kernel_)

BATCH = 5000
X_all_gp_s = scaler_gp.transform(X_all)
n = len(X_all_gp_s)
gp_mu = np.zeros(n, np.float32)
gp_sg = np.zeros(n, np.float32)
print(f'Predicting GP for {n:,} cells...')
for i in range(0, n, BATCH):
    if i % (50 * BATCH) == 0:
        print(f'  {i}/{n} ({100*i/n:.0f}%)')
    mu, sg = gp.predict(X_all_gp_s[i:i+BATCH], return_std=True)
    gp_mu[i:i+BATCH] = mu.astype(np.float32)
    gp_sg[i:i+BATCH] = sg.astype(np.float32)
gp_mu = np.clip(gp_mu, 0, 1)

prob_map_gp_mu = np.full(dem_shape, np.nan, np.float32)
prob_map_gp_sg = np.full(dem_shape, np.nan, np.float32)
prob_map_gp_mu[r_all, c_all] = gp_mu
prob_map_gp_sg[r_all, c_all] = gp_sg
print(f'GP mean: {np.nanmin(prob_map_gp_mu):.4f}-{np.nanmax(prob_map_gp_mu):.4f}')
print(f'GP std:  {np.nanmin(prob_map_gp_sg):.4f}-{np.nanmax(prob_map_gp_sg):.4f}')
save_raster(prob_map_gp_mu, str(DATA_DIR / 'susceptibility_GP_mean.tif'), dem_profile)
save_raster(prob_map_gp_sg, str(DATA_DIR / 'susceptibility_GP_std.tif'),  dem_profile)

# =====================================================================
print('\n' + '=' * 60)
print('STEP 7: Metrics')
print('=' * 60)

label_map = np.zeros(dem_shape, np.int8)
for r, c in zip(ls_r_list, ls_c_list):
    if 0 <= r < dem_shape[0] and 0 <= c < dem_shape[1]:
        label_map[r, c] = 1
y_true = label_map[r_all, c_all]

def compute_metrics(y_true, y_prob, name, thr=0.5):
    yp = (y_prob >= thr).astype(int)
    return {
        'Model': name,
        'AUC'  : round(roc_auc_score(y_true, y_prob), 4),
        'AP'   : round(average_precision_score(y_true, y_prob), 4),
        'Acc'  : round(accuracy_score(y_true, yp), 4),
        'Prec' : round(precision_score(y_true, yp, zero_division=0), 4),
        'Rec'  : round(recall_score(y_true, yp, zero_division=0), 4),
        'F1'   : round(f1_score(y_true, yp, zero_division=0), 4),
        'Brier': round(brier_score_loss(y_true, y_prob), 4),
    }

m_lr = compute_metrics(y_true, prob_lr_all, 'Logistic Regression')
m_gp = compute_metrics(y_true, gp_mu,       'Gaussian Process')
df_met = pd.DataFrame([m_lr, m_gp]).set_index('Model')
print(df_met.to_string())
df_met.to_csv(str(DATA_DIR / 'metrics.csv'))

# =====================================================================
print('\n' + '=' * 60)
print('STEP 8: Figures')
print('=' * 60)

susc_colors = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']
cmap_susc = LinearSegmentedColormap.from_list('susc', susc_colors, N=256)

def get_ls_pixels(inv_ws, dem_transform, dem_shape):
    px_x, px_y = [], []
    for geom in inv_ws.geometry:
        r, c = rowcol(dem_transform, geom.x, geom.y)
        r = int(np.clip(r, 0, dem_shape[0]-1))
        c = int(np.clip(c, 0, dem_shape[1]-1))
        px_x.append(c); px_y.append(r)
    return np.array(px_x), np.array(px_y)

px_x, px_y = get_ls_pixels(inv_ws, dem_transform, dem_shape)

PANEL_KW = dict(fontsize=11, fontweight='bold', ha='left', va='top',
                transform=None)  # usado con ax.transAxes

def add_panel_label(ax, letter):
    ax.text(0.015, 0.985, letter, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

# ---- Fig 1: Covariables del terreno (4 paneles, sin título global) ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

im0 = axes[0,0].imshow(dem_ws, cmap='terrain', interpolation='nearest')
plt.colorbar(im0, ax=axes[0,0], label='m s.n.m.')
add_panel_label(axes[0,0], '(a)')

im1 = axes[0,1].imshow(slope, cmap='YlOrRd', vmin=0, vmax=60, interpolation='nearest')
plt.colorbar(im1, ax=axes[0,1], label='Grados (°)')
add_panel_label(axes[0,1], '(b)')

im2 = axes[1,0].imshow(aspect, cmap='hsv', vmin=0, vmax=360, interpolation='nearest')
plt.colorbar(im2, ax=axes[1,0], label='Grados (°, N=0)')
add_panel_label(axes[1,0], '(c)')

twi_c = np.clip(twi, np.nanpercentile(twi,2), np.nanpercentile(twi,98))
im3 = axes[1,1].imshow(twi_c, cmap='Blues', interpolation='nearest')
plt.colorbar(im3, ax=axes[1,1], label='ITH')
add_panel_label(axes[1,1], '(d)')

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig1_terrain_covariates.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig1 saved')

# ---- Fig 2: Evaluación del modelo de Regresión Logística (sin título) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Curva ROC – solo LR
ax = axes[0]
fpr, tpr, _ = roc_curve(y_true, prob_lr_all)
auc_v = roc_auc_score(y_true, prob_lr_all)
ax.plot(fpr, tpr, color='steelblue', lw=2.5, label=f'AUC = {auc_v:.3f}')
ax.fill_between(fpr, tpr, alpha=0.15, color='steelblue')
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('Tasa de Falsos Positivos', fontsize=11)
ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=11)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
add_panel_label(ax, '(a)')

# Curva Precisión-Recall – solo LR
ax = axes[1]
prec, rec, _ = precision_recall_curve(y_true, prob_lr_all)
ap_v = average_precision_score(y_true, prob_lr_all)
ax.plot(rec, prec, color='steelblue', lw=2.5, label=f'AP = {ap_v:.3f}')
ax.fill_between(rec, prec, alpha=0.15, color='steelblue')
ax.set_xlabel('Exhaustividad (Recall)', fontsize=11)
ax.set_ylabel('Precisión', fontsize=11)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
add_panel_label(ax, '(b)')

plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig2_model_evaluation.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig2 saved')

# ---- Fig 3: Mapa de susceptibilidad – Regresión Logística (sin título) ----
fig, ax = plt.subplots(figsize=(9, 10))
im = ax.imshow(prob_map_lr, cmap=cmap_susc, vmin=0, vmax=1, interpolation='nearest')
ax.scatter(px_x, px_y, s=4, c='black', marker='.', alpha=0.5,
           label=f'Deslizamientos (n={len(inv_ws)})')
plt.colorbar(im, ax=ax, label='P(deslizamiento)', fraction=0.035, pad=0.04)
ax.legend(loc='lower right', fontsize=9, markerscale=3)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig3_LR_susceptibility.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig3 saved')

# ---- Fig 4: Proceso Gaussiano – media y desviación (sin título) ----
fig, axes = plt.subplots(1, 2, figsize=(17, 9))

im0 = axes[0].imshow(prob_map_gp_mu, cmap=cmap_susc, vmin=0, vmax=1, interpolation='nearest')
axes[0].scatter(px_x, px_y, s=4, c='black', marker='.', alpha=0.5,
                label=f'Deslizamientos (n={len(inv_ws)})')
plt.colorbar(im0, ax=axes[0], label='Media P(deslizamiento)', fraction=0.035, pad=0.04)
axes[0].legend(loc='lower right', fontsize=8, markerscale=3)
axes[0].set_xticks([]); axes[0].set_yticks([])
add_panel_label(axes[0], '(a)')

std_c = np.clip(prob_map_gp_sg, 0, np.nanpercentile(prob_map_gp_sg, 98))
im1 = axes[1].imshow(std_c, cmap='plasma', interpolation='nearest')
axes[1].scatter(px_x, px_y, s=4, c='white', marker='.', alpha=0.5,
                label=f'Deslizamientos (n={len(inv_ws)})')
plt.colorbar(im1, ax=axes[1], label='Desviación estándar', fraction=0.035, pad=0.04)
axes[1].legend(loc='lower right', fontsize=8, markerscale=3)
axes[1].set_xticks([]); axes[1].set_yticks([])
add_panel_label(axes[1], '(b)')

plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig4_GP_susceptibility_mean_std.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig4 saved')

# ---- Fig 5: Distribución de clases de susceptibilidad – solo LR (sin título) ----
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
clabels_es = ['Muy Baja', 'Baja', 'Moderada', 'Alta', 'Muy Alta']
colors_cls  = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']

def get_class_pcts(pmap):
    valid = pmap[~np.isnan(pmap)]
    counts = [np.sum((valid >= lo) & (valid < hi)) for lo, hi in zip(bins[:-1], bins[1:])]
    counts[-1] += np.sum(valid == 1.0)
    total = sum(counts)
    return [100*c/total for c in counts]

pcts = get_class_pcts(prob_map_lr)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(clabels_es, pcts, color=colors_cls, edgecolor='black', linewidth=0.7)
for bar, pct in zip(bars, pcts):
    ax.annotate(f'{pct:.1f}%', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Área de la cuenca (%)', fontsize=11)
ax.set_ylim(0, max(pcts)*1.18)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig5_susceptibility_classes.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig5 saved')

# ---- Fig 6: Distribuciones de covariables (sin título global) ----
feat_names_es = ['Pendiente (°)', 'Aspecto (°)', 'ITH']
label_es = ['Fondo', 'Deslizamiento']
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, (ax, fname) in enumerate(zip(axes, feat_names_es)):
    ls_vals = dataset[dataset.label==1][feats[i]].values
    bg_vals = dataset[dataset.label==0][feats[i]].values
    ax.hist(bg_vals, bins=40, alpha=0.6, color='steelblue', label=label_es[0], density=True)
    ax.hist(ls_vals, bins=40, alpha=0.6, color='firebrick', label=label_es[1], density=True)
    ax.set_xlabel(fname, fontsize=11); ax.set_ylabel('Densidad', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    add_panel_label(ax, f'({chr(97+i)})')
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig6_covariate_distributions.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig6 saved')

# Eliminar Fig7 antigua si existe
import os
old_fig7 = str(FIG_DIR / 'Fig7_covariate_distributions.png')
if os.path.exists(old_fig7): os.remove(old_fig7)
old_fig5_old = str(FIG_DIR / 'Fig5_comparison_LR_GP.png')
if os.path.exists(old_fig5_old): os.remove(old_fig5_old)
old_fig6_old = str(FIG_DIR / 'Fig6_susceptibility_classes.png')
if os.path.exists(old_fig6_old): os.remove(old_fig6_old)
print('Old figures cleaned up')

# =====================================================================
# Save results for the notebook
results = {
    'slope': slope, 'aspect': aspect, 'twi': twi, 'dem_ws': dem_ws,
    'prob_map_lr': prob_map_lr, 'prob_map_gp_mu': prob_map_gp_mu,
    'prob_map_gp_sg': prob_map_gp_sg,
    'y_true': y_true, 'prob_lr_all': prob_lr_all, 'gp_mu': gp_mu,
    'm_lr': m_lr, 'm_gp': m_gp, 'df_met': df_met,
    'y_te': y_te, 'y_prob_te': y_prob_te, 'y_pred_te': y_pred_te,
    'valid_cells': valid_cells, 'area_km2': area_km2,
    'n_ls': len(inv_ws), 'n_bg': len(bg_df),
    'cv_auc_mean': float(cv_auc.mean()), 'cv_auc_std': float(cv_auc.std()),
    'lr_coef': lr.coef_[0].tolist(), 'feats': feats,
    'gp_kernel': str(gp.kernel_),
    'dem_res': dem_res, 'dem_shape': dem_shape,
}
with open(str(DATA_DIR / 'results.pkl'), 'wb') as f:
    pickle.dump(results, f)

print('\n' + '=' * 60)
print('ALL STEPS COMPLETE')
print('=' * 60)
print(df_met.to_string())
