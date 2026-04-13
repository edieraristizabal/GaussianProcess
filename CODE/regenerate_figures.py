"""Regenera todas las figuras cargando resultados pre-calculados del pickle."""
import numpy as np
import pandas as pd
import pickle, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from pathlib import Path

BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'DATA'
FIG_DIR  = BASE_DIR / 'FIGURAS'

with open(str(DATA_DIR / 'results.pkl'), 'rb') as f:
    R = pickle.load(f)

slope          = R['slope']
aspect         = R['aspect']
twi            = R['twi']
dem_ws         = R['dem_ws']
prob_map_lr    = R['prob_map_lr']
prob_map_gp_mu = R['prob_map_gp_mu']
prob_map_gp_sg = R['prob_map_gp_sg']
y_true         = R['y_true']
prob_lr_all    = R['prob_lr_all']
gp_mu          = R['gp_mu']
m_lr           = R['m_lr']
m_gp           = R['m_gp']
dataset        = None  # reconstruir desde feats
n_ls           = R['n_ls']
dem_shape      = R['dem_shape']
dem_transform  = None  # no necesario para figuras de imagen

# Reconstruir dataset desde pickled arrays no disponibles: lo recreamos
# usando slope/aspect/twi del pickle y el inventario recortado
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point

WS_PATH  = str(DATA_DIR / 'Cuenca_Iguana.shp')
INV_PATH = str(DATA_DIR / 'Deslizamientos_Iguana.gpkg')
DEM_PATH = str(DATA_DIR / 'DEM_10m.tif')

ws = gpd.read_file(WS_PATH)
with rasterio.open(DEM_PATH) as src:
    dem_crs = src.crs
    dem_transform = src.transform

inv_ws = gpd.read_file(INV_PATH)

# Pixeles de deslizamientos
px_x, px_y = [], []
for geom in inv_ws.geometry:
    r, c = rowcol(dem_transform, geom.x, geom.y)
    px_x.append(int(np.clip(c, 0, dem_shape[1]-1)))
    px_y.append(int(np.clip(r, 0, dem_shape[0]-1)))
px_x, px_y = np.array(px_x), np.array(px_y)

# ---- Estilos ----
susc_colors = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']
cmap_susc   = LinearSegmentedColormap.from_list('susc', susc_colors, N=256)

def add_panel_label(ax, letter):
    ax.text(0.015, 0.985, letter, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

print('Generando figuras...')

# ====================================================================
# Fig 1 – Covariables del terreno (sin título)
# ====================================================================
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

twi_c = np.clip(twi, np.nanpercentile(twi, 2), np.nanpercentile(twi, 98))
im3 = axes[1,1].imshow(twi_c, cmap='Blues', interpolation='nearest')
plt.colorbar(im3, ax=axes[1,1], label='ITH')
add_panel_label(axes[1,1], '(d)')

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig1_terrain_covariates.png'), dpi=200, bbox_inches='tight')
plt.close()
print('  Fig1 ✓')

# ====================================================================
# Fig 2 – Evaluación del modelo RL: curva ROC y Precisión-Recall
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
print('  Fig2 ✓')

# ====================================================================
# Fig 3 – Mapa de susceptibilidad – Regresión Logística (sin título)
# ====================================================================
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
print('  Fig3 ✓')

# ====================================================================
# Fig 4 – Proceso Gaussiano: media y desviación estándar (sin título)
# ====================================================================
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
print('  Fig4 ✓')

# ====================================================================
# Fig 5 – Distribución de clases de susceptibilidad (solo RL, sin título)
# ====================================================================
bins        = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
clabels_es  = ['Muy Baja', 'Baja', 'Moderada', 'Alta', 'Muy Alta']
colors_cls  = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']

def get_class_pcts(pmap):
    valid  = pmap[~np.isnan(pmap)]
    counts = [np.sum((valid >= lo) & (valid < hi)) for lo, hi in zip(bins[:-1], bins[1:])]
    counts[-1] += int(np.sum(valid == 1.0))
    total  = sum(counts)
    return [100*c/total for c in counts]

pcts = get_class_pcts(prob_map_lr)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(clabels_es, pcts, color=colors_cls, edgecolor='black', linewidth=0.7)
for bar, pct in zip(bars, pcts):
    ax.annotate(f'{pct:.1f}%',
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Área de la cuenca (%)', fontsize=11)
ax.set_ylim(0, max(pcts)*1.18)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig5_susceptibility_classes.png'), dpi=200, bbox_inches='tight')
plt.close()
print('  Fig5 ✓')

# ====================================================================
# Fig 6 – Distribuciones de covariables: deslizamientos vs. fondo (sin título)
# ====================================================================
# Reconstruir muestras desde arrays guardados
import rasterio
from rasterio.features import geometry_mask
from pathlib import Path as P2

with rasterio.open(DEM_PATH) as src:
    ws_proj = gpd.read_file(WS_PATH).to_crs(src.crs)
    ws_mask = geometry_mask(ws_proj.geometry, transform=src.transform,
                            invert=True, out_shape=(src.height, src.width))

valid_mask = ws_mask & ~np.isnan(slope) & ~np.isnan(aspect) & ~np.isnan(twi)
r_all, c_all = np.where(valid_mask)

np.random.seed(42)
ls_rows, ls_cols = [], []
for geom in inv_ws.geometry:
    r, c = rowcol(dem_transform, geom.x, geom.y)
    ls_rows.append(r); ls_cols.append(c)
ls_cells_set = set(zip(ls_rows, ls_cols))

ls_s  = [(slope[r,c], aspect[r,c], twi[r,c]) for r,c in ls_cells_set
          if 0<=r<dem_shape[0] and 0<=c<dem_shape[1] and valid_mask[r,c]]
n_ls2 = len(ls_s)
idx_bg = np.random.choice(len(r_all), size=min(len(r_all), n_ls2*5), replace=False)
bg_s   = []
for i in idx_bg:
    r, c = int(r_all[i]), int(c_all[i])
    if (r,c) not in ls_cells_set:
        bg_s.append((slope[r,c], aspect[r,c], twi[r,c]))
    if len(bg_s) >= n_ls2: break

ls_arr = np.array(ls_s[:n_ls2])
bg_arr = np.array(bg_s[:n_ls2])

feat_names_es = ['Pendiente (°)', 'Aspecto (°)', 'ITH']
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, (ax, fname) in enumerate(zip(axes, feat_names_es)):
    ax.hist(bg_arr[:,i], bins=40, alpha=0.6, color='steelblue',
            label='Fondo', density=True)
    ax.hist(ls_arr[:,i], bins=40, alpha=0.6, color='firebrick',
            label='Deslizamiento', density=True)
    ax.set_xlabel(fname, fontsize=11)
    ax.set_ylabel('Densidad', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    add_panel_label(ax, f'({chr(97+i)})')
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig6_covariate_distributions.png'), dpi=200, bbox_inches='tight')
plt.close()
print('  Fig6 ✓')

# Limpiar figuras antiguas que ya no aplican
for old in ['Fig5_comparison_LR_GP.png', 'Fig6_susceptibility_classes.png',
            'Fig7_covariate_distributions.png']:
    p = str(FIG_DIR / old)
    if os.path.exists(p):
        os.remove(p)
        print(f'  Eliminado: {old}')

print('\nTodas las figuras generadas en FIGURAS/')
