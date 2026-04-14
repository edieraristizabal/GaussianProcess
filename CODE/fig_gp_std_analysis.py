"""
Análisis de la desviación estándar del Proceso Gaussiano:
  - Histograma de la distribución de la desviación estándar
  - Diagramas de dispersión: covariables morfométricas vs. desviación estándar
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import pickle
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.features import geometry_mask
from pathlib import Path

BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'DATA'
FIG_DIR  = BASE_DIR / 'FIGURAS'

# ── Cargar resultados ────────────────────────────────────────────────
with open(str(DATA_DIR / 'results.pkl'), 'rb') as f:
    R = pickle.load(f)

slope          = R['slope']
aspect         = R['aspect']
twi            = R['twi']
prob_map_gp_sg = R['prob_map_gp_sg']
dem_shape      = R['dem_shape']

# Coordenadas de deslizamientos
DEM_PATH = str(DATA_DIR / 'DEM_10m.tif')
WS_PATH  = str(DATA_DIR / 'Cuenca_Iguana.shp')
INV_PATH = str(DATA_DIR / 'Deslizamientos_Iguana.gpkg')

ws     = gpd.read_file(WS_PATH)
inv_ws = gpd.read_file(INV_PATH)
with rasterio.open(DEM_PATH) as src:
    dem_crs       = src.crs
    dem_transform = src.transform
    ws_proj = ws.to_crs(dem_crs)
    ws_mask = geometry_mask(ws_proj.geometry, transform=dem_transform,
                             invert=True, out_shape=(src.height, src.width))

# ── Máscara válida ───────────────────────────────────────────────────
valid_mask = ws_mask & ~np.isnan(slope) & ~np.isnan(aspect) & ~np.isnan(twi) & ~np.isnan(prob_map_gp_sg)
r_all, c_all = np.where(valid_mask)

slope_v  = slope[r_all, c_all]
aspect_v = aspect[r_all, c_all]
twi_v    = twi[r_all, c_all]
std_v    = prob_map_gp_sg[r_all, c_all]

# ── Índices de deslizamientos ────────────────────────────────────────
ls_r, ls_c = [], []
for geom in inv_ws.geometry:
    r, c = rowcol(dem_transform, geom.x, geom.y)
    if 0 <= r < dem_shape[0] and 0 <= c < dem_shape[1] and valid_mask[r, c]:
        ls_r.append(r); ls_c.append(c)
ls_r, ls_c = np.array(ls_r), np.array(ls_c)

ls_slope  = slope[ls_r, ls_c]
ls_aspect = aspect[ls_r, ls_c]
ls_twi    = twi[ls_r, ls_c]
ls_std    = prob_map_gp_sg[ls_r, ls_c]

# Submuestreo del fondo para los scatter plots (evitar sobresaturación)
np.random.seed(42)
N_SAMPLE = 15000
idx_bg = np.random.choice(len(std_v), min(N_SAMPLE, len(std_v)), replace=False)
sl_bg  = slope_v[idx_bg];  asp_bg = aspect_v[idx_bg]
twi_bg = twi_v[idx_bg];    std_bg = std_v[idx_bg]

# ── Descriptivos ─────────────────────────────────────────────────────
print('=== Estadísticos de la desviación estándar del PG ===')
print(f'  Mín:      {std_v.min():.5f}')
print(f'  Máx:      {std_v.max():.5f}')
print(f'  Media:    {std_v.mean():.5f}')
print(f'  Mediana:  {np.median(std_v):.5f}')
for p in [5, 25, 75, 95, 99]:
    print(f'  P{p:2d}:      {np.percentile(std_v, p):.5f}')
r_sl,  p_sl  = stats.spearmanr(slope_v,  std_v)
r_asp, p_asp = stats.spearmanr(aspect_v, std_v)
r_twi, p_twi = stats.spearmanr(twi_v,    std_v)
print(f'\n  Spearman rho (pendiente  vs std): {r_sl:+.4f}  p={p_sl:.2e}')
print(f'  Spearman rho (aspecto    vs std): {r_asp:+.4f}  p={p_asp:.2e}')
print(f'  Spearman rho (ITH        vs std): {r_twi:+.4f}  p={p_twi:.2e}')

# ── Colormap desviación ───────────────────────────────────────────────
cmap_std = 'plasma'
std_clip = np.clip(std_v, 0, np.percentile(std_v, 98))

# ====================================================================
# FIGURA 7: Análisis completo de la desviación estándar del PG
# Layout:  fila superior  → histograma (izq) + boxplots por clase (der)
#          fila inferior  → 3 scatter plots (pendiente, aspecto, ITH vs σ)
# ====================================================================
fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Colores para deslizamientos vs. cuenca ───────────────────────────
C_BASIN = '#4393c3'
C_LS    = '#d6604d'
ALPHA_S = 0.25    # transparencia scatter fondo
MS      = 10      # tamaño puntos deslizamiento

# ─────────────────────────────────────────────────────────────────────
# Panel (a): Histograma + KDE de σ (cuenca completa vs. deslizamientos)
# ─────────────────────────────────────────────────────────────────────
ax_hist = fig.add_subplot(gs[0, 0:2])

# Clip para visibilidad (eliminar cola larga del 1%)
std_plot    = np.clip(std_v,  0, np.percentile(std_v, 99))
ls_std_plot = np.clip(ls_std, 0, np.percentile(std_v, 99))

n_bins = 80
ax_hist.hist(std_plot, bins=n_bins, density=True, color=C_BASIN,
             alpha=0.55, label='Cuenca completa', edgecolor='none')
ax_hist.hist(ls_std_plot, bins=n_bins, density=True, color=C_LS,
             alpha=0.70, label='Deslizamientos', edgecolor='none')

# KDE
kde_basin = stats.gaussian_kde(std_plot, bw_method='scott')
kde_ls    = stats.gaussian_kde(ls_std_plot, bw_method='scott')
x_kde = np.linspace(0, std_plot.max(), 400)
ax_hist.plot(x_kde, kde_basin(x_kde), color=C_BASIN, lw=2.5)
ax_hist.plot(x_kde, kde_ls(x_kde),    color=C_LS,    lw=2.5)

# Medias
ax_hist.axvline(np.mean(std_plot),    color=C_BASIN, lw=1.5, ls='--',
                label=f'Media cuenca = {np.mean(std_plot):.4f}')
ax_hist.axvline(np.mean(ls_std_plot), color=C_LS,    lw=1.5, ls='--',
                label=f'Media desliz. = {np.mean(ls_std_plot):.4f}')

ax_hist.set_xlabel('Desviación estándar posterior σ', fontsize=11)
ax_hist.set_ylabel('Densidad', fontsize=11)
ax_hist.legend(fontsize=8.5, loc='upper right')
ax_hist.grid(True, alpha=0.3)
ax_hist.text(0.015, 0.97, '(a)', transform=ax_hist.transAxes,
             fontsize=12, fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

# ─────────────────────────────────────────────────────────────────────
# Panel (b): Box-plots de σ por clase de susceptibilidad
# ─────────────────────────────────────────────────────────────────────
ax_box = fig.add_subplot(gs[0, 2])

prob_lr = R['prob_map_lr']
prob_lr_v = prob_lr[r_all, c_all]
bins_cls  = [0, 0.2, 0.4, 0.6, 0.8, 1.001]
cls_names = ['Muy\nBaja', 'Baja', 'Mod.', 'Alta', 'Muy\nAlta']
cls_colors_hex = ['#1a9641', '#a6d96a', '#f7f79b', '#fdae61', '#d7191c']

groups = []
for lo, hi in zip(bins_cls[:-1], bins_cls[1:]):
    mask_c = (prob_lr_v >= lo) & (prob_lr_v < hi)
    groups.append(std_v[mask_c])

bp = ax_box.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color='black', lw=2),
                    whiskerprops=dict(lw=1.2),
                    flierprops=dict(marker='.', alpha=0.2, ms=2),
                    showfliers=True)
for patch, col in zip(bp['boxes'], cls_colors_hex):
    patch.set_facecolor(col); patch.set_alpha(0.85)

ax_box.set_xticks(range(1, 6))
ax_box.set_xticklabels(cls_names, fontsize=8.5)
ax_box.set_ylabel('Desviación estándar posterior σ', fontsize=10)
ax_box.grid(True, axis='y', alpha=0.3)
ax_box.text(0.015, 0.97, '(b)', transform=ax_box.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

# ─────────────────────────────────────────────────────────────────────
# Paneles (c)(d)(e): Scatter σ vs. pendiente / aspecto / ITH
# ─────────────────────────────────────────────────────────────────────
scatter_data = [
    (sl_bg,  ls_slope,  'Pendiente (°)',  r_sl,  '(c)'),
    (asp_bg, ls_aspect, 'Aspecto (°)',    r_asp, '(d)'),
    (twi_bg, ls_twi,    'ITH',           r_twi, '(e)'),
]

for col_idx, (bg_x, ls_x, xlabel, rho, panel_lbl) in enumerate(scatter_data):
    ax = fig.add_subplot(gs[1, col_idx])

    # Fondo (submuestreado)
    ax.scatter(bg_x, std_bg, s=2, c=C_BASIN, alpha=ALPHA_S,
               rasterized=True, label='Cuenca')
    # Deslizamientos
    ax.scatter(ls_x, ls_std, s=MS, c=C_LS, alpha=0.85,
               edgecolors='none', label='Deslizamientos', zorder=5)

    # Línea de tendencia LOWESS sobre la muestra de fondo
    from scipy.stats import binned_statistic
    x_all = np.concatenate([bg_x, ls_x])
    y_all = np.concatenate([std_bg, ls_std])
    n_bins_tr = 30
    bin_means, bin_edges, _ = binned_statistic(x_all, y_all, statistic='median', bins=n_bins_tr)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    valid_bins  = ~np.isnan(bin_means)
    ax.plot(bin_centers[valid_bins], bin_means[valid_bins],
            color='black', lw=2, ls='-', label='Mediana por intervalo', zorder=6)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Desviación estándar σ', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right',
              handlelength=1, borderpad=0.5)
    ax.grid(True, alpha=0.25)
    # Texto correlación
    ax.text(0.97, 0.97,
            f'ρ = {rho:+.3f}',
            transform=ax.transAxes, fontsize=9.5,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9, ec='gray'))
    ax.text(0.015, 0.97, panel_lbl, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

plt.savefig(str(FIG_DIR / 'Fig7_GP_std_analysis.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print('Fig7 guardada: FIGURAS/Fig7_GP_std_analysis.png')
