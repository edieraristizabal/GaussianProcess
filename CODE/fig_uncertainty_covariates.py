# =====================================================================
# Uncertainty vs. Covariates Figure
# Self-contained: retrains GPR to get posterior std dev, saves
# corrected susceptibility_GP_std.tif, then generates Fig10 and Fig11.
# =====================================================================
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'DATA'
FIG_DIR  = BASE_DIR / 'FIGURAS'
np.random.seed(42)  # same seed as run_analysis.py

# ── Labels ────────────────────────────────────────────────────────────
GEO_LABELS = {1: 'Depósitos', 2: 'Metamórficas', 3: 'Volcánicas N.',
               4: 'Graníticas', 5: 'Volcánicas S.'}
LC_LABELS  = {1: 'Bosque', 2: 'Pastos', 3: 'Urb. Formal',
               4: 'Urb. Informal', 5: 'Condominios'}
GEO_COLORS = ['#8c510a', '#bf812d', '#dfc27d', '#80cdc1', '#35978f']
LC_COLORS  = ['#1a9641', '#a6d96a', '#d7191c', '#fdae61', '#e0a86e']

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Load data (identical to run_analysis.py)
# ─────────────────────────────────────────────────────────────────────
print('STEP 1: Loading data...')
ws      = gpd.read_file(str(DATA_DIR / 'Cuenca_Iguana.shp'))
inv_gdf = gpd.read_file(str(DATA_DIR / 'Deslizamientos_Iguana.gpkg'),
                         layer='Deslizamientos_Iguana')

with rasterio.open(str(DATA_DIR / 'DEM_5m.tif')) as src:
    dem_crs   = src.crs
    dem_trans = src.transform
    dem_shape = src.shape
    dem_prof  = src.profile

ws_proj  = ws.to_crs(dem_crs)
ws_mask  = geometry_mask(ws_proj.geometry, transform=dem_trans,
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
    gdf    = gpd.read_file(path).to_crs(dem_crs)
    ws_g   = ws.union_all()
    gdf_in = gdf[gdf.intersects(ws_g)]
    shapes = ((geom, val) for geom, val in zip(gdf_in.geometry, gdf_in[column]))
    raster = rasterize(shapes, out_shape=dem_shape, transform=dem_trans,
                       fill=0, all_touched=True, dtype=np.float32)
    raster[~ws_mask] = np.nan
    return raster

slope   = load_resample('slope_5m')
aspect  = load_resample('aspect_5m')
twi     = load_resample('twi_5m')
geology  = rasterize_gpkg(DATA_DIR / 'geosurface_map.gpkg', column='Codigo')
landcov  = rasterize_gpkg(DATA_DIR / 'landcover_map.gpkg',  column='code')

valid_mask = ws_mask & ~np.isnan(slope) & ~np.isnan(geology) & ~np.isnan(landcov)

inv_proj = inv_gdf.to_crs(dem_crs)
ls_r, ls_c = [], []
for geom in inv_proj.geometry:
    r, c = rowcol(dem_trans, geom.x, geom.y)
    if 0 <= r < dem_shape[0] and 0 <= c < dem_shape[1] and valid_mask[r, c]:
        ls_r.append(r); ls_c.append(c)
label_map = np.zeros(dem_shape, np.int8)
label_map[ls_r, ls_c] = 1

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Target encoding (same as run_analysis.py)
# ─────────────────────────────────────────────────────────────────────
r_valid, c_valid = np.where(valid_mask)
df_all = pd.DataFrame({
    'Slope': slope[r_valid, c_valid], 'Aspect': aspect[r_valid, c_valid],
    'TWI': twi[r_valid, c_valid], 'Geology': geology[r_valid, c_valid],
    'Landcover': landcov[r_valid, c_valid], 'Label': label_map[r_valid, c_valid]
})

def target_encode(raster, df):
    enc    = df.groupby('cat')['ls'].mean().to_dict()
    mapped = np.vectorize(enc.get)(raster.flatten())
    return mapped.reshape(raster.shape).astype(np.float32)

geo_e = target_encode(geology,  pd.DataFrame({'cat': geology[valid_mask],  'ls': label_map[valid_mask]}))
lc_e  = target_encode(landcov,  pd.DataFrame({'cat': landcov[valid_mask],  'ls': label_map[valid_mask]}))

feat_dict = {'Slope': slope, 'Geology': geo_e, 'Landcover': lc_e}
names     = ['Slope', 'Geology', 'Landcover']

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Same balanced training set (seed=42 → same as run_analysis.py)
# ─────────────────────────────────────────────────────────────────────
idx_ls  = np.where(df_all['Label'] == 1)[0]
idx_bg  = np.where(df_all['Label'] == 0)[0]
idx_train = np.concatenate([
    np.random.choice(idx_ls, 250, replace=False),
    np.random.choice(idx_bg, 250, replace=False)
])

X = np.column_stack([feat_dict[n][r_valid, c_valid] for n in names])
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X[idx_train])
X_f    = scaler.transform(X)

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Train GPR and predict std dev over full grid
# ─────────────────────────────────────────────────────────────────────
print('STEP 4: Training GPR and predicting std dev (may take a few minutes)...')
gpr = GaussianProcessRegressor(
    kernel=1.0**2 * Matern(length_scale=[1.0]*3, nu=1.5),
    alpha=0.1
).fit(X_tr, df_all['Label'].iloc[idx_train])

std_winner = np.zeros(len(X_f), dtype=np.float32)
CHUNK = 50_000
for i in range(0, len(X_f), CHUNK):
    _, s = gpr.predict(X_f[i:i+CHUNK], return_std=True)
    std_winner[i:i+CHUNK] = s.astype(np.float32)
    if i % 200_000 == 0:
        print(f'  Predicted {min(i+CHUNK, len(X_f)):,}/{len(X_f):,} cells...')

print(f'  std dev range: {std_winner.min():.4f} – {std_winner.max():.4f}')

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Save corrected susceptibility_GP_std.tif
# ─────────────────────────────────────────────────────────────────────
std_map = np.full(dem_shape, np.nan, np.float32)
std_map[r_valid, c_valid] = std_winner

out_tif = str(DATA_DIR / 'susceptibility_GP_std.tif')
profile = dem_prof.copy()
profile.update(dtype=rasterio.float32, count=1, nodata=-9999)
with rasterio.open(out_tif, 'w', **profile) as dst:
    out_arr = np.where(np.isnan(std_map), -9999, std_map).astype(np.float32)
    dst.write(out_arr, 1)
print(f'  Saved corrected std raster → {out_tif}')

# ─────────────────────────────────────────────────────────────────────
# STEP 6: Build analysis DataFrame
# ─────────────────────────────────────────────────────────────────────
df_plot = pd.DataFrame({
    'Slope'    : slope[r_valid, c_valid],
    'Aspect'   : aspect[r_valid, c_valid],
    'TWI'      : twi[r_valid, c_valid],
    'Geology'  : geology[r_valid, c_valid].astype(int),
    'Landcover': landcov[r_valid, c_valid].astype(int),
    'Std'      : std_winner,
})
df_plot = df_plot[df_plot['Geology'].isin(GEO_LABELS) & df_plot['Landcover'].isin(LC_LABELS)]
print(f'  Valid cells for plotting: {len(df_plot):,}')

# Subsample for scatter (hexbin) plots
rng    = np.random.default_rng(0)
MAX_PT = 200_000
if len(df_plot) > MAX_PT:
    idx    = rng.choice(len(df_plot), MAX_PT, replace=False)
    df_sub = df_plot.iloc[idx].reset_index(drop=True)
else:
    df_sub = df_plot.copy()

# Load susceptibility mean for violin panel
with rasterio.open(str(DATA_DIR / 'susceptibility_GP_mean.tif')) as src:
    susc_raw = src.read(1).astype(np.float32)
    nd = src.nodata if src.nodata is not None else -9999
    susc_raw[susc_raw == nd] = np.nan

susc_v  = susc_raw[r_valid, c_valid]
std_v   = std_winner
ok_susc = ~np.isnan(susc_v)
susc_v  = susc_v[ok_susc]
std_v   = std_v[ok_susc]

# ─────────────────────────────────────────────────────────────────────
# STEP 7: Fig10 – uncertainty vs. covariates (3 panels, NO violin)
# ─────────────────────────────────────────────────────────────────────
print('STEP 7: Generating Fig10 (uncertainty vs covariates)...')
LABEL_SIZE = 13
TICK_SIZE  = 11
TITLE_SIZE = 14

fig = plt.figure(figsize=(18, 12))
# 6-column grid: top row 3×(2 cols each), bottom row 2×(3 cols each)
# → both rows fill full width equally
gs = gridspec.GridSpec(2, 6, figure=fig,
                       hspace=0.42, wspace=0.38,
                       left=0.06, right=0.97, top=0.92, bottom=0.08)

letters = iter('ABCDE')

# ── Top row: continuous variables – hexbin + binned mean ──────────────
cont_vars = [
    ('Slope',  'Pendiente (°)', [0, 70],   18),
    ('Aspect', 'Aspecto (°)',   [0, 360],  18),
    ('TWI',    'ITH',           [2, 18],   16),
]
y_global_max = np.percentile(df_sub['Std'], 99)

col_slices = [slice(0, 2), slice(2, 4), slice(4, 6)]
for (var, xlabel, xlim, n_bins), sl in zip(cont_vars, col_slices):
    ax = fig.add_subplot(gs[0, sl])  # each panel = 2 of 6 cols → full width
    x  = df_sub[var].values
    y  = df_sub['Std'].values

    # Hexbin (log density)
    hb = ax.hexbin(x, y, gridsize=55, cmap='YlOrRd', mincnt=1, bins='log',
                   linewidths=0.15,
                   extent=[xlim[0], xlim[1], 0, y_global_max])
    cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label('log₁₀(celdas)', fontsize=TICK_SIZE - 1)
    cb.ax.tick_params(labelsize=TICK_SIZE - 1)

    # Binned mean ± IQR
    bins   = np.linspace(xlim[0], xlim[1], n_bins + 1)
    mids   = 0.5 * (bins[:-1] + bins[1:])
    m, q25, q75 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = y[(x >= lo) & (x < hi)]
        if len(vals) > 20:
            m.append(np.mean(vals))
            q25.append(np.percentile(vals, 25))
            q75.append(np.percentile(vals, 75))
        else:
            m.append(np.nan); q25.append(np.nan); q75.append(np.nan)
    m, q25, q75 = np.array(m), np.array(q25), np.array(q75)
    ok = ~np.isnan(m)
    ax.plot(mids[ok], m[ok], color='navy', lw=2.2, zorder=5, label='Media')
    ax.fill_between(mids[ok], q25[ok], q75[ok],
                    color='navy', alpha=0.20, zorder=4, label='IQR (25–75%)')

    ax.set_xlim(xlim); ax.set_ylim(0, y_global_max)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel('Desv. estándar posterior', fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.text(0.03, 0.97, next(letters), transform=ax.transAxes,
            fontsize=TITLE_SIZE, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, ec='none'))
    if var == 'Slope':
        ax.legend(fontsize=TICK_SIZE - 1, loc='upper right', framealpha=0.8)

# ── Bottom-left: Geology boxplot (spans 2 cols → wider panel) ────────
ax_geo = fig.add_subplot(gs[1, 0:3])
codes_g  = sorted(GEO_LABELS.keys())
data_g   = [df_plot[df_plot['Geology'] == c]['Std'].values for c in codes_g]
labels_g = [GEO_LABELS[c] for c in codes_g]

order_g   = np.argsort([np.median(d) if len(d) else 0 for d in data_g])
data_g    = [data_g[i] for i in order_g]
labels_g  = [labels_g[i] for i in order_g]
colors_g  = [GEO_COLORS[i] for i in order_g]

bp = ax_geo.boxplot(data_g, patch_artist=True, notch=False, widths=0.55,
                    medianprops=dict(color='black', lw=2),
                    whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3),
                    flierprops=dict(marker='.', markersize=2, alpha=0.15))
for patch, c in zip(bp['boxes'], colors_g):
    patch.set_facecolor(c); patch.set_alpha(0.80)
for i, d in enumerate(data_g, 1):
    if len(d):
        ax_geo.scatter(i, np.mean(d), marker='D', color='black', s=45, zorder=5)

ax_geo.set_xticks(range(1, len(codes_g) + 1))
ax_geo.set_xticklabels(labels_g, rotation=28, ha='right', fontsize=TICK_SIZE)
ax_geo.set_xlabel('Geología', fontsize=LABEL_SIZE)
ax_geo.set_ylabel('Desv. estándar posterior', fontsize=LABEL_SIZE)
ax_geo.tick_params(axis='y', labelsize=TICK_SIZE)
ax_geo.scatter([], [], marker='D', color='black', s=45, label='Media')
ax_geo.legend(fontsize=TICK_SIZE - 1, loc='upper left', framealpha=0.8)
ax_geo.text(0.03, 0.97, next(letters), transform=ax_geo.transAxes,
            fontsize=TITLE_SIZE, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, ec='none'))

# ── Bottom-right: Landcover boxplot (spans 2 cols → wider panel) ─────
ax_lc = fig.add_subplot(gs[1, 3:6])
codes_l  = sorted(LC_LABELS.keys())
data_l   = [df_plot[df_plot['Landcover'] == c]['Std'].values for c in codes_l]
labels_l = [LC_LABELS[c] for c in codes_l]

order_l   = np.argsort([np.median(d) if len(d) else 0 for d in data_l])
data_l    = [data_l[i] for i in order_l]
labels_l  = [labels_l[i] for i in order_l]
colors_l  = [LC_COLORS[i] for i in order_l]

bp2 = ax_lc.boxplot(data_l, patch_artist=True, notch=False, widths=0.55,
                    medianprops=dict(color='black', lw=2),
                    whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3),
                    flierprops=dict(marker='.', markersize=2, alpha=0.15))
for patch, c in zip(bp2['boxes'], colors_l):
    patch.set_facecolor(c); patch.set_alpha(0.80)
for i, d in enumerate(data_l, 1):
    if len(d):
        ax_lc.scatter(i, np.mean(d), marker='D', color='black', s=45, zorder=5)

ax_lc.set_xticks(range(1, len(codes_l) + 1))
ax_lc.set_xticklabels(labels_l, rotation=28, ha='right', fontsize=TICK_SIZE)
ax_lc.set_xlabel('Cobertura del Suelo', fontsize=LABEL_SIZE)
ax_lc.set_ylabel('Desv. estándar posterior', fontsize=LABEL_SIZE)
ax_lc.tick_params(axis='y', labelsize=TICK_SIZE)
ax_lc.scatter([], [], marker='D', color='black', s=45, label='Media')
ax_lc.legend(fontsize=TICK_SIZE - 1, loc='upper left', framealpha=0.8)
ax_lc.text(0.03, 0.97, next(letters), transform=ax_lc.transAxes,
           fontsize=TITLE_SIZE, fontweight='bold', va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, ec='none'))

fig.suptitle(
    'Estructura de la incertidumbre del PGC en el espacio de covariables',
    fontsize=15, fontweight='bold', y=0.97)

out_path8 = FIG_DIR / 'Fig11_uncertainty_covariates.png'
plt.savefig(out_path8, dpi=300, bbox_inches='tight')
plt.close()
print(f'Fig11 saved → {out_path8}')

# ─────────────────────────────────────────────────────────────────────
# STEP 8: Fig11 – incertidumbre vs susceptibilidad (continua)
#         1 fila × 2 columnas: PGR-KDE (izq.) | PGC (der.)
# ─────────────────────────────────────────────────────────────────────
print('STEP 8: Generating Fig11 (uncertainty vs susceptibility, continuous)...')

# ── Cargar resultados del PGC ─────────────────────────────────────────
with rasterio.open(str(DATA_DIR / 'susceptibility_GP_mean.tif')) as src:
    susc_gpc_raw = src.read(1).astype(np.float32)
    nd = src.nodata if src.nodata is not None else -9999
    susc_gpc_raw[susc_gpc_raw == nd] = np.nan

gpc_susc = susc_gpc_raw[r_valid, c_valid]
gpc_std  = std_winner
ok_gpc   = ~np.isnan(gpc_susc) & ~np.isnan(gpc_std)
gpc_susc = gpc_susc[ok_gpc]
gpc_std  = gpc_std[ok_gpc]
print(f'  PGC: {len(gpc_susc):,} celdas validas  '
      f'susc [{gpc_susc.min():.3f}, {gpc_susc.max():.3f}]  '
      f'std [{gpc_std.min():.4f}, {gpc_std.max():.4f}]')

# ── Cargar resultados del PGR-KDE ─────────────────────────────────────
with rasterio.open(str(DATA_DIR / 'gpr_kde_susceptibility.tif')) as src:
    susc_kde_raw = src.read(1).astype(np.float32)
    nd = src.nodata if src.nodata is not None else -9999
    susc_kde_raw[susc_kde_raw == nd] = np.nan

with rasterio.open(str(DATA_DIR / 'gpr_kde_std.tif')) as src:
    std_kde_raw = src.read(1).astype(np.float32)
    nd = src.nodata if src.nodata is not None else -9999
    std_kde_raw[std_kde_raw == nd] = np.nan

kde_susc = susc_kde_raw[r_valid, c_valid]
kde_std  = std_kde_raw[r_valid, c_valid]
ok_kde   = ~np.isnan(kde_susc) & ~np.isnan(kde_std)
kde_susc = kde_susc[ok_kde]
kde_std  = kde_std[ok_kde]
print(f'  PGR-KDE: {len(kde_susc):,} celdas validas  '
      f'susc [{kde_susc.min():.3f}, {kde_susc.max():.3f}]  '
      f'std [{kde_std.min():.4f}, {kde_std.max():.4f}]')

# ── Función auxiliar: media y cuartiles por bins ──────────────────────
def bin_stats(x, y, n_bins=60):
    bins = np.linspace(float(x.min()), float(x.max()), n_bins + 1)
    cx, mu, q25, q75 = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() > 50:
            ys = y[m]
            cx.append((lo + hi) / 2)
            mu.append(float(np.mean(ys)))
            q25.append(float(np.percentile(ys, 25)))
            q75.append(float(np.percentile(ys, 75)))
    return (np.array(cx), np.array(mu),
            np.array(q25), np.array(q75))

# ── Figura 1×2 ───────────────────────────────────────────────────────
fig11, axes11 = plt.subplots(1, 2, figsize=(16, 6), sharey=False,
                              gridspec_kw={'wspace': 0.25})

panels = [
    (axes11[0], kde_susc, kde_std,  'PGR-KDE', '#2171b5', 'A'),
    (axes11[1], gpc_susc, gpc_std,  'PGC',     '#cb181d', 'B'),
]

hb_last = None
for ax, susc, std, title, color, letter in panels:
    hb = ax.hexbin(susc, std, gridsize=70, bins='log',
                   cmap='YlOrRd', mincnt=1, linewidths=0.2)
    hb_last = hb

    cx, mu, q25, q75 = bin_stats(susc, std)
    ax.plot(cx, mu, color=color, lw=2.5, zorder=5, label='Media binada')
    ax.fill_between(cx, q25, q75, color=color, alpha=0.28,
                    zorder=4, label='IQR (25-75%)')

    ax.set_xlabel('Susceptibilidad (predicción normalizada)',
                  fontsize=LABEL_SIZE + 1)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE, framealpha=0.85, loc='upper right')

    # Línea de media global de incertidumbre
    ax.axhline(float(np.mean(std)), color='grey', lw=1.2, ls='--', alpha=0.7,
               label=f'Media global ({np.mean(std):.3f})')

    # Etiqueta de panel
    ax.text(0.02, 0.98, f'({letter}) {title}', transform=ax.transAxes,
            fontsize=LABEL_SIZE + 1, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85, ec='none'))

    # Colorbar individual por panel
    cb = plt.colorbar(hb, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label('Celdas (log₁₀)', fontsize=TICK_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE - 1)

axes11[0].set_ylabel('Desviación estándar posterior (incertidumbre)',
                     fontsize=LABEL_SIZE + 1)
axes11[1].set_ylabel('Desviación estándar posterior (incertidumbre)',
                     fontsize=LABEL_SIZE + 1)

plt.tight_layout()
out_path9 = FIG_DIR / 'Fig12_uncertainty_vs_susceptibility.png'
plt.savefig(out_path9, dpi=300, bbox_inches='tight')
plt.close()
print(f'Fig12 saved → {out_path9}')
