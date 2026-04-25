"""Regenerate Fig7 (PGR-KDE susceptibility) and Fig9 (PGR-KDE std uncertainty)."""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject, Resampling
from rasterio.plot import plotting_extent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import gaussian_kde as scipy_kde
import pyproj
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'DATA'
FIG_DIR  = BASE_DIR / 'FIGURAS'
np.random.seed(42)

chunk = 50000

def add_cartography(ax, extent, crs):
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon_breaks = np.arange(-75.7, -75.5, 0.02)
    lat_breaks = np.arange(6.24, 6.36, 0.02)
    x_ticks, _ = transformer.transform(lon_breaks, np.full_like(lon_breaks, 6.3))
    _, y_ticks  = transformer.transform(np.full_like(lat_breaks, -75.6), lat_breaks)
    ax.set_xticks(x_ticks); ax.set_yticks(y_ticks)
    ax.set_xticklabels([f'{abs(l):.2f}°W' for l in lon_breaks], fontsize=12)
    ax.set_yticklabels([f'{l:.2f}°N' for l in lat_breaks], fontsize=12)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.88),
                arrowprops=dict(facecolor='black', width=3, headwidth=10),
                ha='center', va='center', fontsize=14, fontweight='bold',
                xycoords='axes fraction')
    scale_len_m = 2000
    scale_x = extent[0] + (extent[1] - extent[0]) * 0.4
    scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
    ax.plot([scale_x, scale_x + scale_len_m], [scale_y, scale_y],
            color='black', linewidth=3)
    ax.text(scale_x + scale_len_m / 2,
            scale_y + (extent[3] - extent[2]) * 0.02, '2 km',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# ── 1. Load rasters ───────────────────────────────────────────────────
print('Loading data...')
ws = gpd.read_file(str(DATA_DIR / 'Cuenca_Iguana.shp'))
with rasterio.open(str(DATA_DIR / 'DEM_5m.tif')) as src:
    dem_crs = src.crs; dem_trans = src.transform
    dem_shape = src.shape; dem_prof = src.profile

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
    gdf    = gpd.read_file(path).to_crs(dem_crs)
    ws_g   = ws.union_all()
    gdf_in = gdf[gdf.intersects(ws_g)]
    shapes = ((geom, val) for geom, val in zip(gdf_in.geometry, gdf_in[column]))
    raster = rasterize(shapes, out_shape=dem_shape, transform=dem_trans,
                       fill=0, all_touched=True, dtype=np.float32)
    raster[~ws_mask] = np.nan
    return raster

slope    = load_resample('slope_5m')
aspect   = load_resample('aspect_5m')
twi      = load_resample('twi_5m')
geology  = rasterize_gpkg(DATA_DIR / 'geosurface_map.gpkg', column='Codigo')
landcov  = rasterize_gpkg(DATA_DIR / 'landcover_map.gpkg',  column='code')

valid_mask = ws_mask & ~np.isnan(slope) & ~np.isnan(geology) & ~np.isnan(landcov)
r_valid, c_valid = np.where(valid_mask)

inv = gpd.read_file(str(DATA_DIR / 'Deslizamientos_Iguana.gpkg'),
                    layer='Deslizamientos_Iguana').to_crs(dem_crs)
ls_r, ls_c = [], []
for geom in inv.geometry:
    r, c = rowcol(dem_trans, geom.x, geom.y)
    if 0 <= r < dem_shape[0] and 0 <= c < dem_shape[1] and valid_mask[r, c]:
        ls_r.append(r); ls_c.append(c)
label_map = np.zeros(dem_shape, np.int8); label_map[ls_r, ls_c] = 1
y_true = label_map[r_valid, c_valid]
print(f'  Valid cells: {valid_mask.sum():,} | Landslides: {len(ls_r)}')

# ── 2. Target encoding + training set ────────────────────────────────
def target_encode(raster, df):
    enc    = df.groupby('cat')['ls'].mean().to_dict()
    mapped = np.vectorize(enc.get)(raster.flatten())
    return mapped.reshape(raster.shape).astype(np.float32)

df_all = pd.DataFrame({'Slope': slope[r_valid, c_valid],
                       'Geology': geology[r_valid, c_valid],
                       'Landcover': landcov[r_valid, c_valid],
                       'Label': y_true})
geo_e = target_encode(geology, pd.DataFrame({'cat': geology[valid_mask],
                                              'ls': label_map[valid_mask]}))
lc_e  = target_encode(landcov, pd.DataFrame({'cat': landcov[valid_mask],
                                              'ls': label_map[valid_mask]}))
feat_dict = {'Slope': slope, 'Geology': geo_e, 'Landcover': lc_e}
names_gpr = ['Slope', 'Geology', 'Landcover']

idx_ls = np.where(df_all['Label'] == 1)[0]
idx_bg = np.where(df_all['Label'] == 0)[0]
idx_train = np.concatenate([np.random.choice(idx_ls, 250, replace=False),
                             np.random.choice(idx_bg, 250, replace=False)])

# ── 3. KDE target ─────────────────────────────────────────────────────
print('Computing KDE...')
ls_x = np.array([dem_trans.c + (c + 0.5) * dem_trans.a for c in ls_c])
ls_y = np.array([dem_trans.f + (r + 0.5) * dem_trans.e for r in ls_r])
all_x = np.array([dem_trans.c + (c + 0.5) * dem_trans.a for c in c_valid])
all_y = np.array([dem_trans.f + (r + 0.5) * dem_trans.e for r in r_valid])
kde_func = scipy_kde(np.vstack([ls_x, ls_y]), bw_method='silverman')
kde_all = np.zeros(len(c_valid), dtype=np.float32)
for i in range(0, len(c_valid), chunk):
    kde_all[i:i+chunk] = kde_func(np.vstack([all_x[i:i+chunk], all_y[i:i+chunk]]))
kde_all = (kde_all - kde_all.min()) / (kde_all.max() - kde_all.min())

# ── 4. Train PGR-KDE ──────────────────────────────────────────────────
print('Training PGR-KDE...')
X_gpr      = np.column_stack([feat_dict[n][r_valid, c_valid] for n in names_gpr])
scaler_gpr = StandardScaler()
X_tr_gpr   = scaler_gpr.fit_transform(X_gpr[idx_train])
X_f_gpr    = scaler_gpr.transform(X_gpr)

gpr_kde = GaussianProcessRegressor(
    kernel=1.0**2 * Matern(length_scale=[1.0]*3, nu=1.5),
    alpha=0.01, n_restarts_optimizer=3, normalize_y=True)
gpr_kde.fit(X_tr_gpr, kde_all[idx_train])

print('Predicting PGR-KDE...')
gpr_pred    = np.zeros(len(X_f_gpr), dtype=np.float32)
gpr_std_kde = np.zeros(len(X_f_gpr), dtype=np.float32)
for i in range(0, len(X_f_gpr), chunk):
    if i % (10 * chunk) == 0 and i > 0:
        print(f'  {i:,}/{len(X_f_gpr):,}...')
    p, s = gpr_kde.predict(X_f_gpr[i:i+chunk], return_std=True)
    gpr_pred[i:i+chunk]    = p
    gpr_std_kde[i:i+chunk] = s

gpr_pred_clipped = np.clip(gpr_pred, 0, None)

# ── 5. Fig 7 – PGR-KDE susceptibility (raw values, real min–max scale) ─
print('Generating Fig7...')
cmap_susc = LinearSegmentedColormap.from_list('susc',
    ['#238b45','#74c476','#ffeda0','#feb24c','#e31a1c'])
ext = plotting_extent(np.zeros(dem_shape), dem_trans)

gpr_kde_map = np.full(dem_shape, np.nan, np.float32)
gpr_kde_map[r_valid, c_valid] = gpr_pred_clipped
vmin_pgr = 0.2
vmax_pgr = 0.6
print(f'  PGR-KDE susceptibility range (fixed): {vmin_pgr} – {vmax_pgr}')

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(gpr_kde_map, cmap=cmap_susc, vmin=vmin_pgr, vmax=vmax_pgr, extent=ext)
cbar = plt.colorbar(im, ax=ax, fraction=0.035)
cbar.set_label('Susceptibilidad PGR-KDE', fontsize=16)
cbar.ax.tick_params(labelsize=14)
add_cartography(ax, ext, dem_crs)
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig7_gpr_susceptibility.png'), dpi=300)
plt.close()
print('  Saved: FIGURAS/Fig7_gpr_susceptibility.png')

# ── 6. Fig 9 – PGR-KDE std (real min–max scale) ───────────────────────
print('Generating Fig9 (Fig_pgr_std)...')
gpr_std_map = np.full(dem_shape, np.nan, np.float32)
gpr_std_map[r_valid, c_valid] = gpr_std_kde
vmin_std = 0.25
vmax_std = float(np.nanmax(gpr_std_map))
print(f'  PGR-KDE std range (fixed min): {vmin_std} – {vmax_std:.4f}')

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(gpr_std_map, cmap='viridis', vmin=vmin_std, vmax=vmax_std, extent=ext)
cbar = plt.colorbar(im, ax=ax, fraction=0.035)
cbar.set_label('Desviación Estándar (Incertidumbre PGR)', fontsize=16)
cbar.ax.tick_params(labelsize=14)
add_cartography(ax, ext, dem_crs)
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig_pgr_std.png'), dpi=300)
plt.close()
print('  Saved: FIGURAS/Fig_pgr_std.png')

print('Done.')
