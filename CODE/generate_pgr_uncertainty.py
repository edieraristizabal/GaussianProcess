"""Generate PGR-KDE uncertainty figure from saved raster (gpr_kde_std.tif)."""
import numpy as np
import rasterio
import geopandas as gpd
import pyproj
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.plot import plotting_extent
from rasterio.features import geometry_mask
from pathlib import Path

BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'DATA'
FIG_DIR  = BASE_DIR / 'FIGURAS'

# Load DEM metadata for extent and CRS
with rasterio.open(str(DATA_DIR / 'DEM_5m.tif')) as src:
    dem_trans = src.transform
    dem_shape = src.shape
    dem_crs   = src.crs

# Load watershed
ws = gpd.read_file(str(DATA_DIR / 'Cuenca_Iguana.shp')).to_crs(dem_crs)
ws_mask = geometry_mask(ws.geometry, transform=dem_trans, invert=True, out_shape=dem_shape)

# Load PGR-KDE std raster
with rasterio.open(str(DATA_DIR / 'gpr_kde_std.tif')) as src:
    gpr_std_arr = src.read(1).astype(np.float32)
    nodata = src.nodata
    if nodata is not None:
        gpr_std_arr[gpr_std_arr == nodata] = np.nan

gpr_std_arr[~ws_mask] = np.nan
ext = plotting_extent(np.zeros(dem_shape), dem_trans)

def add_cartography(ax, extent, crs):
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon_breaks = np.arange(-75.7, -75.5, 0.02)
    lat_breaks = np.arange(6.24, 6.36, 0.02)
    x_ticks, _ = transformer.transform(lon_breaks, np.full_like(lon_breaks, 6.3))
    _, y_ticks  = transformer.transform(np.full_like(lat_breaks, -75.6), lat_breaks)
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
    ax.text(scale_x + scale_len_m / 2, scale_y + (extent[3] - extent[2]) * 0.02, '2 km',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(gpr_std_arr, cmap='viridis', extent=ext)
cbar = plt.colorbar(im, ax=ax, fraction=0.035)
cbar.set_label('Desviación Estándar (Incertidumbre PGR)', fontsize=16)
cbar.ax.tick_params(labelsize=14)
add_cartography(ax, ext, dem_crs)
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'Fig_pgr_std.png'), dpi=300)
plt.close()
print('Saved: FIGURAS/Fig_pgr_std.png')
