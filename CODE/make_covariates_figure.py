import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import pyproj
from rasterio.plot import plotting_extent
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Config
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'DATA'
FIG_DIR  = BASE_DIR / 'FIGURAS'
FIG_DIR.mkdir(exist_ok=True)

def add_cartography(ax, extent, crs, north_arrow=True, scale_bar=True, north_pos=(0.94, 0.94)):
    # Remove all coordinate and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    
    if north_arrow:
        # Move North arrow based on provided fraction
        ax.annotate('N', xy=north_pos, xytext=(north_pos[0], north_pos[1] - 0.08),
                    arrowprops=dict(facecolor='black', width=1.2, headwidth=5),
                    ha='center', va='center', fontsize=10, fontweight='bold', xycoords='axes fraction', zorder=10)
    
    if scale_bar:
        # Move scale bar to bottom-center
        scale_len_m = 2000
        center_x = extent[0] + (extent[1] - extent[0]) * 0.5
        scale_x = center_x - scale_len_m / 2
        scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
        ax.plot([scale_x, scale_x + scale_len_m], [scale_y, scale_y], color='black', linewidth=1.5, zorder=10)
        ax.text(center_x, scale_y + (extent[3] - extent[2])*0.02, '2 km',
                ha='center', va='bottom', fontsize=9, fontweight='bold', zorder=10)

def add_panel_letter(ax, letter):
    # Position inside top-left
    ax.text(0.03, 0.97, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='none'), zorder=11)

def load_resample(name, dem_shape, dem_trans, dem_crs, categorical=False):
    path = DATA_DIR / f'{name}.tif'
    with rasterio.open(str(path)) as src:
        dest = np.zeros(dem_shape, np.float32)
        reproject(rasterio.band(src, 1), dest, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dem_trans, dst_crs=dem_crs,
                  resampling=Resampling.nearest if categorical else Resampling.bilinear)
        dest[dest == src.nodata] = np.nan
        return dest

def rasterize_gpkg(path, column, dem_shape, dem_trans, dem_crs, ws_mask):
    map_gdf = gpd.read_file(path).to_crs(dem_crs)
    shapes = ((geom, val) for geom, val in zip(map_gdf.geometry, map_gdf[column]))
    raster = rasterize(shapes, out_shape=dem_shape, transform=dem_trans, fill=0, all_touched=True, dtype=np.float32)
    raster[~ws_mask] = np.nan
    return raster

def main():
    print("STEP 1: Loading & Preparing data...")
    DEM_PATH = DATA_DIR / 'DEM_5m.tif'
    WS_PATH  = DATA_DIR / 'Cuenca_Iguana.shp'
    
    ws = gpd.read_file(WS_PATH)
    with rasterio.open(DEM_PATH) as src:
        dem_crs = src.crs; dem_trans = src.transform; dem_shape = src.shape
        dem_extent = plotting_extent(src)
    
    ws_proj = ws.to_crs(dem_crs)
    ws_mask = geometry_mask(ws_proj.geometry, transform=dem_trans, invert=True, out_shape=dem_shape)
    
    # Continuous variables
    slope = load_resample('slope_5m', dem_shape, dem_trans, dem_crs)
    aspect = load_resample('aspect_5m', dem_shape, dem_trans, dem_crs)
    twi = load_resample('twi_5m', dem_shape, dem_trans, dem_crs)
    
    # Categorical variables
    geology = rasterize_gpkg(DATA_DIR / 'geosurface_map.gpkg', 'Codigo', dem_shape, dem_trans, dem_crs, ws_mask)
    landcover = rasterize_gpkg(DATA_DIR / 'landcover_map.gpkg', 'code', dem_shape, dem_trans, dem_crs, ws_mask)
    
    # Masking continuous
    slope[~ws_mask] = np.nan
    aspect[~ws_mask] = np.nan
    twi[~ws_mask] = np.nan
    
    # Plotting
    print("STEP 2: Generating 5-variable figure with final aesthetic refinements...")
    fig = plt.figure(figsize=(18, 11))
    # 2 rows, 6 columns to allow for centering 2 plots in the bottom row
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.1, wspace=0.2)
    
    # Axes for Row 1 (Continuous)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    
    # Axes for Row 2 (Categorical, centered)
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])
    
    def setup_colorbar(im, ax, label, data):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)
        # Bringing label closer to the bar (negative labelpad)
        cbar.set_label(label, fontsize=11, labelpad=-18) 
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        cbar.set_ticks([vmin, vmax])
        cbar.ax.set_yticklabels([f'{vmin:.1f}', f'{vmax:.1f}'], fontsize=9)
        return cbar

    # 1. Slope
    im1 = ax1.imshow(slope, cmap='viridis', extent=dem_extent)
    setup_colorbar(im1, ax1, 'Pendiente (°)', slope)
    add_panel_letter(ax1, 'A'); add_cartography(ax1, dem_extent, dem_crs)
    
    # 2. Aspect
    im2 = ax2.imshow(aspect, cmap='twilight', extent=dem_extent)
    setup_colorbar(im2, ax2, 'Aspecto (°)', aspect)
    add_panel_letter(ax2, 'B'); add_cartography(ax2, dem_extent, dem_crs)
    
    # 3. TWI
    im3 = ax3.imshow(twi, cmap='GnBu', extent=dem_extent)
    setup_colorbar(im3, ax3, 'ITH', twi)
    add_panel_letter(ax3, 'C'); add_cartography(ax3, dem_extent, dem_crs)
    
    # Categorical Setup
    geo_colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']
    lc_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    geo_labels = {1: 'Depósitos', 2: 'Metamórficas', 3: 'Volcánicas (S)', 4: 'Graníticas', 5: 'Volcánicas (V)'}
    lc_labels = {1: 'Bosque', 2: 'Pastos', 3: 'Urbano (F)', 4: 'Urbano (I)', 5: 'Condominios'}
    
    def plot_categorical(ax, data, colors, labels, label_letter):
        unique_vals = sorted([v for v in np.unique(data[~np.isnan(data)]) if v != 0])
        n_vals = len(unique_vals)
        cmap = ListedColormap(colors[:n_vals])
        ax.imshow(data, cmap=cmap, extent=dem_extent)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=labels.get(val, f'Val {val}')) for i, val in enumerate(unique_vals)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True, framealpha=0.8, edgecolor='none', labelspacing=0.2, handlelength=1.0)
        add_panel_letter(ax, label_letter)
        # Move North arrow to middle-right for categorical maps
        add_cartography(ax, dem_extent, dem_crs, north_pos=(0.94, 0.5))

    plot_categorical(ax4, geology, geo_colors, geo_labels, 'D')
    plot_categorical(ax5, landcover, lc_colors, lc_labels, 'E')
    
    output_path = FIG_DIR / 'Fig_covariates.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSUCCESS: Final refined figure saved to {output_path}")

if __name__ == "__main__":
    main()
