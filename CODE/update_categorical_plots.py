import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.features import geometry_mask
import numpy as np
from pathlib import Path

def load_lookup(path):
    # Use any whitespace as separator and handle headers
    df = pd.read_csv(path, sep=None, engine='python')
    # Standardize column names
    df.columns = ['Code', 'Label']
    return df.set_index('Code')['Label'].to_dict()

def update_plots():
    DATA_DIR = Path('DATA')
    FIG_DIR = Path('FIGURAS')
    FIG_DIR.mkdir(exist_ok=True)
    
    # Load landslide inventory
    inv = gpd.read_file(DATA_DIR / 'Deslizamientos_Iguana.gpkg', layer='Deslizamientos_Iguana')
    # Load basin mask for area calculation
    dem_path = DATA_DIR / 'DEM_5m.tif'
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        transform = src.transform
        shape = src.shape
    
    ws = gpd.read_file(DATA_DIR / 'Cuenca_Iguana.shp').to_crs(dem_crs)
    ws_mask = geometry_mask(ws.geometry, transform=transform, invert=True, out_shape=shape)
    
    # Files to process
    configs = [
        {
            'name': 'Geología',
            'map': DATA_DIR / 'geosurface_map.gpkg',
            'lookup': DATA_DIR / 'geosurface_legend.txt',
            'col': 'Codigo'
        },
        {
            'name': 'Cobertura',
            'map': DATA_DIR / 'landcover_map.gpkg',
            'lookup': DATA_DIR / 'landcover_lookup.txt',
            'col': 'code'
        }
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, cfg in zip(axes, configs):
        print(f"Processing {cfg['name']}...")
        
        # Load map and lookup
        map_df = gpd.read_file(cfg['map']).to_crs(dem_crs)
        # Ensure code is int
        map_df[cfg['col']] = map_df[cfg['col']].astype(int)
        lookup = load_lookup(cfg['lookup'])
        
        # 1. Basin Distribution (using geometry areas)
        # Intersect map with basin
        ws_geom = ws.union_all()
        map_in_ws = map_df.copy()
        map_in_ws['geometry'] = map_in_ws.intersection(ws_geom)
        map_in_ws = map_in_ws[~map_in_ws.is_empty]
        
        # Compute area for each feature
        map_in_ws['area'] = map_in_ws.geometry.area
        basin_areas = map_in_ws.groupby(cfg['col'])['area'].sum().to_dict()
        total_area = sum(basin_areas.values())
        basin_prop = {code: area/total_area for code, area in basin_areas.items()}
        
        # 2. Landslide Distribution (spatial join)
        # Convert landslides to centroids if they are polygons, or just points
        inv_proj = inv.to_crs(dem_crs)
        # Assuming inventory is points. If polygons, use centroids for category sampling
        if (inv_proj.geometry.type == 'Polygon').any() or (inv_proj.geometry.type == 'MultiPolygon').any():
            inv_points = inv_proj.copy()
            inv_points['geometry'] = inv_proj.geometry.centroid
        else:
            inv_points = inv_proj
            
        inv_in_ws = inv_points[inv_points.intersects(ws_geom)]
        
        # Spatial join to get category for each landslide
        slides_with_cat = gpd.sjoin(inv_in_ws, map_df[[cfg['col'], 'geometry']], how='left', predicate='intersects')
        slide_counts = slides_with_cat[cfg['col']].value_counts().to_dict()
        total_slides = sum(slide_counts.values())
        slide_prop = {code: count/total_slides for code, count in slide_counts.items() if not np.isnan(code)}
        
        # 3. Combine for plotting
        all_codes = sorted(list(set(basin_prop.keys()) | set(slide_prop.keys())))
        plot_data = []
        for code in all_codes:
            if code == 0: continue # Skip water/excluded
            label = lookup.get(code, f"Code {code}")
            plot_data.append({'Category': label, 'Proporción': basin_prop.get(code, 0), 'Tipo': 'Cuenca'})
            plot_data.append({'Category': label, 'Proporción': slide_prop.get(code, 0), 'Tipo': 'Deslizamiento'})
            
        df_plot = pd.DataFrame(plot_data)
        
        # Plot
        sns.barplot(data=df_plot, x='Category', y='Proporción', hue='Tipo', palette=['steelblue', 'firebrick'], alpha=0.8, ax=ax)
        ax.set_ylabel('Proporción (Índice de Frecuencia)', fontsize=18)
        ax.set_xlabel('', fontsize=18)
        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(fontsize=14)
        
    plt.tight_layout()
    output_path = FIG_DIR / 'Fig4_distributions_categorical.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    update_plots()
