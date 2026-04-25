import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

def reclassify_landcover():
    input_path = 'DATA/landcover.tif'
    output_gpkg = 'DATA/landcover_map.gpkg'
    output_lookup = 'DATA/landcover_lookup.txt'
    
    mapping = {
        1: 1,  # Bosque
        2: 2, 3: 2, 4: 2,  # Pastos
        7: 3, 10: 3,  # Urbano Formal
        9: 4, 11: 4,  # Urbano Informal
        6: 5, 8: 5,  # Condominios
        5: 0   # Water
    }
    
    labels = {
        1: "Bosque y Matorral",
        2: "Pastos y Suelos Desnudos",
        3: "Urbano Formal e Infraestructura",
        4: "Urbano Informal y Disperso",
        5: "Condominios y Fincas"
    }

    print(f"Reading {input_path}...")
    with rasterio.open(input_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        
        # Initialize reclassified array with zeros
        reclassed = np.zeros_like(data, dtype=np.uint8)
        
        # Apply mapping for valid pixels
        mask = (data != nodata)
        for old_val, new_val in mapping.items():
            reclassed[mask & (data == old_val)] = new_val
            
        # Create lookup table file
        with open(output_lookup, 'w') as f:
            f.write("Code\tLabel\n")
            for code, label in sorted(labels.items()):
                f.write(f"{code}\t{label}\n")
        print(f"Lookup table saved to {output_lookup}")

        # Vectorize
        print("Vectorizing raster (this may take a moment)...")
        # Only vectorize values 1-5 (excluding 0/water/nodata)
        mask_vector = (reclassed > 0) & (reclassed <= 5)
        results = (
            {'properties': {'code': int(v)}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(reclassed, mask=mask_vector, transform=transform))
        )
        
        # Create GeoDataFrame
        geoms = list(results)
        if not geoms:
            print("No valid areas found to vectorize.")
            return

        gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
        
        # Dissolve by code to simplify
        print("Dissolving polygons by code...")
        gdf = gdf.dissolve(by='code').reset_index()
        
        # Add labels
        gdf['label'] = gdf['code'].map(labels)
        
        # Save to GPKG
        gdf.to_file(output_gpkg, driver="GPKG")
        print(f"Reclassified landcover saved to {output_gpkg}")

if __name__ == "__main__":
    reclassify_landcover()
