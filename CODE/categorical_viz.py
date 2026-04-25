import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_cat_map(tif_path, output_path, title):
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        nodata = src.nodata
        valid_mask = data != nodata
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(data, cmap='tab20', interpolation='nearest')
        
        mask = np.isfinite(data) & (data != src.nodata)
        unique_vals = np.unique(data[mask])
        colors = [im.cmap(im.norm(v)) for v in unique_vals]
        patches = [plt.plot([], [], marker="s", ms=10, ls="", color=c, label=f"{int(v)}")[0] for v, c in zip(unique_vals, colors)]
        
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, title="Código")
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

FIG_DIR = Path('FIGURAS')
FIG_DIR.mkdir(exist_ok=True)

plot_cat_map('DATA/geology.tif', FIG_DIR / 'Review_Geology_Codes.png', 'Mapa de Códigos de Geología')
plot_cat_map('DATA/landcover.tif', FIG_DIR / 'Review_Landcover_Codes.png', 'Mapa de Códigos de Cobertura')

print("Maps generated in FIGURAS/Review_Geology_Codes.png and Review_Landcover_Codes.png")
