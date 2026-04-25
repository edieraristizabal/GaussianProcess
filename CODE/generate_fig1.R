library(sf)
library(terra)
library(ggplot2)
library(tidyterra)
library(ggspatial)
library(cowplot)

# Paths
data_dir <- "DATA"
fig_dir <- "FIGURAS"
dir.create(fig_dir, showWarnings = FALSE)

print("Loading layers...")
dem <- rast(file.path(data_dir, "DEM_10m.tif"))
ws <- st_read(file.path(data_dir, "Cuenca_Iguana.shp"), quiet = TRUE)
ls_pts <- st_read(file.path(data_dir, "Deslizamientos_Iguana.gpkg"), quiet = TRUE)
drainage <- st_read(file.path(data_dir, "Drenajes.shp"), quiet = TRUE)
drainage <- drainage[drainage$grid_code >= 6, ]
aburra_path <- "/home/edier/Documents/INVESTIGACION/PAPERS/ELABORACION/landinv_aburrav/DATA/boundary.gpkg"
aburra <- st_read(aburra_path, quiet = TRUE)

# Ensure same CRS
dem_crs <- crs(dem)
ws <- st_transform(ws, dem_crs)
ls_pts <- st_transform(ls_pts, dem_crs)
drainage <- st_transform(drainage, dem_crs)
aburra <- st_transform(aburra, dem_crs)

print("Generating Hillshade...")
slp <- terrain(dem, "slope", unit = "radians")
asp <- terrain(dem, "aspect", unit = "radians")
h1 <- shade(slp, asp, angle = 45, direction = 225)
h2 <- shade(slp, asp, angle = 45, direction = 315)
h3 <- shade(slp, asp, angle = 45, direction = 135)
hsh <- (h1 + h2 + h3) / 3

print("Creating Main Plot...")
p_main <- ggplot() +
  geom_spatraster(data = hsh, show.legend = FALSE, alpha = 0.9) +
  scale_fill_gradient(low = "gray40", high = "white", na.value = "transparent") +
  geom_sf(data = drainage, color = "steelblue", linewidth = 0.4, alpha = 0.8) +
  geom_sf(data = ws, fill = NA, color = "black", linewidth = 0.8) +
  geom_sf(data = ls_pts, color = "red", size = 0.6, alpha = 0.6) +
  coord_sf(datum = st_crs(4326), expand = FALSE) + # WGS84 Grid
  scale_x_continuous(breaks = seq(-76, -75, by = 0.02)) +
  scale_y_continuous(breaks = seq(6, 7, by = 0.02)) +
  annotation_scale(location = "bl", pad_x = unit(0.4, "npc"), width_hint = 0.2) +
  annotation_north_arrow(location = "tl", which_north = "true", 
                         style = north_arrow_fancy_orienteering()) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.title = element_blank(),
    axis.text = element_text(size = 12.8)
  )

print("Creating Inset Plot...")
# Simplify ws for inset
ws_sim <- st_simplify(st_union(ws), dTolerance = 50)
p_inset <- ggplot() +
  geom_sf(data = aburra, fill = "gray90", color = "gray50", linewidth = 0.1) +
  geom_sf(data = ws_sim, fill = "red", color = "red") +
  theme_void() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    plot.margin = margin(0,0,0,0)
  )

print("Combining and Saving...")
final_plot <- ggdraw(p_main) +
  draw_plot(p_inset, x = 0.78, y = 0.65, width = 0.175, height = 0.175)

ggsave(file.path(fig_dir, "Fig1_study_area.png"), final_plot, width = 8, height = 8, dpi = 300)
print("Fig1_study_area.png saved successfully.")
