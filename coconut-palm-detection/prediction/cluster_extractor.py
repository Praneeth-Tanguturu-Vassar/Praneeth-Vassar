# cluster_extractor.py — hex-grid plantation cluster filter
#
# ClusterExtractor.extract_clusters(gdf):
#   1. Builds a hexagonal grid over the detection bounding box
#   2. Counts how many tree centroids fall in each hex cell
#   3. Keeps only cells with >= min_trees_per_hex trees (plantation density filter)
#   4. Returns the subset of trees that fall inside dense hex cells
#   This removes isolated stray detections and retains plantation blocks

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


class ClusterExtractor:
    def __init__(self, hex_size=15, min_trees_per_hex=3):
        self.hex_size          = hex_size           # metres per hex cell
        self.min_trees_per_hex = min_trees_per_hex  # minimum density threshold

    def _generate_hex_grid(self, bounds, crs):
        xmin, ymin, xmax, ymax = bounds
        width  = self.hex_size
        height = np.sqrt(3) / 2 * width
        cols   = int((xmax - xmin) / width) + 3
        rows   = int((ymax - ymin) / height) + 3
        polygons = []
        for col in range(cols):
            for row in range(rows):
                x_offset = xmin + col * width
                y_offset = ymin + row * height
                if col % 2 == 1:
                    y_offset += height / 2
                hexagon = Polygon([
                    (x_offset + width * np.cos(np.radians(angle)),
                     y_offset + width * np.sin(np.radians(angle)))
                    for angle in range(0, 360, 60)
                ])
                polygons.append(hexagon)
        return gpd.GeoDataFrame(geometry=polygons, crs=crs)

    def extract_clusters(self, trees_gdf):
        if trees_gdf.empty or trees_gdf.crs is None:
            return trees_gdf.iloc[0:0]
        if 'geometry' not in trees_gdf or not all(trees_gdf.geometry.type == 'Polygon'):
            raise ValueError("GeoDataFrame must contain only Polygon geometries.")

        trees_gdf = trees_gdf.copy()
        trees_gdf['centroid'] = trees_gdf.geometry.centroid
        centroids = gpd.GeoDataFrame(geometry=trees_gdf['centroid'], crs=trees_gdf.crs)

        hexgrid = self._generate_hex_grid(centroids.total_bounds, centroids.crs)
        joined  = gpd.sjoin(centroids, hexgrid, how='left', predicate='intersects')
        counts  = joined.groupby('index_right').size()
        hexgrid['tree_count'] = counts.reindex(hexgrid.index).fillna(0).astype(int)

        dense_hexes = hexgrid[hexgrid['tree_count'] >= self.min_trees_per_hex].copy()
        if dense_hexes.empty:
            return trees_gdf.iloc[0:0]

        cluster_area    = unary_union(dense_hexes.geometry.buffer(self.hex_size * 0.3))
        filtered_trees  = trees_gdf[trees_gdf['centroid'].intersects(cluster_area)].copy()
        filtered_trees.drop(columns=['centroid'], inplace=True, errors='ignore')
        print('Cluster extraction done.')
        return filtered_trees
