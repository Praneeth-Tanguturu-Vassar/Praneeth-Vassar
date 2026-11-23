# postprocessor.py — KD-tree neighbour majority-vote to fix isolated mis-labels
#
# CocoPalmPostProcessor.quick_fix(gdf):
#   1. Computes centroid of every detected tree box
#   2. Estimates typical tree spacing from nearest-neighbour distances
#   3. For each tree, finds all neighbours within 3.5× median spacing
#   4. If 75%+ of those neighbours are class X and this tree is not X → relabel it
#   This corrects isolated palm↔coconut swaps without needing a re-run of the model

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter
import pandas as pd
from shapely.geometry import Point


class CocoPalmPostProcessor:
    def __init__(self):
        pass

    def _get_centroids(self):
        centroids = []
        for geom in self.gdf.geometry:
            if hasattr(geom, 'centroid'):
                centroids.append([geom.centroid.x, geom.centroid.y])
            else:
                centroids.append([geom.x, geom.y])
        return np.array(centroids)

    def find_neighbors_in_radius(self, radius):
        tree = cKDTree(self.centroids)
        return tree.query_ball_tree(tree, radius)

    def estimate_tree_spacing(self, sample_size=1000):
        sample_indices = np.random.choice(len(self.centroids),
                                          min(sample_size, len(self.centroids)),
                                          replace=False)
        distances = []
        tree = cKDTree(self.centroids)
        for idx in sample_indices:
            dists, _ = tree.query(self.centroids[idx], k=6)
            distances.extend(dists[1:])
        distances = np.array(distances)
        median_spacing   = np.median(distances)
        suggested_radius = median_spacing * 3.5
        return median_spacing, suggested_radius

    def correct_isolated_misclassifications(self, search_radius=None,
                                            min_neighbors=8, majority_threshold=0.8):
        if search_radius is None:
            _, search_radius = self.estimate_tree_spacing()
        neighbors_list = self.find_neighbors_in_radius(search_radius)
        corrected_gdf  = self.gdf.copy()
        corrected_gdf['corrected']  = False
        corrected_gdf['orig_class'] = corrected_gdf['class'].copy()
        corrected_gdf['nbr_count']  = 0
        corrected_gdf['maj_class']  = ''
        corrected_gdf['maj_ratio']  = 0.0
        corrections_made = 0
        for i, neighbors in enumerate(neighbors_list):
            neighbors = [n for n in neighbors if n != i]
            if len(neighbors) < min_neighbors:
                continue
            neighbor_classes = [self.gdf.iloc[n]['class'] for n in neighbors]
            class_counts     = Counter(neighbor_classes)
            if len(class_counts) == 0:
                continue
            majority_class = class_counts.most_common(1)[0][0]
            majority_count = class_counts[majority_class]
            majority_ratio = majority_count / len(neighbors)
            corrected_gdf.iloc[i, corrected_gdf.columns.get_loc('nbr_count')] = len(neighbors)
            corrected_gdf.iloc[i, corrected_gdf.columns.get_loc('maj_class')] = majority_class
            corrected_gdf.iloc[i, corrected_gdf.columns.get_loc('maj_ratio')] = majority_ratio
            current_class = self.gdf.iloc[i]['class']
            if current_class != majority_class and majority_ratio >= majority_threshold:
                corrected_gdf.iloc[i, corrected_gdf.columns.get_loc('class')] = majority_class
                corrected_gdf.iloc[i, corrected_gdf.columns.get_loc('corrected')] = True
                corrections_made += 1
        return corrected_gdf

    def analyze_neighborhood_patterns(self, search_radius=None, sample_size=500):
        if search_radius is None:
            _, search_radius = self.estimate_tree_spacing()
        sample_indices = np.random.choice(len(self.gdf),
                                          min(sample_size, len(self.gdf)), replace=False)
        neighbors_list   = self.find_neighbors_in_radius(search_radius)
        analysis_results = []
        for i in sample_indices:
            neighbors = [n for n in neighbors_list[i] if n != i]
            if len(neighbors) < 3:
                continue
            current_class    = self.gdf.iloc[i]['class']
            neighbor_classes = [self.gdf.iloc[n]['class'] for n in neighbors]
            class_counts     = Counter(neighbor_classes)
            if len(class_counts) == 1:
                homogeneity    = 1.0
                majority_class = list(class_counts.keys())[0]
            else:
                majority_class = class_counts.most_common(1)[0][0]
                homogeneity    = class_counts[majority_class] / len(neighbors)
            analysis_results.append({
                'tree_class':          current_class,
                'neighbor_count':      len(neighbors),
                'majority_class':      majority_class,
                'homogeneity':         homogeneity,
                'is_outlier':          current_class != majority_class,
                'potential_correction': current_class != majority_class and homogeneity >= 0.8,
            })
        return pd.DataFrame(analysis_results)

    def quick_fix(self, gdf, shp_path=None):
        if shp_path is not None:
            self.gdf = gpd.read_file(shp_path)
        else:
            self.gdf = gdf.copy()
        self.centroids = self._get_centroids()
        _, suggested_radius = self.estimate_tree_spacing()
        self.analyze_neighborhood_patterns(suggested_radius)
        corrected_gdf = self.correct_isolated_misclassifications(
            search_radius=suggested_radius,
            min_neighbors=10,
            majority_threshold=0.75,
        )
        col_rename_map = {
            'original_class': 'orig_class',
            'neighbor_count': 'nbr_count',
            'majority_class': 'maj_class',
            'majority_ratio': 'maj_ratio',
        }
        return corrected_gdf.rename(columns=col_rename_map)
