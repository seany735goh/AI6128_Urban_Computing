import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import osmnx as ox
from matplotlib import pyplot as plt
from shapely.geometry import Polygon

folder = '.'


def plot_graph(results_df, G, file_name):

    fig, ax = ox.plot_graph(G, figsize=(30,20), show=False, node_size=0, bgcolor='w', edge_color="k", edge_linewidth=0.5)
    for i, mgeom in zip(results_df.index, results_df['match_geom'].values):
        mgeom = mgeom.lstrip('LINESTRING(').rstrip(')')
        mgeom = mgeom.split(',')
        mgeom_lon_lat = [m.split(' ') for m in mgeom]
        lon = [float(l) for l in np.array(mgeom_lon_lat)[:,0] ]
        lat = [float(l) for l in np.array(mgeom_lon_lat)[:,1] ]
        plt.plot(lon, lat, marker='o', label=f'Trip {i+1}')
    plt.legend(fontsize=25)
    plt.savefig(os.path.join(folder, file_name))
    plt.show()


if __name__ == "__main__":

    # Create the street network within the bounding box of Porto's borders
    bounds = (-8.70, -8.57, 41.19, 41.13)
    x1,x2,y1,y2 = bounds
    boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    G = ox.graph_from_polygon(boundary_polygon, network_type='drive')

    # Plot FMM results
    results_fmm_df = pd.read_csv(os.path.join(folder, 'fmm_mapped_trajectory.csv'))
    results_fmm_df = results_fmm_df.head(10)
    plot_graph(results_fmm_df, G, 'FastMapMatch - First 10 Trips')

    # Plot STMatch results
    results_stm_df = pd.read_csv(os.path.join(folder, 'stm_mapped_trajectory.csv'))
    results_stm_df = results_stm_df.head(10)
    plot_graph(results_stm_df, G, 'STMatch - First 10 Trips')
