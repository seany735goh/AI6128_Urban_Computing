import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import osmnx as ox
from shapely.geometry import Polygon

folder = '.'

# turn it back on and turn on/off logging to your console
ox.settings.use_cache = True
ox.settings.log_console = False


def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)

    # We need an unique ID for each edge
    gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype='int')

    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)


if __name__ == "__main__":
    # Read processed data
    train_1000 = pd.read_csv('Train_1000_processed.csv').to_numpy()

    # Create the street network within the bounding box of Porto's borders
    bounds = (-8.70, -8.57, 41.19, 41.13)
    x1,x2,y1,y2 = bounds
    boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    G = ox.graph_from_polygon(boundary_polygon, network_type='drive')

    # Save street network
    save_graph_shapefile_directional(G, filepath=os.path.join(folder, 'porto-network'))

    # Plot first 10 trips
    fig, ax = ox.plot_graph(G, figsize=(30,20), show=False, node_size=0, bgcolor='w', edge_color="k", edge_linewidth=0.5)
    for i in range(0,10):
        lon_lat = np.fromstring(
            train_1000[i,8].replace('[', '').replace(']', '').replace('\\', '').replace('n', ''),
            sep=' ').reshape((-1, 2))
        lon = lon_lat[:,0]
        lat = lon_lat[:,1]
        plt.plot(lon,lat, marker='o', label=f'Trip {i+1}')
    plt.legend(fontsize=25)
    plt.savefig(os.path.join(folder, 'GPS Trajectories - First 10 Trips'))
    plt.show()

