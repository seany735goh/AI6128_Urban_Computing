import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon

folder = '.'


def map_trajectory_fid_to_osmid_and_get_count(results, edges):
    """Function to map fid to osmid and get count of each osmid in all trajectories."""
    road_segment_counts = {}
    for idx in results['idx'].values:
        traject_path = results.loc[results['idx'] == idx, 'match_path'].values[0]
        traject_path = json.loads(traject_path)
        for fid in traject_path:
            edges_info = edges[edges['fid'] == fid].to_dict(orient='records')[0]
            osmid = edges_info['osmid']
            if osmid not in road_segment_counts.keys():
                road_segment_counts[osmid] = 1
            else:
                road_segment_counts[osmid] += 1
    return road_segment_counts


if __name__ == "__main__":

    # Create the street network within the bounding box of Porto's borders
    bounds = (-8.70, -8.57, 41.19, 41.13)
    x1,x2,y1,y2 = bounds
    boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    G = ox.graph_from_polygon(boundary_polygon, network_type='drive')

    # Read trajectory file
    results_fmm_df = pd.read_csv(os.path.join(folder, 'fmm_mapped_trajectory.csv'))

    # Read network file
    edges_shapefile = gpd.read_file(os.path.join(folder, "porto-network/edges.shp"))
    edges_shapefile = edges_shapefile[['name', 'u', 'v', 'length', 'osmid', 'fid', 'geometry']]

    # Map trajectory fid to edges file
    road_segment_counts = map_trajectory_fid_to_osmid_and_get_count(
        results_fmm_df, edges_shapefile)

    # Get top 5 segments
    top_5_segments = sorted(road_segment_counts, key=road_segment_counts.get, reverse=True)[:5]
    top_5_segments_df = pd.DataFrame(data={
        'osmid': top_5_segments
    })

    # Get route information of top 5 segments
    top_5_geoms_df = top_5_segments_df.merge(edges_shapefile, on=['osmid'], how='left')
    top_5_geoms_df

    # Plot trajectories
    fig, ax = ox.plot_graph(G, figsize=(30,20),
                            show=False, node_size=0, bgcolor='w', edge_color="k", edge_linewidth=0.5)
    for osmid in top_5_geoms_df['osmid'].unique():
        name = top_5_geoms_df.loc[top_5_geoms_df['osmid'] == osmid, 'name'].unique()[0]
        count = road_segment_counts[osmid]

        geom_lon_lat = []
        for edge in top_5_geoms_df[top_5_geoms_df['osmid'] == osmid].itertuples():
            geom = str(edge.geometry)
            geom = geom.lstrip('LINESTRING (').rstrip(')')
            geom = geom.split(', ')
            geom_lon_lat += [m.split(' ') for m in geom]
        geom_lon_lat = sorted(geom_lon_lat)

        lon = [float(l) for l in np.array(geom_lon_lat)[:,0] ]
        lat = [float(l) for l in np.array(geom_lon_lat)[:,1] ]
        plt.plot(lon, lat, linestyle='-', label=f'{name} (Trip Counts: {count})', linewidth=3)

    plt.legend(fontsize=25)
    plt.savefig(os.path.join(folder, 'Top 5 Most Traversed Road Segments'))
    plt.show()
