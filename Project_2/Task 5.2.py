import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import json
import geopandas as gpd
import osmnx as ox
from matplotlib import pyplot as plt
from shapely.geometry import Polygon

folder = '.'

def map_fid_to_avg_travel_time(results_fmm_df):
    """Assume travel speed per trip is same throughout the trip."""
    cols = ['trip_id', 'name', 'length', 'u', 'v', 'osmid', 'time_taken']
    road_segment_time = []

    for idx in results_fmm_df['idx'].values:
        time_taken = results_fmm_df.loc[results_fmm_df['idx'] == idx, 'time_taken'].values[0]

        traject_path = results_fmm_df.loc[results_fmm_df['idx'] == idx, 'match_path'].values[0]
        traject_path = json.loads(traject_path)
        traject_path_df = pd.DataFrame(data={'fid': traject_path})
        traject_path_df = traject_path_df.merge(edges_shapefile, on='fid', how='left')

        traject_path_df['total_length'] = traject_path_df['length'].sum()
        traject_path_df['time_taken'] = time_taken * (traject_path_df['length'] / traject_path_df['total_length'])
        traject_path_df['trip_id'] = idx

        road_segment_time.append(traject_path_df[cols])

    road_segment_time = pd.concat(road_segment_time, ignore_index=True)

    return road_segment_time


if __name__ == "__main__":

    # create the street network within the bounding box of Porto's borders
    bounds = (-8.70, -8.57, 41.19, 41.13)
    x1,x2,y1,y2 = bounds
    boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    G = ox.graph_from_polygon(boundary_polygon, network_type='drive')

    # Read trajectory file
    results_fmm_df = pd.read_csv(os.path.join(folder, 'fmm_mapped_trajectory.csv'))

    # Read network file
    edges_shapefile = gpd.read_file(os.path.join(folder, "porto-network/edges.shp"))
    edges_shapefile = edges_shapefile[['name', 'u', 'v', 'length', 'osmid', 'fid', 'geometry']]

    # Map each fid to an average time for each trip
    road_segment_time = map_fid_to_avg_travel_time(results_fmm_df)

    # Find average travel time for each fid across all trips
    avg_road_segment_time = (road_segment_time
                             .groupby(['name', 'length', 'osmid', 'u', 'v'], as_index=False)
                             .agg(avg_time=('time_taken', 'mean')))

    # Find average time per unit length of fid
    avg_road_segment_time['avg_time_per_unit_length'] = (
        avg_road_segment_time['avg_time'] / avg_road_segment_time['length'])

    # For each osmid, find average of average time per unit length, and total length of road segment
    total_avg_road_segment_time = (avg_road_segment_time
                                   .groupby(['name', 'osmid'], as_index=False)
                                   .agg({
                                        'avg_time_per_unit_length': 'mean',
                                        'length': 'sum'
                                    }))

    # Total average time for road segment is average time per unit length multiplied by total length
    total_avg_road_segment_time['avg_time'] = (
        total_avg_road_segment_time['avg_time_per_unit_length'] *
        total_avg_road_segment_time['length'])

    # Get top 5 road segments by average time
    top_5_segments = total_avg_road_segment_time.sort_values(by='avg_time', ascending=False).head()
    top_5_segments = [{'osmid': i, 'time_taken': j} for i, j in
                      zip(top_5_segments['osmid'],  top_5_segments['avg_time'])]

    # Map back to original file
    top_5_segments_df = pd.DataFrame(data=top_5_segments)
    top_5_geoms_df = top_5_segments_df.merge(edges_shapefile, on=['osmid'], how='left')

    # Plot trajectories
    fig, ax = ox.plot_graph(G, figsize=(30,20),
                            show=False, node_size=0, bgcolor='w', edge_color="k", edge_linewidth=0.5)

    # Original GPS
    for osmid in top_5_geoms_df['osmid'].unique():
        name = top_5_geoms_df.loc[top_5_geoms_df['osmid'] == osmid, 'name'].unique()[0]
        time_taken = top_5_geoms_df.loc[top_5_geoms_df['osmid'] == osmid, 'time_taken'].unique()[0]

        geom_lon_lat = []
        for edge in top_5_geoms_df[top_5_geoms_df['osmid'] == osmid].itertuples():
            geom = str(edge.geometry)
            geom = geom.lstrip('LINESTRING (').rstrip(')')
            geom = geom.split(', ')
            geom_lon_lat += [m.split(' ') for m in geom]
        # geom_lon_lat = sorted(geom_lon_lat)

        lon = [float(l) for l in np.array(geom_lon_lat)[:,0] ]
        lat = [float(l) for l in np.array(geom_lon_lat)[:,1] ]
        plt.plot(lon, lat, marker='o', label=f'{name} (Time Taken: {time_taken:.2f}mins)', linewidth=3)

    plt.legend(fontsize=25)
    plt.savefig(os.path.join(folder, 'Top 5 Road Segments with Longest Average Travelling Time'))
    plt.show()

