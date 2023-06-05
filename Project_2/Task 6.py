import os
import numpy as np
import pandas as pd
import osmnx as ox
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from fmm import Network,NetworkGraph,FastMapMatch,FastMapMatchConfig,UBODT

folder = '.'

if __name__ == "__main__":
    
    # Create the street network within the bounding box of Porto's borders
    bounds = (-8.70, -8.57, 41.19, 41.13)
    x1,x2,y1,y2 = bounds
    boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    G = ox.graph_from_polygon(boundary_polygon, network_type='drive')

    # Read original data as df
    df = pd.read_csv(os.path.join(folder, "Train_1000.csv"), nrows=1000)
    
    # Save data as numpy array
    train_1000 = df.to_numpy()
    
    # Change trajectory data from string type to array of coordinates
    for i in range(len(train_1000)):
        data = train_1000[i,8][2:-2]
        data = data.replace(']','')
        data = data.replace('[','')
        data = data.split(',')
        
        # reshape if there is more than 1 set of coordinates
        if len(data) > 1:
            data = np.asarray(data, dtype=float)
            data = data.reshape((len(data)//2,2))
        
        # else put 0 for no trajectory data
        else:
            data = np.asarray([[0.0, 0.0]])
        train_1000[i,8] = data

    # Plot trips trajectory before outlier removal
    fig, ax = ox.plot_graph(G, figsize=(25,25), show=False, node_size=0, bgcolor='w', edge_color="k", edge_linewidth=0.5)
    colors = ['g', 'r', 'c', 'b']
    traj = [2, 3, 59, 64]
    for i in range(4):
        lon = train_1000[traj[i],8][:,0]
        lat = train_1000[traj[i],8][:,1]
        plt.plot(lon,lat, marker='o', color=colors[i], label=f'Trip {traj[i]+1}')
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(folder, 'Trips with Outliers (Before Outlier Removal).png'))
    
    
    # Remove outlying GPS coordinates
    for i in range(1000):
        GPS_trajectory = train_1000[i,8]
        num_points = len(GPS_trajectory)
        route_dist = 0.0
        if num_points > 1:
            # get average distance between points
            for j in range(num_points-1):
                lon_1 = GPS_trajectory[j,0]
                lat_1 = GPS_trajectory[j,1]
                lon_2 = GPS_trajectory[j+1,0]
                lat_2 = GPS_trajectory[j+1,1]
                route_dist += ((lon_1 - lon_2)**2 + (lat_1 - lat_2)**2)**0.5
            ave_dist = route_dist / (num_points-1)
            j = 0
    
            # compare point j with j+1 until all points compared/removed
            while j < (num_points-1):
                lon_1 = GPS_trajectory[j,0]
                lat_1 = GPS_trajectory[j,1]
                lon_2 = GPS_trajectory[j+1,0]
                lat_2 = GPS_trajectory[j+1,1]
                dist = ((lon_1 - lon_2)**2 + (lat_1 - lat_2)**2)**0.5
    
                # if distance between point exceed threshold of 5 times average distance
                if dist > 5*ave_dist:
                    # remove next point
                    GPS_trajectory = np.delete(GPS_trajectory, j+1, 0)
                    # go back to point j
                    j = j-1
                
                # update num_points and j
                num_points = len(GPS_trajectory)
                j = j + 1
        
        train_1000[i,8] = GPS_trajectory
    
    
    # Plot trips trajectory with outliers removed
    fig, ax = ox.plot_graph(G, figsize=(25,25), show=False, node_size=0, bgcolor='w', edge_color="k", edge_linewidth=0.5)
    colors = ['g', 'r', 'c', 'b']
    traj = [2, 3, 59, 64]
    for i in range(4):
        lon = train_1000[traj[i],8][:,0]
        lat = train_1000[traj[i],8][:,1]
        plt.plot(lon,lat, marker='o', color=colors[i], label=f'Trip {traj[i]+1}')
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(folder, 'Trips with Outliers (After Outlier Removal).png'))
    
    
    # Prepare data for FMM algorithm
    input_data2 = []
    for i in range(1000):
        txt = train_1000[i, 8]
        if not isinstance(txt, float):
            txt = ','.join([f"{i[0]} {i[1]}" for i in txt.tolist()])
            txt = f"LINESTRING({txt})"
        input_data2.append({
            'id': train_1000[i, 0],
            'geom': txt
        })
    input_data2 = pd.DataFrame(input_data2)
    input_data2.to_csv(os.path.join(folder, 'trips2.csv'), index=False, sep=';')
    
    
    # Read in network file saved in Task 2
    network = Network(os.path.join(folder, "porto-network/edges.shp"),"fid","u","v")
    print("Nodes {} edges {}".format(network.get_node_count(),network.get_edge_count()))
    
    graph = NetworkGraph(network)
    
    # Load UBODT data
    ubodt = UBODT.read_ubodt_csv(os.path.join(folder, "porto-ubodt.txt"))
    
    # Create and define FMM model
    model = FastMapMatch(network,graph,ubodt)
    k = 8
    radius = 0.01
    gps_error = 0.001
    fmm_config = FastMapMatchConfig(k,radius,gps_error)
    

    # Map Matching using previously set FMM model
    results2 = []
    for idx, geom in enumerate(input_data2['geom'].values):
        # print(f'idx: {idx}, \t geom: {geom}')
        result2 = model.match_wkt(input_data2['geom'].values[idx], fmm_config)
        results2.append({
            'idx': idx,
            'match_path': list(result2.cpath),
            'match_edge_by_pt': list(result2.opath),
            'match_edge_by_idx': list(result2.indices),
            'match_geom': result2.mgeom.export_wkt(),  # lat and lon information
            'match_pt': result2.pgeom.export_wkt(),
            'edge_id': [c.edge_id for c in result2.candidates],
            'source': [c.source for c in result2.candidates],
            'target': [c.target for c in result2.candidates],
            'error': [c.error for c in result2.candidates],
            'length': [c.length for c in result2.candidates],
            'offset': [c.offset for c in result2.candidates],
            'spdist': [c.spdist for c in result2.candidates],
            'ep': [c.ep for c in result2.candidates],
            'tp': [c.tp for c in result2.candidates],
        })
    
    # Plot trips trajectory after map matching
    fig, ax = ox.plot_graph(G, figsize=(25,25), show=False, node_size=0, bgcolor='w', edge_color="k", edge_linewidth=0.5)
    colors = ['g', 'r', 'c', 'b']
    traj = [2, 3, 59, 64]
    for i in range(4):
        mgeom = results2[traj[i]]['match_geom']
        mgeom = mgeom.lstrip('LINESTRING(').rstrip(')')
        mgeom = mgeom.split(',')
        mgeom_lon_lat = [m.split(' ') for m in mgeom]
        # print(mgeom[0])
    
        # Plot trajectories
        lon = [float(l) for l in np.array(mgeom_lon_lat)[:,0] ]
        lat = [float(l) for l in np.array(mgeom_lon_lat)[:,1] ]
        plt.plot(lon, lat, marker='o', color=colors[i], label=f'Trip {traj[i]+1}')
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(folder, 'Trips with Outliers (After Outlier Removal and After FMM).png'))