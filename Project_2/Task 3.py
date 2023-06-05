import os

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from fmm import (
    Network, NetworkGraph, FastMapMatch, FastMapMatchConfig, UBODT, UBODTGenAlgorithm,
    STMATCH, STMATCHConfig)

folder = '.'


def convert_to_gpsconfig_input(df):
    """Function to convert input file to format accepted by FMM"""
    input_data = []
    for i in range(1000):
        txt = df[i, 8]
        txt = np.fromstring(
            df[i, 8].replace('[', '').replace(']', '').replace('\\', '').replace('n', ''),
            sep=' ')
        try:
            txt = txt.reshape((-1, 2))
            time_taken = (txt.shape[0] - 1) * (15 / 60) # Time taken in minutes
        except ValueError as e:
            print(i)
            print(str(e))
            txt = txt[0]
            time_taken = 0
        if not isinstance(txt, float):
            txt = ','.join([f"{i[0]} {i[1]}" for i in txt.tolist()])
            txt = f"LINESTRING({txt})"
        input_data.append({
            'id': df[i, 0],
            'geom': txt,
            'time_taken': time_taken
        })
    return pd.DataFrame(input_data)


if __name__ == "__main__":
    # Read the top n rows of csv file as a dataframe
    df = pd.read_csv(os.path.join(folder, "Train_1000_processed.csv")).to_numpy()

    # Convert train_1000.csv to GPSConfig input file
    input_data = convert_to_gpsconfig_input(df)

    # Read in network file saved in Task 2
    network = Network(os.path.join(folder, "porto-network/edges.shp"),"fid","u","v")
    print("Nodes {} edges {}".format(network.get_node_count(), network.get_edge_count()))

    graph = NetworkGraph(network)

    # To run FastMapMatch
    if not os.path.exists(os.path.join(folder, "porto-ubodt.txt")):
        # Precompute an UBODT file
        ubodt_gen = UBODTGenAlgorithm(network,graph)

        status = ubodt_gen.generate_ubodt(
            filename=os.path.join(folder, "porto-ubodt.txt"),
            delta=4,
            binary=False, use_omp=True)

    # Load UBODT data
    ubodt = UBODT.read_ubodt_csv(os.path.join(folder, "porto-ubodt.txt"))

    # Create FMM model
    model = FastMapMatch(network,graph,ubodt)

    # Define FMM configuration
    k = 4  # number of candidates
    radius = 0.4  # search radius
    gps_error = 0.5  # gps error
    fmm_config = FastMapMatchConfig(k,radius,gps_error)

    # Match process
    results = []
    for idx, geom in enumerate(input_data['geom'].values):
        try:
            result = model.match_wkt(geom, fmm_config)
            results.append({
                'idx': idx,
                'match_path': list(result.cpath), # fid in edges file
                'match_geom': result.mgeom.export_wkt(), # lat and lon information,
                'time_taken': input_data.iloc[idx]['time_taken']
            })
        except Exception:
            print(idx, geom)

    results_fmm_df = pd.DataFrame.from_dict(results)
    results_fmm_df.to_csv(os.path.join(folder, 'fmm_mapped_trajectory.csv'), index=False)

    # To run STMatch
    print('STMatch')
    stmodel = STMATCH(network,graph)

    k = 4
    gps_error = 0.5
    radius = 0.4
    vmax = 30
    factor = 1.5
    stmatch_config = STMATCHConfig(k, radius, gps_error, vmax, factor)

    # Match process
    st_results = []
    for idx, geom in enumerate(input_data['geom'].values):
        try:
            result = stmodel.match_wkt(geom, stmatch_config)
            st_results.append({
                'idx': idx,
                'match_path': list(result.cpath), # fid in edges file
                'match_geom': result.mgeom.export_wkt(), # lat and lon information
            })
        except Exception:
            print(idx, geom)

    results_stm_df = pd.DataFrame.from_dict(st_results)
    results_stm_df.to_csv(os.path.join(folder, 'stm_mapped_trajectory.csv'), index=False)
