import os
import pandas as pd
import numpy as np

folder = '.'


def process_trajectory_data(data_array):
    """Function to change trajectory data from string type to array of coordinates."""

    # change trajectory data from string type to array of coordinates
    for i in range(len(data_array)):
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
            data = 0.0
        data_array[i,8] = data
    
    return data_array


if __name__ == "__main__":
    # Read the top n rows of csv file as a dataframe
    df = pd.read_csv(os.path.join(folder, "Train_1000.csv"), nrows=1000)

    # Print the shape of the dataframe
    print("Dataframe shape:", df.shape)

    # Convert to numpy array
    train_1000 = df.to_numpy()

    # Process trajectory data
    train_1000 = process_trajectory_data(train_1000)

    # Save trajectory data to csv
    pd.DataFrame(train_1000).to_csv("Train_1000_processed.csv", index=False)
