import pickle
from concurrent.futures.process import ProcessPoolExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import folium
from geoletrld.distances import EuclideanDistance, RotatingGenericDistance, InterpolatedTimeDistance
from geoletrld.model import Geolet
from geoletrld.partitioners import GeohashPartitioner, FeaturePartitioner
from geoletrld.selectors import ClusteringSelector
from geoletrld.utils import Trajectories
from sklearn.cluster import KMeans

from tqdm import tqdm_notebook
from glob import glob
from tqdm.auto import tqdm

from seolet.resampling import travel_distance_resample, absolute_distance_resample, time_resample

if __name__ == "__main__":
    df = pd.read_csv('Data/nari-trawling-71vessels_cleaned.csv')
    df['timestamp'] = df.time.apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M:%S").timestamp())
    df.sort_values(["trID", "timestamp"], inplace=True)

    trajectories = Trajectories.from_DataFrame(df, tid="trID", latitude="Y", longitude="X", time="timestamp")

    #pickle.dump(trajectories, open("Data/Not_Resampled/trj.trj", "wb"))

    processes_travel_distance = dict()
    processes_abs_distance = dict()
    processes_time_resample = dict()
    with ProcessPoolExecutor(max_workers=128) as executor:
        for tr_id, tr_data in tqdm(trajectories.items(), desc="submitting process"):
            processes_travel_distance[tr_id] = executor.submit(travel_distance_resample, tr_data, 100)
            processes_abs_distance[tr_id] = executor.submit(absolute_distance_resample, tr_data, 100)
            processes_time_resample[tr_id] = executor.submit(time_resample, tr_data, 5*50)

    trajectories_travel_distance = Trajectories()
    trajectories_abs_distance = Trajectories()
    trajectories_time_resample = Trajectories()

    for tr_id, tr_data_process in tqdm(processes_travel_distance.items(), desc="Collecting results"):
        trajectories_travel_distance[tr_id] = tr_data_process.result()

    for tr_id, tr_data_process in tqdm(processes_abs_distance.items(), desc="Collecting results"):
        trajectories_abs_distance[tr_id] = tr_data_process.result()

    for tr_id, tr_data_process in tqdm(processes_time_resample.items(), desc="Collecting results"):
        trajectories_time_resample[tr_id] = tr_data_process.result()

    #pickle.dump(trajectories_travel_distance, open("Data/Resampled/trajectories_travel_distance_100.trj", "wb"))
    #pickle.dump(trajectories_abs_distance, open("Data/Resampled/trajectories_abs_distance_100.trj", "wb"))
    pickle.dump(trajectories_time_resample, open(
        "Data/Resampled/trajectories_time_resample_100/trajectories_time_resample_5mins.trj", "wb"))
