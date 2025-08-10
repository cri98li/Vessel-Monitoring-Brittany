import pickle

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

def main():
    name = "resample_travel_distance_100m_10km"

    with open(f'{name}/{name}.trj', 'rb') as fd:
        trajectories = pickle.load(fd)
    n_jobs = 120 * 2

    #dist = lambda **args: RotatingGenericDistance(EuclideanDistance(), **args)

    dist = EuclideanDistance
    dist_name = "euc"

    model = Geolet(
        #partitioner=GeohashPartitioner(precision=5, verbose=True),
        partitioner=FeaturePartitioner(feature="distance", threshold=10*1000),
        # selector=GapSelector(k=10, distance=dist(), n_jobs=n_jobs, verbose=True),
        selector=ClusteringSelector(clustering_fun=None,
                                    distance=dist(shape_error='invert'),
                                    n_jobs=n_jobs,
                                    verbose=True),
        distance=dist(n_jobs=n_jobs, verbose=True),
        model_to_fit=None,
        subset_candidate_geolet=10**9,#10000,
        subset_trj_in_selection=10,
        verbose=True,
    )

    model.fit(trajectories, np.zeros((len(trajectories),)))

    test_namev=f"{name}_{dist_name}"
    #test_namev = "eucl_dist_sliding_win_5km_ep"
    if not os.path.exists(f'{name}/{test_namev}/'):
        os.mkdir(f'{name}/{test_namev}/')

    for geolet_id, geolet_data in tqdm(model.selected_geolets.items(), desc="Saving results"):
        m = folium.Map(location=[geolet_data.latitude[0], geolet_data.longitude[0]], zoom_start=12)
        folium.PolyLine(list(zip(geolet_data.latitude, geolet_data.longitude)), color='blue').add_to(m)
        folium.Marker([geolet_data.latitude[0], geolet_data.longitude[0]], popup=geolet_id).add_to(m)
        m.save(f"{name}/{test_namev}/{geolet_id}_FIXED.html")


    pickle.dump(model, open(f"{name}/{test_namev}/model_FIXED.pkl", "wb"))


if __name__ == "__main__":
    main()