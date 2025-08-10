# Useful link: https://blog.dailydoseofds.com/p/transform-decision-tree-into-matrix

import pickle


import numpy as np
import os
import folium
from geoletrld.distances import (EuclideanDistance, RotatingGenericDistance,
                                 InterpolatedTimeDistance, LCSSTrajectoryDistance)
from geoletrld.model import Geolet
from geoletrld.partitioners import GeohashPartitioner, FeaturePartitioner
from geoletrld.selectors import ClusteringSelector, GapSelector
from tqdm.auto import tqdm

def main():
    with open(f'trj.trj', 'rb') as fd:
        trajectories = pickle.load(fd)
    n_jobs = 120 * 4

    dist = lambda **args: RotatingGenericDistance(EuclideanDistance(agg=np.sum, shape_error='invert'), **args)
    #dist = lambda **args: RotatingGenericDistance(InterpolatedTimeDistance(agg=np.mean), **args)
    #dist = lambda **args: RotatingGenericDistance(LCSSTrajectoryDistance(n_jobs=2), **args)

    #dist = EuclideanDistance()
    #dist = InterpolatedTimeDistance
    #dist = LCSSTrajectoryDistance
    dist_name = "sum_rot_euc"

    model = Geolet(
        #partitioner=GeohashPartitioner(precision=4, verbose=True),
        #partitioner=FeaturePartitioner(feature="time", threshold=8*(60*60), overlapping=.25),
        partitioner=FeaturePartitioner(feature="distance", threshold=10 * 1000, overlapping=.1),
        #selector=GapSelector(k=1000, distance=dist(), n_jobs=n_jobs, verbose=True, startegy='geolet'),
        selector=ClusteringSelector(clustering_fun=None, distance=dist(), n_jobs=n_jobs, verbose=True),
        distance=dist(n_jobs=n_jobs, verbose=True),
        model_to_fit=None,
        subset_candidate_geolet=10**9,#10000,
        subset_trj_in_selection=10,
        verbose=True,
    )

    model.fit(trajectories, np.zeros((len(trajectories),)))

    name= "FIXED_not_resample_10km"
    test_namev=f"{name}_{dist_name}"
    #test_namev = "eucl_dist_sliding_win_5km_ep"
    if not os.path.exists(f'{test_namev}/'):
        os.mkdir(f'{test_namev}/')


    pickle.dump(model, open(f"{test_namev}/model.pkl", "wb"))
    print("fatto")


if __name__ == "__main__":
    main()