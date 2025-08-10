import numpy as np
import pandas as pd
from geoletrld.utils import Trajectory
from scipy.interpolate import interp1d
from shapely import LineString

import CaGeo.algorithms.BasicFeatures as bf
from tqdm.auto import tqdm
import geopandas as geopd

from seolet.utils import haversine


def travel_distance_resample(trajectory: Trajectory, distance_in_meters=1):
    distances = bf.distance(trajectory.latitude, trajectory.longitude, accurate=True).cumsum() * 1000
    total_dist = distances[-1]
    new_distances = np.arange(0, total_dist, distance_in_meters)

    resampled_trace = []
    for dim in range(trajectory.values.shape[0]):
        interp_func = interp1d(distances, trajectory.values[dim], kind='linear')
        resampled_trace.append(interp_func(new_distances))

    resampled_trace = np.array(resampled_trace)

    return Trajectory(values=resampled_trace)


def absolute_distance_resample(trajectory: Trajectory, distance_in_meters=10):
    new_latitude_list = [trajectory.latitude[0]]
    new_longitude_list = [trajectory.longitude[0]]
    latitude = trajectory.latitude
    longitude = trajectory.longitude
    for i in range(1, len(trajectory.longitude) - 1):
        dist = bf.distance(
            np.array([new_latitude_list[-1], latitude[i]]).T,
            np.array([new_longitude_list[-1], longitude[i]]),
            accurate=True
        )[-1] * 1000
        if dist > distance_in_meters:
            ls = LineString([
                [longitude[i - 1], latitude[i - 1]],
                [longitude[i], latitude[i]]
            ])

            dist_m_1 = 0
            lookback = 1
            while dist_m_1 == 0:
                dist_m_1 = haversine(longitude[i - lookback], latitude[i - lookback], longitude[i], latitude[i])
                lookback += 1
            ls.interpolate([(dist_m_1 - (dist - distance_in_meters)) / dist_m_1], normalized=True)

            new_longitude_list.append(longitude[i])
            new_latitude_list.append(latitude[i])

    new_time = np.linspace(trajectory.time.min(), trajectory.time.max(), len(new_latitude_list))
    return Trajectory(latitude=np.array(new_latitude_list), longitude=np.array(new_longitude_list), time=new_time)


def time_resample(trajectory: Trajectory, time_in_seconds=10):
    n_sample = int((trajectory.time.max() - trajectory.time.min()) / time_in_seconds)
    new_time = np.linspace(trajectory.time.min(), trajectory.time.max(), n_sample)
    new_lat = np.interp(new_time, trajectory.time, trajectory.latitude)
    new_lon = np.interp(new_time, trajectory.time, trajectory.longitude)

    return Trajectory(latitude=new_lat, longitude=new_lon, time=new_time)
