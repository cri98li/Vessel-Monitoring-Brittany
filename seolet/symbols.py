import numpy as np
from geoletrld.utils import Trajectory
from geolib import geohash
import CaGeo.algorithms.BasicFeatures as bf

from seolet.MySAX import MySAX

def _move_point(lat, lon, distance, bearing):
    R = 6371000  # Earth radius in meters
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing)

    # Calculate the new latitude
    new_lat = np.arcsin(np.sin(lat_rad) * np.cos(distance / R) +
                        np.cos(lat_rad) * np.sin(distance / R) * np.cos(bearing_rad))
    
    # Calculate the new longitude
    new_lon = lon_rad + np.arctan2(np.sin(bearing_rad) * np.sin(distance / R) * np.cos(lat_rad),
                                   np.cos(distance / R) - np.sin(lat_rad) * np.sin(new_lat))
    
    # Convert back to degrees
    new_lat = np.degrees(new_lat)
    new_lon = np.degrees(new_lon)
    
    return new_lat, new_lon

def absolute_position(trajectory: Trajectory, precision):
    symbols = []
    for i in range(len(trajectory.latitude)):
        symbols.append(geohash.encode(lat=trajectory.latitude[i], lon=trajectory.longitude[i], precision=precision))

    return np.array(symbols)


def inverse_absolute_position(symbols):
    latitudes = []
    longitudes = []
    for symbol in symbols:
        lat, lon = geohash.decode(symbol)
        latitudes.append(lat)
        longitudes.append(lon)

    return Trajectory(latitude=np.array(latitudes), longitude=np.array(longitudes))


def semi_absolute_position(trajectory: Trajectory, precision, suffix_len):
    symbols = []
    for i in range(len(trajectory.latitude)):
        enc = geohash.encode(lat=trajectory.latitude[i], lon=trajectory.longitude[i], precision=precision)
        symbols.append(enc[precision-suffix_len:])

    return np.array(symbols)

def inverse_semi_absolute_position(suffixes, precision, suffix_len):
    prefix = "".join(['a' for _ in range(precision-suffix_len)])

    latitudes = []
    longitudes = []
    for suffix in suffixes:
        full_geohash = prefix + suffix
        lat, lon = geohash.decode(full_geohash)
        latitudes.append(lat)
        longitudes.append(lon)

    return np.array(latitudes), np.array(longitudes)

def relative_direction_position(trajectory: Trajectory, min_dist, n_symbols=8 + 1):
    symbols=None
    if n_symbols-1 == 4:
        symbols = [x for x in "ENWS"]
    elif n_symbols-1 == 8:
        symbols = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    elif n_symbols-1 == 16:
        symbols = ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW", "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE"]
    sax = MySAX(n_bins=n_symbols-1, compress=1, symbols=symbols).fit()

    dist = bf.distance(lat=trajectory.latitude, lon=trajectory.longitude, accurate=False)*1000
    direction = bf.direction(lat=trajectory.latitude, lon=trajectory.longitude)

    symbols = []
    d_sum = 0
    for d, dir in zip(dist, direction):
        if d_sum + d < min_dist:
            symbols.append("§")
            d_sum += d
        else:
            symbols.append(sax.transform([dir])[0][0])
            d_sum = 0

    return np.array(symbols)

def inverse_relative_direction_position(symbols, start_lat, start_lon, min_dist, n_symbols=8 + 1):
    if n_symbols-1 not in [4, 8, 16]:
        raise ValueError("Current implementation only supports 4, 8 and 16 symbols.")

    directions_map = {
        "E": 90, "NE": 45, "N": 0, "NW": 315, "W": 270, "SW": 225, "S": 180, "SE": 135,
        "ENE": 67.5, "NNE": 22.5, "NNW": 337.5, "WNW": 292.5, "WSW": 247.5,
        "SSW": 202.5, "SSE": 157.5, "ESE": 112.5
    }

    latitudes = [start_lat]
    longitudes = [start_lon]
    current_lat, current_lon = start_lat, start_lon
    d_sum = 0

    for sym in symbols:
        if sym == "§":
            d_sum += min_dist
        else:
            bearing = directions_map[sym]
            # Move from current point by d_sum + min_dist in the bearing direction
            dist = d_sum + min_dist
            new_lat, new_lon = bf.move_point(current_lat, current_lon, dist, bearing)
            latitudes.append(new_lat)
            longitudes.append(new_lon)
            current_lat, current_lon = new_lat, new_lon
            d_sum = 0

    return Trajectory(latitude=np.array(latitudes), longitude=np.array(longitudes))


def relative_turning_position(trajectory: Trajectory, min_dist, n_symbols=8 + 1):
    symbols=None
    if n_symbols-1 == 4:
        symbols = [x for x in "ENWS"]
    elif n_symbols-1 == 8:
        symbols = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    elif n_symbols-1 == 16:
        symbols = ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW", "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE"]
    sax = MySAX(n_bins=n_symbols-1, compress=1, symbols=symbols).fit()

    dist = bf.distance(lat=trajectory.latitude, lon=trajectory.longitude, accurate=False)*1000
    turning = bf.turningAngles(lat=trajectory.latitude, lon=trajectory.longitude)

    for i in range(len(turning)):
        if turning[i] < -np.pi:
            turning[i] += np.pi
        elif turning[i] > np.pi:
            turning[i] -= np.pi

    symbols = []
    d_sum = 0
    for d, dir in zip(dist, turning):
        if d_sum + d < min_dist:
            symbols.append("§")
            d_sum += d
        else:
            symbols.append(sax.transform([dir])[0][0])
            d_sum = 0

    return np.array(symbols)

def inverse_relative_turning_position(symbols, start_lat, start_lon, min_dist, n_symbols=8 + 1):
    directions_map = {
        "E": 90, "NE": 45, "N": 0, "NW": 315, "W": 270, "SW": 225, "S": 180, "SE": 135,
        "ENE": 67.5, "NNE": 22.5, "NNW": 337.5, "WNW": 292.5, "WSW": 247.5,
        "SSW": 202.5, "SSE": 157.5, "ESE": 112.5
    }

    latitudes = [start_lat]
    longitudes = [start_lon]
    current_lat, current_lon = start_lat, start_lon
    current_bearing = 0  # Starting with no initial turning

    d_sum = 0
    for sym in symbols:
        if sym == "§":
            d_sum += min_dist
        else:
            bearing = directions_map[sym]
            # Calculate new bearing change from the current direction
            turning_angle = bearing - current_bearing
            # Normalize turning angle within -180 to 180 degrees
            if turning_angle < -180:
                turning_angle += 360
            elif turning_angle > 180:
                turning_angle -= 360
            current_bearing = bearing

            # Move from current point by d_sum + min_dist in the current bearing direction
            dist = d_sum + min_dist
            new_lat, new_lon = _move_point(current_lat, current_lon, dist, current_bearing)
            latitudes.append(new_lat)
            longitudes.append(new_lon)
            current_lat, current_lon = new_lat, new_lon
            d_sum = 0

    return Trajectory(latitude=np.array(latitudes), longitude=np.array(longitudes))
