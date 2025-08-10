import math
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor

import folium
import numpy as np
from flask import Flask, request, send_file, jsonify, render_template_string, render_template
import pandas as pd
import pickle
from glob import glob
import os
import matplotlib.pyplot as plt
from folium import FeatureGroup
from geoletrld.distances import EuclideanDistance
from geoletrld.distances._DistancesUtils import rotate
from geoletrld.utils import Trajectory
from matplotlib import colors
from scipy.optimize import Bounds, shgo
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from tqdm.auto import tqdm
from minisom import MiniSom
from scipy.spatial.distance import cdist

n_jobs = 128
clustering_alg = ["kMeans", "kMedoids", "kMedoids_pre", 'SOM']

app = Flask(__name__)
with app.app_context():
    models = dict()
    for filepath in tqdm(glob("../Data/Resampled/*/*/model.pkl") + glob("../Data/Not_Resampled/*/model.pkl")
            , desc="loading models"):
        with open(filepath, 'rb') as fd:
            models[filepath] = pickle.load(fd)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify(list(models.keys()))

@app.route("/selectors", methods=["GET"])
def list_selection_technique():
    return jsonify(clustering_alg)

def run_clu(clu, dist_matrix, geolets, geolets_keys):
    medoids = dict()
    intra_cluster_distance = np.zeros((len(geolets),))

    labels = clu.fit_predict(dist_matrix)

    for idx, label in enumerate(labels):
        intra_cluster_distance[idx] = np.sum(dist_matrix[idx][labels == label])

    for label in tqdm(np.unique(labels), desc="Collecting medoids"):
        min_idx = np.argmin(intra_cluster_distance[labels == label])
        key = geolets_keys[labels == label][min_idx]
        medoids[f"{key}"] = (geolets[key], list(geolets_keys[labels == label]))

    return medoids

def run_som(dist_matrix, geolets, geolets_keys):
    som = MiniSom(100, 1, dist_matrix.shape[1], sigma=.5, learning_rate=.5, neighborhood_function='gaussian',
                  random_seed=42)

    som.train_batch(dist_matrix, 10000, verbose=True)

    def find_nearest(centroid, dist_matrix):
        distances = cdist(centroid, dist_matrix, metric='cityblock').flatten()
        return np.argmin(distances)
    cluster_labels = np.ravel_multi_index(np.array([som.winner(x) for x in dist_matrix]).T, (100, 1))
    centroids_position = [find_nearest(centroid, dist_matrix) for centroid in tqdm(som.get_weights())]
    centroids_clu_id = cluster_labels[centroids_position]
    print(centroids_clu_id)

    return {geolet_id: (geolets[geolet_id], list(geolets_keys[cluster_labels==cluster_id])) for geolet_id, cluster_id in
            zip(geolets_keys[centroids_position], centroids_clu_id)}

@app.route("/run_selector", methods=["GET"])
def run_selector():
    selected_selector = request.args.get('selected_selector').lower()
    selected_model = request.args.get('selected_model')
    geolets = models[selected_model].candidate_geolets
    geolets_keys = np.array(list(models[selected_model].candidate_geolets.keys()))
    dist_matrix = models[selected_model].selector.dist_matrix

    print(selected_selector, selected_model)

    medoids = dict()
    if selected_selector == "kmeans":
        medoids = run_clu(KMeans(n_clusters=100), dist_matrix, geolets, geolets_keys)
    elif selected_selector == "kmedoids":
        medoids = run_clu(KMedoids(n_clusters=100), dist_matrix, geolets, geolets_keys)
    elif selected_selector == "kmedoids_pre":
        medoids = run_clu(KMedoids(n_clusters=100, metric="precomputed"), dist_matrix, geolets, geolets_keys)
    elif selected_selector == "som":
        medoids = run_som(dist_matrix, geolets, geolets_keys)

    if os.path.exists("static/tmp"):
        shutil.rmtree("static/tmp")

    paths = []
    os.mkdir("static/tmp")
    for geolet_id, (geolet_data, cluster_records) in medoids.items():
        plt.plot(geolet_data.longitude, geolet_data.latitude, color='blue')
        plt.scatter(geolet_data.longitude[0], geolet_data.latitude[0], color='red')
        #plt.title(geolet_id, fontsize=24)
        plt.gca().set_aspect('equal')
        plt.gca().axis('off')
        plt.tight_layout()
        path = f"static/tmp/{geolet_id}.png"
        plt.savefig(path)
        paths.append((path, cluster_records))
        plt.close()

    return jsonify(paths)


@app.route("/find_similar", methods=["GET"])
def find_similar():
    selected_selector = request.args.get('selected_selector').lower()
    selected_model = request.args.get('selected_model')
    geolet_name = request.args.get('geolet')
    clu_elements = request.args.getlist('clu_elements[]')
    geolets_keys = np.array(list(models[selected_model].candidate_geolets.keys()))

    clu_elements_positions = [i for i, k in enumerate(models[selected_model].selector.geolets_keys) if k in clu_elements]

    dist_matrix = models[selected_model].selector.dist_matrix[np.ix_(clu_elements_positions, clu_elements_positions)]
    geolets = {k: v for k, v in models[selected_model].candidate_geolets.items() if k in clu_elements}
    geolets_keys = geolets_keys[clu_elements_positions]

    print(geolet_name, len(clu_elements), dist_matrix.shape)
    n_clusters = min(len(dist_matrix), 100)

    medoids = dict()
    if selected_selector == "kmeans":
        medoids = run_clu(KMeans(n_clusters=n_clusters), dist_matrix, geolets, geolets_keys)
    elif selected_selector == "kmedoids":
        medoids = run_clu(KMedoids(n_clusters=n_clusters), dist_matrix, geolets, geolets_keys)
    elif selected_selector == "kmedoids_pre":
        medoids = run_clu(KMedoids(n_clusters=n_clusters, metric="precomputed"), dist_matrix, geolets, geolets_keys)
    elif selected_selector == "som":
        medoids = run_som(dist_matrix, geolets, geolets_keys)

    if os.path.exists("static/tmp"):
        shutil.rmtree("static/tmp")

    paths = []
    os.mkdir("static/tmp")
    for geolet_id, (geolet_data, cluster_records) in medoids.items():
        plt.plot(geolet_data.longitude, geolet_data.latitude, color='blue')
        plt.scatter(geolet_data.longitude[0], geolet_data.latitude[0], color='red')
        #plt.title(geolet_id, fontsize=24)
        plt.gca().set_aspect('equal')
        plt.gca().axis('off')
        plt.tight_layout()
        path = f"static/tmp/{geolet_id}.png"
        plt.savefig(path)
        paths.append((path, cluster_records))
        plt.close()

    return jsonify(paths)

@app.route("/geolets", methods=["GET"])
def list_geolets():
    selected_model = request.args.get('selected_model')
    return jsonify(list(models[selected_model].candidate_geolets.keys()))

def read_trj(selected_model):
    if "Not_Resampled" in selected_model:
        trj_path = "/".join(selected_model.split("/")[:-2]) + "/trj.trj"
    else:
        trj_path = ("/".join(selected_model.split("/")[:-1])
                    .replace("_rot_euc", "")
                    .replace("_euc", "")
                    + ".trj")
    with open(trj_path, "rb") as f:
        trj = pickle.load(f)
    return trj
@app.route("/trj", methods=["GET"])
def list_trj():
    selected_model = request.args.get('selected_model')
    return jsonify([str(el) for el in read_trj(selected_model).keys()])

@app.route("/generate_map", methods=["GET"])
def generate_map():
    selected_model = request.args.get('selected_model')
    trj = read_trj(selected_model)

    geolet_sub = models[selected_model].candidate_geolets[request.args.get('selected_geolet')]
    trajectory = trj[int(request.args.get('selected_trj'))]

    d = my_euc
    if "rot_euc" in request.args.get('selected_model'):
        d = my_rot_eu

    dist_to_trj = np.hstack([[.0], d(trajectory=trajectory, geolet=geolet_sub, agg=np.mean)])
    dist_to_trj_norm = MinMaxScaler().fit_transform(np.hstack([[.0], dist_to_trj]).reshape(-1, 1)**(1/3)).flatten()[1:]
    m = folium.Map(location=[geolet_sub.latitude[0], geolet_sub.longitude[0]], zoom_start=10)
    color_map = plt.cm.RdYlGn(dist_to_trj_norm)

    len_geo = len(geolet_sub.latitude)
    len_trajectory = len(trajectory.latitude)

    #near_segments_layer = FeatureGroup(name="Near segments")
    #far_segments_layer = FeatureGroup(name="Far segments")
    geolet_layer = FeatureGroup(name="Geolet")

    #near_segments_layer.add_to(m)
    #far_segments_layer.add_to(m)
    layers = [FeatureGroup(name=f'{i / 10}-{(i+1)/10}') for i in range(10)]
    for layer in layers:
        layer.add_to(m)
    geolet_layer.add_to(m)

    folium.LayerControl().add_to(m)

    for i in tqdm(range(0, len_trajectory - len_geo + 1, len_geo)):
        segment = list(zip(trajectory.latitude[i:i + len_geo], trajectory.longitude[i:i + len_geo +1]))
        dist = np.min(dist_to_trj_norm[i:i + len_geo])
        dist_mean = np.mean(dist_to_trj_norm[i:i + len_geo])
        dist_argmin = i + np.argmin(dist_to_trj_norm[i:i + len_geo])
        dist_str = f'min: {round(dist, 4)}<br>mean: {round(dist_mean, 4)}<br>std: {round(np.std(dist_to_trj_norm[i:i + len_geo]), 4)}'

        poly = folium.PolyLine(segment, color=colors.to_hex(color_map[dist_argmin]), weight=2, opacity=0.2,
                            popup=dist_str)

        d_interval = int(dist*10)
        poly.add_to(layers[d_interval])

        #if dist < .3:
        #    poly.add_to(near_segments_layer)
        #else:
        #    poly.add_to(far_segments_layer)

    folium.PolyLine(list(zip(geolet_sub.latitude, geolet_sub.longitude)), color='cyan', weight=3).add_to(geolet_layer)

    m.save('static/tmp/map.html')

    return jsonify(['static/tmp/map.html'])


#=================OTHER FUNCTIONS
def my_euc(trajectory, geolet, agg=np.sum) -> tuple:
    len_geo = len(geolet.latitude)
    len_trajectory = len(trajectory.latitude)

    if len_geo > len_trajectory:
        return np.array([1])

    res = np.zeros(len_trajectory - len_geo + 1)
    geolet_normalized, _ = Trajectory._first_point_normalize(geolet.lat_lon)

    for i in range(len_trajectory - len_geo + 1):
        trj_normalized, _ = Trajectory._first_point_normalize(trajectory.lat_lon[:, i:i + len_geo])
        res[i] = agg(((trj_normalized - geolet_normalized) ** 2)) ** .5

    return res


def objective_function(angle, trajectory: Trajectory, geolet: Trajectory, distance):
    try:
        rotated_geolet = rotate(geolet.copy(), angle)

        return distance(trajectory=trajectory, geolet=rotated_geolet)[0]
    except:
        traceback.print_exception()


def my_rot_eu(trajectory, geolet, agg=np.sum) -> tuple:
    len_geo = len(geolet.latitude)
    len_trajectory = len(trajectory.latitude)

    if len_geo > len_trajectory:
        return np.array([1])

    res = np.zeros(len_trajectory - len_geo + 1)
    geolet_normalized = geolet.copy().normalize()

    processes = []
    bounds = Bounds([0], [2 * math.pi], )
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for i in tqdm(range(len_trajectory - len_geo + 1), desc="submitting"):
            sub_trajectory = Trajectory(latitude=trajectory.latitude[i:i + len_geo],
                                        longitude=trajectory.longitude[i:i + len_geo],
                                        time=trajectory.time[i:i + len_geo])

            processes.append(executor.submit(shgo, objective_function, sampling_method="sobol",
                                             args=(sub_trajectory, geolet_normalized, EuclideanDistance.best_fitting),
                                             bounds=bounds))

        for i, process in enumerate(tqdm(processes, desc="retrieving results")):
            sub_trajectory = Trajectory(latitude=trajectory.latitude[i:i + len_geo],
                                        longitude=trajectory.longitude[i:i + len_geo],
                                        time=trajectory.time[i:i + len_geo])
            result = process.result()
            angle = result.x
            dist, idx = EuclideanDistance.best_fitting(trajectory=sub_trajectory,
                                                       geolet=rotate(geolet_normalized, angle),
                                                       agg=agg)
            res[i] = dist

    return res







if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
