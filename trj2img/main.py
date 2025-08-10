from concurrent.futures.process import ProcessPoolExecutor
from datetime import datetime

import cv2
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import psutil
from tqdm.auto import tqdm
import scipy
from PIL import Image, ImageDraw
from sklearn.preprocessing import OrdinalEncoder
from skimage.transform import hough_line
from matplotlib import cm
import pandas as pd


def draw_image(lines, resolution="auto", padding=10, verbose=False):
    if type(lines) != np.ndarray:
        lines = np.array(lines)

    min_x = lines[:, ::2].min()
    max_x = lines[:, ::2].max()
    min_y = lines[:, 1::2].min()
    max_y = lines[:, 1::2].max()

    if resolution == "auto":
        resolution = (max_x - min_x + padding * 2, max_y - min_y + padding * 2)

    if verbose:
        print(min_x, max_x, min_y, max_y)
        print(resolution)

    image = Image.new("1", resolution, "black")

    draw = ImageDraw.Draw(image)

    scale_x = (resolution[0] - 2 * padding) / max((max_x - min_x), 1)
    scale_y = (resolution[1] - 2 * padding) / max((max_x - min_x), 1)

    for line in lines:
        line = (
            int((line[0] - min_x) * scale_x + padding),
            int((line[1] - min_y) * scale_y + padding),
            int((line[2] - min_x) * scale_x + padding),
            int((line[3] - min_y) * scale_y + padding)
        )
        if verbose:
            print(line)
        draw.line(line, fill="white", width=1)

    return image

def time_based_sliding_window(df:pd.DataFrame, threshold:float):
    diff = df.timestamp.diff().values
    c1 = df.X.values
    c2 = df.Y.values

    start_i = 0
    end_i = 1
    current =[]
    while True:
        if len(df) == end_i:
            if len(current) != 0:
                yield current
            break

        if diff[start_i+1:end_i].sum() >= threshold:
            yield current
            start_i += 1
            current = current[1:]
        else:
            current += [(c1[end_i-1], c2[end_i-1], c1[end_i], c2[end_i])]
            end_i += 1

def inner_loop(lines, tested_angles):
    img = draw_image(lines, resolution=(1920 * 2, 1920 * 2), verbose=False)
    h, theta, d = hough_line(np.array(img), theta=tested_angles)
    h_sum = np.log(1 + h).sum(axis=0)
    fft = np.abs(scipy.fft.fft(h_sum))
    return fft

def main():
    df = pd.read_csv('../Data/nari-trawling-71vessels_cleaned.csv')
    df['timestamp'] = df.time.apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M:%S").timestamp())
    df.sort_values(["trID", "timestamp"], inplace=True)
    df.to_csv("nari-trawling-71vessels_cleaned.csv", index=False)
    print("saved")
    angle_step = 1
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, int(180 / angle_step), endpoint=False)

    executor = ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False))
    for tid in tqdm(df.trID.unique(), leave=True, position=0):
        processes = []
        lines_list = list(time_based_sliding_window(df[df.trID == tid], threshold=60 * 60 * 3)) # 3h
        for lines in tqdm(lines_list, leave=False, position=1, desc=f'Submitting {tid}'):
            future = executor.submit(inner_loop, lines, tested_angles)
            processes.append(future)
            """img = draw_image(lines, resolution=(1920*2, 1920*2), verbose=False)
            h, theta, d = hough_line(np.array(img), theta=tested_angles)
            h_sum = np.log(1 + h).sum(axis=0)
            fft = np.abs(scipy.fft.fft(h_sum))
            matrix.append(fft)"""

        matrix = []
        for process in tqdm(processes, leave=False, position=1, desc=f'Retrieving {tid}'):
            matrix.append(process.result())

        matrix = np.array(matrix).T

        arr = np.zeros((int(180 / angle_step), int(180 / angle_step)))
        arr += matrix[
            min(int(180 / angle_step) - 1, matrix.shape[0] - 1), min(int(180 / angle_step) - 1, matrix.shape[1] - 1)]

        np.save(f"converted/{tid}.npy", matrix)

if __name__ == "__main__":
    main()