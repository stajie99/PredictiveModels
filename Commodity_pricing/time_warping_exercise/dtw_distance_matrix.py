import numpy as np
import matplotlib.pyplot as plt
# library 1
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwviz
# library 2
from tslearn.metrics import dtw, dtw_path
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# library 3
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def plot_dtw_matrix(series1, series2):
    distance_matrix = dtw.distance_matrix_fast(np.array([series1, series2]))
    path = dtw.warping_path(series1, series2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='viridis', origin='lower')
    plt.plot([p[1] for p in path], [p[0] for p in path], 'r-', linewidth=2)
    plt.colorbar(label='Accumulated Distance')
    plt.title('DTW Distance Matrix with Warping Path')
    plt.xlabel('Series 2 Index')
    plt.ylabel('Series 1 Index')
    plt.show()

# Create sample time series
series1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
series2 = np.array([1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

plot_dtw_matrix(series1, series2)