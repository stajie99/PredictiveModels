import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwviz

# Create sample time series
series1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
series2 = np.array([1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

# Calculate DTW distance
distance = dtw.distance(series1, series2)
print(f"DTW Distance: {distance}")

# Get the full distance matrix
distance_matrix = dtw.distance_matrix(np.array([series1, series2]))
print(f"Distance Matrix:\n{distance_matrix}")

# Visualize the warping path
path = dtw.warping_path(series1, series2)
dtwviz.plot_warping(series1, series2, path, "DTW Warping Path")