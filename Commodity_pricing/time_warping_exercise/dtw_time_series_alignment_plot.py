# AttributeError: 'function' object has no attribute 'warping_path
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import dtw, dtw_path
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def plot_dtw_alignment(series1, series2, path):
    plt.figure(figsize=(12, 8))
    
    # Plot both series
    plt.subplot(2, 1, 1)
    plt.plot(series1, label='Series 1', marker='o')
    plt.plot(series2, label='Series 2', marker='s')
    plt.legend()
    plt.title('Original Time Series')
    
    # Plot alignment
    plt.subplot(2, 1, 2)
    for (i, j) in path:
        plt.plot([i, j], [series1[i], series2[j]], 'k-', alpha=0.1)
    plt.plot(series1, 'o-', label='Series 1')
    plt.plot(series2, 's-', label='Series 2')
    plt.legend()
    plt.title('DTW Alignment')
    
    plt.tight_layout()
    plt.show()

# Generate and plot

# Create sample time series
series1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
series2 = np.array([1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

path = dtw.warping_path(series1, series2)
plot_dtw_alignment(series1, series2, path)