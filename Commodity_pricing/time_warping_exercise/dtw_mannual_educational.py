import numpy as np
def dtw_manual(series1, series2):
    """Manual DTW implementation for understanding"""
    n, m = len(series1), len(series2)
    dtw_matrix = np.zeros((n+1, m+1))
    
    # Initialize with infinity
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(series1[i-1] - series2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )
    
    return dtw_matrix[n, m], dtw_matrix

# Usage
# Create sample time series
series1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
series2 = np.array([1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

distance, matrix = dtw_manual(series1, series2)
print(f"Manual DTW Distance: {distance}")