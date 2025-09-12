from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# Create sample time series
series1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
series2 = np.array([1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

# Fast approximate DTW
distance, path = fastdtw(series1.reshape(-1, 1), 
                         series2.reshape(-1, 1), 
                         dist=euclidean)

print(f"FastDTW Distance: {distance}")
print(f"Warping Path: {path[:10]}...")  # Show first 10 points