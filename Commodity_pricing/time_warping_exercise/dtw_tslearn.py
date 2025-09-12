# not working yet
# File "D:\Python\Lib\site-packages\tslearn\metrics\dtw_variants.py", line 807, in dtw
    # return _njit_dtw(s1, s2, mask=mask)
import numpy as np
from tslearn.metrics import dtw, dtw_path
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


# Create sample time series
series1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
series2 = np.array([1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

# Normalize time series
scaler = TimeSeriesScalerMeanVariance()
series1_norm = scaler.fit_transform(series1.reshape(-1, 1))
series2_norm = scaler.fit_transform(series2.reshape(-1, 1))

# Calculate DTW
dtw_distance = dtw(series1_norm, series2_norm)
print(f"DTW Distance: {dtw_distance}")

# Get the warping path
path, sim = dtw_path(series1_norm, series2_norm)
print(f"Warping Path: {path}")