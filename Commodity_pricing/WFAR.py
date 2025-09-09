# RMSE of WFAR, FAR, VAR, AR, AR*, SAR, ARX and ARX* models 
# % Input: electricity price data of Nord Pool market in 2013/01/01-2017/12/31
# % Output: WFAR_NP.mat including
# %              1. RMSE_NP:  RMSE matrix (M by 8, M=24 i.e. hourly) of the above 8 models
# %              2. DM: the DM statistics           
# %              3. DMvec: the multivariate DM statistics 

# % References:
# % Uniejewski, B., Weron, R. and Ziel, F. (2018). Variance stabilizing transformations for electricity spot price forecasting, IEEE Transactions on Power Systems 33(2): 2219�C2229.
# % Ziel F. and Weron R. (2018). Day-ahead electricity price forecasting with high dimensional structures: Univariate vs. multivariate models, Energy Economics 70: 396�C420.

# Creat by Jiejie Zhang, last modified on 2018.10.03

# Translated from Matlab to Python by Jiejie Zhang, last modified on 2025.09.09

# 1. Create fdaM functions folder
# 2. Prepare input consilidated price data of Nord Pool market.
import numpy as np
import pandas as pd
import scipy.io

# data prepare Method 1:
# # import sys
# # import subprocess

# # # Install packages from within Python
# # subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "xlrd"])

# # Load data
# data = pd.read_excel('./NPdata/elspot-prices_2013_hourly_eur.xls', engine='xlrd') 

# data prepare Method 2:

# Load electricity price data
data = scipy.io.loadmat('./NPdata/electricity_prices.mat')
fwhole = np.array(data['fwhole'])  # matric in matlab is equvilant to array in python
# forecasting step = 1, i.e. 1-day ahead
step = 1
# take log10(.+1) to avoid the impact of extreme spikes and deal with 0 prices
fwhole = np.log10(fwhole + 1)
M = 24 # no. of obs per day
flog = fwhole
f = flog.reshape(M, -1) # -1 means calculate this dimension
print(f.shape)

