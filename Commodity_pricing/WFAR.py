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
from datetime import datetime, date, timedelta

# data prepare Method 1:
# # import sys
# # import subprocess

# # # Install packages from within Python
# # subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "xlrd"])

# # Load data
# data = pd.read_excel('./NPdata/elspot-prices_2013_hourly_eur.xls', engine='xlrd') 

# data prepare Method 2:

# 1. Load electricity price data
data = scipy.io.loadmat('./NPdata/electricity_prices.mat')
fwhole = np.array(data['fwhole'])  # matric in matlab is equvilant to array in python
fdatewhole = np.array(data['fdatewhole']).reshape(-1, 1)
# 2. Data pre-processing
# forecasting step = 1, i.e. 1-day ahead
step = 1
# take log10(.+1) to avoid the impact of extreme spikes and deal with 0 prices
fwhole = np.log10(fwhole + 1)
M = 24 # no. of obs per day
flog = fwhole
f = flog.reshape(M, -1) # -1 means calculate this dimension
print(f.shape)
N = f.shape[1]
print(N)
train_sample = 300
wz = 30 # rolling window size, alternatives 60, 180

# 3. Prepare prices and dates data for time warping
f_wz = f[:, train_sample-wz : train_sample-1]
N_forecast = N-train_sample-step+1

# get day number and day of week for the first day in the window
firstday_inwindow_dayno = datetime.strptime(fdatewhole[train_sample-wz][0][0], '%d/%m/%Y').weekday()
firstday_inwindow_dayofweek = datetime.strptime(fdatewhole[train_sample-wz][0][0], '%d/%m/%Y').strftime("%A")
# >>> firstday_inwindow_dayno
# 5
# >>> firstday_inwindow_dayofweek
# 'Saturday'
# - Standard numbering: Monday=0, Tuesday=1, ..., Sunday=6

# 4. Time Warping by fixed warping functions
# 4.1 default values for warping parameters
lambda_ = 0
option_parallel = 0
option_closepool = 0
option_smooth = 0
option_sparam = 25
option_showplot = 0
t = np.arange(1, M+1).reshape(-1, 1)
# 4.2 Do time warping
## gam is the daily warping function
_, _, _, _, _, gam, _, _ = timewarping1(f_wz, t, lambda_, option_parallel, option_closepool, option_smooth, option_sparam, option_showplot)
