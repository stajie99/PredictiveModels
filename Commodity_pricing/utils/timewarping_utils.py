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