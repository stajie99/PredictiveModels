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
from timewarping_utils import timewarping1, KarcherMeansof_warpingfunctions, replicate_km_gam

# data prepare Method 1:
# # import sys
# # import subprocess

# # # Install packages from within Python
# # subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "xlrd"])

# # Load data
# data = pd.read_excel('./NPdata/elspot-prices_2013_hourly_eur.xls', engine='xlrd') 

# data prepare Method 2:

def load_electricity_data():
    """Load and preprocess electricity price data"""
    # 1. Load electricity price data
    data = scipy.io.loadmat('./NPdata/electricity_prices.mat')
    fwhole = np.array(data['fwhole'])
    fdatewhole = np.array(data['fdatewhole']).reshape(-1, 1)
    
    # 2. Data pre-processing
    # forecasting step = 1, i.e. 1-day ahead
    step = 1
    # take log10(.+1) to avoid the impact of extreme spikes and deal with 0 prices
    fwhole = np.log10(fwhole + 1)
    M = 24  # no. of obs per day
    flog = fwhole
    f = flog.reshape(M, -1)
    
    print(f"  Electricity Price Data shape: {f.shape}")
    N = f.shape[1]
    print(f"  Number of days: {N}")
    
    return f, fdatewhole, step, M, N

def setup_forecasting_parameters(f, fdatewhole, step, M, N):
    """Set up forecasting parameters and windowing"""
    train_sample = 300
    wz = 30  # rolling window size, alternatives 60, 180
    # 3. Prepare prices and dates data for time warping
    f_wz = f[:, train_sample-wz : train_sample-1]
    N_forecast = N - train_sample - step + 1
    
    # Get day number and day of week for the first day in the window
    first_date_str = fdatewhole[train_sample-wz][0][0]
    first_date = datetime.strptime(first_date_str, '%d/%m/%Y')
    firstday_inwindow_dayno = first_date.weekday()
    firstday_inwindow_dayofweek = first_date.strftime("%A")
    
    print(f"  First day in window: {firstday_inwindow_dayofweek} (day {firstday_inwindow_dayno})")
    print(f"  Forecast window size: {wz}")
    print(f"  Number of forecasted days: {N_forecast}")
    
    return f_wz, train_sample, wz, N_forecast, firstday_inwindow_dayno, firstday_inwindow_dayofweek
# >>> firstday_inwindow_dayno
# 5
# >>> firstday_inwindow_dayofweek
# 'Saturday'
# - Standard numbering: Monday=0, Tuesday=1, ..., Sunday=6

def setup_warping_parameters(M):
    """Set up time warping parameters"""
    # 4. Time Warping by fixed warping functions
    # 4.1 default values for warping parameters
    warping_params = {
        'lambda_': 0,
        'option_parallel': 0,
        'option_closepool': 0,
        'option_smooth': 0,
        'option_sparam': 25,
        'option_showplot': 0
    }
    
    # t = np.arange(1, M+1).reshape(-1, 1)
    t = np.arange(1, M+1)

    return warping_params, t


def debug_timewarping(f_wz, t, warping_params):
    """Debug wrapper for timewarping function"""
    print(f"  Input f_wz shape: {f_wz.shape}")
    print(f"  Input t shape: {t.shape}")
    print(f"  Warping parameters: {warping_params}")
    
    try:
        # 4.2 Do time warping
        # gam is the daily warping function
        result = timewarping1(
            f_wz, t, 
            warping_params['lambda_'], 
            warping_params['option_parallel'], 
            warping_params['option_closepool'], 
            warping_params['option_smooth'], 
            warping_params['option_sparam'], 
            warping_params['option_showplot']
        )
        _, _, _, _, _, gam, _, _ = result
        print(f"Time warping completed successfully")
        print(f"Gam shape: {gam.shape if hasattr(gam, 'shape') else 'No shape attribute'}")
        
        return gam, result
        
    except Exception as e:
        print(f"Error in timewarping: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
# def main(): defines a function called main() that serves as the entry point of a Python program when 
# the script is executed directly.
    """Main execution function"""
    print("Starting electricity price forecasting...")
    
    try:
        # Load data
        print("Step 1: Loading data...")
        f, fdatewhole, step, M, N = load_electricity_data()
        
        # Setup parameters
        print("Step 2: Setting up forecasting parameters...")
        f_wz, train_sample, wz, N_forecast, day_no, day_name = setup_forecasting_parameters(
            f, fdatewhole, step, M, N
        )
        
        # Setup warping parameters
        print("Step 3: Setting up warping parameters...")
        warping_params, t = setup_warping_parameters(M)
        
        # Perform time warping with debugging
        print("Step 4: Performing time warping...")
        gam, full_result = debug_timewarping(f_wz, t, warping_params)
        
        if gam is not None:
            print("Time warping completed successfully!")
            # You can add further processing here
        else:
            print("Time warping failed.")

        # Get Karcher Mean of warping functions by day of week
        print("Step 5: Getting the Karcher Mean of warping functions as 7 seasonal fixed warping functions...")
        KM_gam, _ = KarcherMeansof_warpingfunctions(gam, day_no)
        KM_gam_final = replicate_km_gam(KM_gam, N_forecast, wz)
        print(f'KM_gam_final is {KM_gam_final}')
            
    except Exception as e:
        print(f"Main execution error: {e}")
        import traceback
        traceback.print_exc()

# Add this to help debug the timewarping1 function
def inspect_timewarping1():
    """Helper function to inspect the timewarping1 function"""
    try:
        # Try to import and inspect the function
        from inspect import signature
        sig = signature(timewarping1)
        print(f"timewarping1 signature: {sig}")
        print(f"timewarping1 module: {timewarping1.__module__}")
    except Exception as e:
        print(f"Cannot inspect timewarping1: {e}")

if __name__ == "__main__":
# This condition checks if the script is being run directly (not imported)
# 1. __name__ is a special variable in Python
# 2. "__main__" means the script is executed directly
# 3. If the script is imported as a module, __name__ becomes the module name

    # Uncomment the line below if you need to inspect the timewarping1 function
    # inspect_timewarping1()
    
    main()