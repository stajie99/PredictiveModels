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
from timewarping_utils import timewarping1, SqrtMeanInverse

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

def replicate_km_gam(KM_gam, N_forecast, wz):
    """
    Replicate KM_gam matrix similar to MATLAB code
    
    Parameters:
    -----------
    KM_gam : numpy array, shape (7, M)
        Karcher means of warping functions
    N_forecast : int
        Forecast horizon
    wz : int
        Window size or additional parameter
    
    Returns:
    --------
    KM_gam_final : numpy array, shape (N_forecast + wz, M)
        Replicated matrix
    """
    
    # MATLAB: KM_gam = repmat(KM_gam, floor((N_forecast+wz)/7), 1);
    n_repeats = int(np.floor((N_forecast + wz) / 7))
    KM_gam_repeated = np.tile(KM_gam, (n_repeats, 1))
    
    # MATLAB: KM_gam=[KM_gam; KM_gam(1:mod((N_forecast+wz),7),:)];
    remaining_rows = (N_forecast + wz) % 7
    
    if remaining_rows > 0:
        KM_gam_remaining = KM_gam[:remaining_rows, :]
        KM_gam_final = np.vstack([KM_gam_repeated, KM_gam_remaining])
    else:
        KM_gam_final = KM_gam_repeated
    
    # Verify final shape
    expected_rows = N_forecast + wz
    actual_rows = KM_gam_final.shape[0]
    
    if actual_rows != expected_rows:
        print(f"Warning: Expected {expected_rows} rows, got {actual_rows}")
    
    return KM_gam_final

# Usage example
# Assuming you have these variables:
# KM_gam = result from KarcherMeansof_warpingfunctions (shape: 7 x M)
# N_forecast = some integer
# wz = some integer

# KM_gam_final = km_gam(KM_gam, N_forecast, wz)


def KarcherMeansof_warpingfunctions(gam, day_no):
    """
    Karcher Mean of Warping functions, grouped by Mon, Tue till Sun.
    
    Parameters:
    -----------
    gam : numpy array, shape (N, T)
        Warping functions for N days, each with T time points
    day_no : int
        Day number of the first day ### (1=Sunday, 2=Monday, ..., 7=Saturday)
        !!! In python, Monday=0, Sunday=6
    
    Returns:
    --------
    KMgamma : numpy array, shape (7, T)
        Karcher means arranged starting from day_no
    gamI_cate : numpy array, shape (7, T)
        Karcher means categorized by day of week (Sunday to Saturday)
    """
    
    # Get dimensions
    N, T = gam.shape  # N days, T time points
    
    # Create day number sequence starting from day_no
    # MATLAB: dayno = [day_no:7 1:day_no-1]'
    if day_no <= 7:
        dayno = np.concatenate([
            np.arange(day_no, 8),  # day_no to 7
            np.arange(1, day_no)   # 1 to day_no-1
        ])
    else:
        raise ValueError("day_no must be between 1 and 7")
    
    dayno = dayno.reshape(-1, 1)  # Make it a column vector
    
    # MATLAB: Dayno = repmat(dayno, floor(N/7), 1)
    n_repeats = int(np.floor(N / 7))
    Dayno = np.tile(dayno, (n_repeats, 1))
    
    # MATLAB: Dayno = [Dayno; Dayno(1:mod(N,7),:)]
    remaining_rows = N % 7
    if remaining_rows > 0:
        Dayno = np.vstack([Dayno, Dayno[:remaining_rows, :]])
    
    # MATLAB: gam = [Dayno gam]
    gam_with_days = np.hstack([Dayno, gam])
    
    # Initialize arrays for each day of week
    gamSun = np.zeros((0, T))  # Empty array with T columns
    gamMon = np.zeros((0, T))
    gamTue = np.zeros((0, T))
    gamWed = np.zeros((0, T))
    gamThu = np.zeros((0, T))
    gamFri = np.zeros((0, T))
    gamSat = np.zeros((0, T))
    
    # Categorize warping functions by day of week
    for i in range(N):
        day_code = gam_with_days[i, 0]  # First column contains day code
        
        # Extract the warping function (excluding day code column)
        warping_func = gam_with_days[i, 1:].reshape(1, -1)
        
        if day_code == 6:     # Sunday
            gamSun = np.vstack([gamSun, warping_func])
        elif day_code == 0:   # Monday
            gamMon = np.vstack([gamMon, warping_func])
        elif day_code == 1:   # Tuesday
            gamTue = np.vstack([gamTue, warping_func])
        elif day_code == 2:   # Wednesday
            gamWed = np.vstack([gamWed, warping_func])
        elif day_code == 3:   # Thursday
            gamThu = np.vstack([gamThu, warping_func])
        elif day_code == 4:   # Friday
            gamFri = np.vstack([gamFri, warping_func])
        elif day_code == 5:   # Saturday
            gamSat = np.vstack([gamSat, warping_func])
        else:
            raise ValueError(f"Invalid day code: {day_code}")
    
    # Initialize gamI_cate array
    gamI_cate = np.zeros((7, T))
    
    # Compute Karcher means for each day category
    # Note: You'll need to implement or import Jiejie_SqrtMean/SqrtMeanInverse function
    if gamSun.shape[0] > 0:
        gamI_cate[0, :] = SqrtMeanInverse(gamSun)  # Mon (index 0)
    if gamMon.shape[0] > 0:
        gamI_cate[1, :] = SqrtMeanInverse(gamMon)  # Tue (index 1)
    if gamTue.shape[0] > 0:
        gamI_cate[2, :] = SqrtMeanInverse(gamTue)  # Wed (index 2)
    if gamWed.shape[0] > 0:
        gamI_cate[3, :] = SqrtMeanInverse(gamWed)  # Thu (index 3)
    if gamThu.shape[0] > 0:
        gamI_cate[4, :] = SqrtMeanInverse(gamThu)  # Fri (index 4)
    if gamFri.shape[0] > 0:
        gamI_cate[5, :] = SqrtMeanInverse(gamFri)  # Sat (index 5)
    if gamSat.shape[0] > 0:
        gamI_cate[6, :] = SqrtMeanInverse(gamSat)  # Sun (index 6)
    
    # Rearrange according to starting day
    # MATLAB: KMgamma = [gamI_cate(day_no:7,:); gamI_cate(1:day_no-1,:)]
    KMgamma = np.vstack([
        gamI_cate[day_no-1:7, :],    # day_no to 7
        gamI_cate[0:day_no-1, :]     # 1 to day_no-1
    ])
    
    return KMgamma, gamI_cate

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
        KM_gam, _ = KarcherMeansof_warpingfunctions(gam.T, day_no)[0]
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