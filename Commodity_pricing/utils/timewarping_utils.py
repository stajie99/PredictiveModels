import numpy as np
import pandas as pd

def timewarping1(f,t,lambda_=0,option_parallel=1,option_closepool=0,option_smooth=0,option_sparam=25,option_showplot=1):
    # % default options
    # % option.parallel = 0; % turns offs MATLAB parallel processing (need
    # % parallel processing toolbox)
    # % option.closepool = 0; % determines wether to close matlabpool
    # % option.smooth = 0; % smooth data using standard box filter
    # % option.sparam = 25; % number of times to run filter
    # % option.showplot = 1; % turns on and off plotting
    option = {
        'parallel': option_parallel,
        'closepool': option_closepool,
        'smooth': option_smooth,
        'sparam': option_sparam,
        'showplot': option_showplot
    }
    # enture t is a numpy array
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    # ensure f is a numpy array
    if not isinstance(f, np.ndarray):
        f = np.array(f)
    # handle 1d arrays
    if t.ndim == 1:
        t = t.reshape(-1, 1) # make it a 2d column vector
    
    a = t.shape[0]
    if a != 1:
        t = t.T # requires t to be a row vector
    # handle edge case where t has only 1 element
    if len(t.flatten()) > 1:
        binsize = np.mean(np.diff(t.flatten()))
    else:
        binsize = 0 # or np.nan, depending on needs
    
    M, N = f.shape
    f0 = f.copy()

    # choose to smooth f or not
    f = smooth_f(f, option)






    return fn,qn,q0,fmean,mqn,gam,psi,stats

def smooth_f(f, option):
    # smoothing with proper edge handling
    if option['smooth'] != 1:
        return f.copy()
    
    M, N = f.shape
    f_smoothed = f.copy()
    for r in range(option['sparam']):
        # for interior points
        if M>2:
            f_smoothed[1:M-1, :] = (f_smoothed[0:M-2, :] + 2 * f_smoothed[1:M-1, :] + f_smoothed[2:M, :]) / 4

    return f_smoothed