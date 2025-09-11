import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import interpld

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
        f = np.array(f) # M by N
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
    # choose to plot f or not
    if option['showplot'] == 1:
        plt.figure(1)
        plt.clf()
        plt.plot(t, f)
        plt.title('Original data')
        plt.pause(0.1)


    # 1. Compute the q-function of the plot
    fy = calculate_gradient(f, binsize)
    q = calculate_q(fy)

    # 2. Initialization using the original f space
    # 2.1 Calculate mean along rows (horizontal): axis=1
    mnq = np.mean(q, axis=1)
    # 2.2 Calculate Euclidean distances from each column to the mean mnq
    #   i.e. sqrt(sum((q - mnq*one(1, N)).^2, 1))
    dis_qq = np.sqrt(np.sum((q - mnq[:, np.newasis])**2, axis=0))  #Expand the mean column: Instead of explicitly creating a matrix of ones, we use broadcasting by adding a new axis to make mnq a column vector.
    # 2.3 Find the index (day/column) of minimum distance
    min_ind = np.argmin(dis_qq)
    #     Extract the column with minimum distance
    mq = q[:, min_ind]
    mf = f[:, min_ind]
    # Finish initialization

    gam =








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

def calculate_gradient(f, binsize):
    # parameters:
    # f: numpy array
        # input array (2d typically)
    # binsize: float
        # spacing between points
    # returns:
        # fy: numpy array
            # vertical gradient (derivative along columns)
    fy = np.gradient(f, binsize, axis = 1)
    return fy

def calculate_q(fy):
    # calculate q = fy / sqrt(|fy| + eps)
    # parameters:
    # fy: numpy array
        # gradient values
    # returns:
        # q: numpy array
            # Normalized gradient values
    eps = np.finfo(float).eps # machine epsilon for numerical stability
    q = fy / np.sqrt(np.abs(fy) + eps)
    
    return q