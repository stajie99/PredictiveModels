import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import interp1d, CubicSpline
# import matlab.engine
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.integrate import simpson
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

def main():
    result = timewarping1()
    return result

if __name__ == "__main__":
    main()  # Run as script: python timewarping_utils.py
# Another script can import and use functions:
# from timewarping_utils import timewarping1



def timewarping1(f,t,lambda_=0,
                 option_parallel=1,
                 option_closepool=0,
                 option_smooth=0,
                 option_sparam=25,
                 option_showplot=1):
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

    # # enture t is a numpy array
    # if not isinstance(t, np.ndarray):
    #     t = np.array(t)

    # ensure f is a numpy array
    if not isinstance(f, np.ndarray):
        f = np.array(f) # M by N
    # # handle 1d arrays
    # if t.ndim == 1:
    #     t = t.reshape(-1, 1) # make it a 2d column vector
    
    # a = t.shape[0]
    # if a != 1:
    #     t = t.T # requires t to be a row vector


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
    dis_qq = np.sqrt(np.sum((q - mnq[:, np.newaxis])**2, axis=0))  #Expand the mean column: Instead of explicitly creating a matrix of ones, we use broadcasting by adding a new axis to make mnq a column vector.
    # 2.3 Find the index (day/column) of minimum distance
    min_ind = np.argmin(dis_qq)
    #     Extract the column with minimum distance
    mq = q[:, min_ind]
    mf = f[:, min_ind]
    # Finish initialization

    # Side facts:
    #   q: numpy array of shape (features, N, time_points) or similar
    #   mq: numpy array
    #   t: time vector of length M
    #   binsize: scalar value
    #   M: number of time points
    #   N: number of samples
    gam = np.zeros((N, q.shape[0]))
    for k in range(N):
        # the k-th sample, first time/interplation point
        q_c = q[:, k].T 
        # the template function
        mq_c = mq.T
        # Method 1: obtaining warping function
        # # np.linalg.norm(mq_c)： L2 norm for vectors
        # G, T = DynamicProgrammingQ2(mq_c / np.linalg.norm(mq_c), t, q_c / np.linalg.norm(q_c), t, t, t)

        # # Interpolate
        # # Step 1: Create an interpolation function
        # interp_function = interp1d(T, G, kind='linear', fill_value='extrapolate')
        # # Step 2: Use that function to interpolate at points 't'
        # gam0 = interp_function(t)

        # Method 2: (fast simplified) obtaining warping function
        # Fast approximate DTW
        ts1 = mq_c / np.linalg.norm(mq_c)
        ts2 = q_c / np.linalg.norm(q_c)
        distance, gam0 = fastdtw(ts1.reshape(-1, 1), 
                                ts2.reshape(-1, 1), 
                                dist=euclidean)
        # Normaliza and smoothing warping functions
        print(f'For day {k}, warping function is {gam0}')

        warping_func, t_norm, warping_norm = monotonic_smooth_warping(gam0, len(t), len(t))
        t_eval = np.linspace(0, 1, len(t))
        warping_smooth = warping_func(t_eval)
        print(f'For day {k}, warping function (discretized) is {warping_smooth}')

    gamI = SqrtMeanInverse(gam)

    # Compute gradiant 
    dt = 1 / (M - 1)
    gamI_dev = np.gradient(gamI, dt)

    # Interpolate mf using the warping function
    mf_interpolated = interp1d(t, mf, kind='linear', fill_value='extrapolate')(
        (t[-1] - t[0]) * gamI + t[0]
    ).T
    gradient_mf = np.gradient(mf_interpolated, binsize)
    mq = gradient_mf / np.sqrt(np.abs(gradient_mf) + np.finfo(float).eps)


    #### Compute Karcher Mean in SRVF (Square-Root Velocity Function) space with dynamic time warping
    # return mq[:, r+2], q_evolution[:, :, r+2], f_evolution[:, :, r+2], q[:, :, 0], f[:, :, 0], ds
    ## Aligned data & stats
    mq_n, q_n, f_n, q0, f0, ds = compute_karcher_mean_f(q, f, t, lambda_val=0, MaxItr=30, tol=1e-2)
    
    mean_f0 = np.mean(f0, axis=1)
    std_f0 = np.std(f0, axis=1, ddof=0)
    mean_fn = np.mean(f_n, axis=1)
    std_fn = np.std(f_n, axis=1, ddof=0)
    fmean = mean_f0[0] + cumulative_trapezoid(t, mq_n * np.abs(mq_n), initial=0)

    # Initializa fgam and interpolate
    fgam = np.zeros((M, N))
    for ii in range(N):
        # create new time points for interpolation
        new_t = (t[-1] - t[0]) * gam[ii, :] + t[0]

        # interpolate fmean at new time points
        f_interp = interp1d(t, fmean, kind='linear', fill_value='extrapolate')
        fgam[:, ii] = f_interp(new_t)

    # calculate variance along axis 1 (across columns)
    var_fgam = np.var(fgam, axis=1, ddof=0)

    # calculate statistics
    stats={}
    stats['orig_var'] = trapezoid(t, std_f0**2)
    stats['amp_var'] = trapezoid(t, std_fn**2)
    stats['phase_var'] = trapezoid(t, var_fgam)

    # tramspose gam and compute gradient
    gam = gam.T
    binsize = t[1] - t[0]

    # compute gradient
    fy = np.gradient(gam, binsize, axis=0) # gradient along rows (time axis)

    # calculate psi
    psi = np.sqrt(fy + np.finfo(float).eps)

    if option.showplot == 1:
        # Create normalized x-axis for warping functions (0 to 1)
        x_norm = np.arrange(M) / (M - 1)
        # Figure 2: Warping functions
        plt.figure(2)
        plt.clf()
        # Transpose gam to plot each column as a line
        plt.plot(x_norm, gam.T, linewidth=1) 
        plt.axis('square')
        plt.title('Warping functions', fontsize=16)
        plt.xlabel('Normalized time')
        plt.ylabel('r(t)')

        # Figure 3: Warped data
        plt.figure(3)
        plt.clf()
        plt.plot(t, f_n, linewidth=1)
        plt.title(f'Warped data', fontsize=16)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        # Figure 4: Original data with mean +/- STD
        plt.figure(4)
        plt.clf()
        plt.plot(t, mean_f0, 'b-', linewidth=1, label='Mean')
        plt.plot(t, mean_f0 + std_f0, 'r-', linewidth=1, label='Mean + STD')
        plt.plot(t, mean_f0 - std_f0, 'g-', linewidth=1, label='Mean - STD')
        plt.title('Original data: Mean +/- STD', fontsize=16)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()

        # Figure 5: Warped data with mean +/- STD
        plt.figure(5)
        plt.clf()
        plt.plot(t, mean_fn, 'b-', linewidth=1, label='Mean')
        plt.plot(t, mean_fn + std_fn, 'r-', linewidth=1, label='Mean + STD')
        plt.plot(t, mean_fn - std_fn, 'g-', linewidth=1, label='Mean - STD')
        plt.title(f'Warped data: Mean +/- STD', fontsize=16)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()

        # Figure 6: f_mean
        plt.figure(6)
        plt.clf()
        plt.plot(t, fmean, 'g', linewidth=1)
        plt.title(f'f_mean', fontsize=16)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        # Show all plots
        plt.show()

    return f_n, q_n, q0, fmean, mq_n, gam, psi, stats

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

def SqrtMeanInverse(gam):
#       [n,T] = size(gam); → Gets dimensions of input matrix gam
#       dT = 1/(T-1); → Computes step size
#       psi = zeros(n,T-1); → Preallocates output matrix
#       The loop → Computes SRSF (Square-Root Slope Function) for each row
    n, T = gam.shape
    dT = 1 / (T - 1)
    psi = np.zeros((n, T - 1))
    for i in range(n):
        # the SRSF of the inital warping functions
        # 2-norm of psi is 1
        psi[i, :] = np.sqrt(np.diff(gam[i, :]) / dT + np.finfo(float).eps)
    # Compute Karcher Mean of warping functions
    # Find direction
    mnpsi = np.mean(psi, axis = 0) # mean along rows
    dis_qq = np.sqrt(np.sum((psi.T - mnpsi.reshape(-1, 1) * np.ones((1,n)))**2, axis=1))

    min_ind = np.argmin(dis_qq)
    mu = psi[min_ind, :]

    t = 1 # step size
    maxiter = 20
    lvm = np.zeros(maxiter)
    vec = np.zeros((n, psi.shape[1]))
    mu, _, _ = compute_warping_mean(psi, maxiter = 20, t=1, tol=1e-6)

    gam_mu = np.zeros(len(mu) + 1)
    gam_mu[1:] = np.cumsum(mu**2)
    gam_mu = gam_mu / len(mu) # divide by T which is len(mu) + 1

    gamI = invertGamma(gam_mu)

    return gamI


def compute_warping_mean(psi, maxiter = 20, t=1, tol=1e-6):
    # Compute the mean of warping function using Fisher-Rao metric
    # parameters:
    #   psi: numpy array of shape (n, T-1)
    #       the SRSF matrix
    #   maxiter: int, maximum iterations
    #   t: float, step size
    #   tol: float, convergence tolerance
    # returns:
    #   mu: final mean SRSF
    #   lvm: array of objective values
    #   iter: number of iterations performed
    n, T_minus_1 = psi.shape
    T = T_minus_1 + 1
    dT = 1 / T_minus_1
    # Initialize variables
    mu = psi[np.argmin(np.linalg.norm(psi - np.mean(psi, axis=0), axis=1))].copy()
    lvm = np.zeros(maxiter)
    vec = np.zeros_like(psi)

    # Time points for integration
    time_points = np.linspace(0, 1, T_minus_1)
    
    for iter in range(maxiter):
        for i in range(n):
            v = psi[i,:] - mu
            # Inner product using Simpson intergration
            dot1 = simpson(time_points, mu * psi[i, 1])
            # Clamp dot product to [-1, 1] for acos
            dot_limited = np.clip(dot1, -1, 1)
            # Fisher-Rao distance
            len_val = np.arccos(dot_limited)

            if len_val > 0.0001:
                # Shooting vector
                vec[i, :] = (len_val / np.sin(len_val)) * (psi[i, :] - np.cos(len_val) * mu)
            else:
                vec[i, :] = np.zeros(T_minus_1)
            
            # Average direction
            vm = np.mean(vec, axis=0)
            # Length of vm (L2 norm weighted by dT)
            lvm[iter] = np.sqrt(np.sum(vm**2) * dT)

            # Update mean using exponential map
            if lvm[iter] > 1e-10: # avoid division by zero
                mu = (np.cos(t * lvm[iter]) * mu + (np.sin(t * lvm[iter]) / lvm[iter]) * vm)
            else:
                mu = mu.copy()
            # check convergence
            if lvm[iter] < tol or iter == maxiter - 1:
                break

            return mu, lvm[:iter+1], iter+1

def invertGamma(gam):
    # Invert a warping function gamma
    # Parameters:
    #   gam: numpy array of shape (N,)
    #       warping function (monotonically increasing from 0 to 1)
    # returns:
    #   gamI: numpy array of shape (N,)
    #       Inverse warping function
    N = len(gam)
    x = np.linspace(0, 1, N) # N uniform distribution points on [0, 1]
    # Create inverse function using interpolation
    # we want gamI such that gamI(s) = t, where gam(t) = s
    interp_func = interp1d(gam, x, kind='linear',
                           bounds_error=False,
                           fill_value='extrapolate')
    gamI = interp_func(x) # interpolate at original x points
    #
    # handle NaN values (edge cases)
    if np.isnan(gamI[-2]): # gamI(N-1) in matlab
        gamI[-2] = 0.95
    if np.isnan(gamI[-1]):
        gamI[-1] = 1
    else:
        # normalize to ensure endpoint is exactly 1
        gamI = gamI / gamI[-1]

    return gamI 


def compute_karcher_mean_f(q, f, t, lambda_val=0, MaxItr=30, tol=1e-2):
    # Compute Karcher mean of functions in SRVF space
    # Parameters:
    #   q: numpy array of shape (M, N)
    #       SRVF representations of functions
    #   f: numpy array of shape (M, N)
    #       orginal functions
    #   t: numpy arrage of shape (M,)
    #       Time points
    #   lambda_val: float
    #       Regularization parameter
    #   MaxItr: int
    #       Maximum iterations
    #   tol: float
    #       Convergence tolerance
    #   Returns:
    #       mq: numpy array
    #           Karcher mean in SRVF space
    #       q_evolution: list
    #           Evolution of q through interations
    #       f_evolution: list
    #           Evolution of f through interations
    #       ds: list
    #           Objective function values

    M, N = q.shape
    binsize = t[1] - t[0]
    eps = np.finfo(float).eps

    print(f'Computing Karcher mean of {N} functions in SRVF space...')
    # Initilize arrays
    ds = [np.inf]
    qun = np.zeros(MaxItr)
    mq = np.zeros((M, MaxItr + 1))
    mq[:, 0] = np.mean(q, axis=1) # initial mean

    # Storage for evolution
    # q_evolution = [q.copy()]
    # f_evolution = [f.copy()]
    q_evolution = np.zeros((MaxItr, q.shape[0], q.shape[1]))
    f_evolution = np.zeros((MaxItr, q.shape[0], q.shape[1]))

    for r in range(MaxItr):
        # Matching Step - find optimal warping for each function
        gam = np.zeros((N, M))
        gam_dev = np.zeros((N, M))
        q_temp = np.zeros((M, N))
        f_temp = np.zeros((M, N))

        for k in range(N):
            q_c = q[:, k]
            mq_c = mq[:, r]

            # Normalize and call dynamic programming if available

            # Method 2: (fast simplified) obtaining warping function
            # Fast approximate DTW
            ts1 = mq_c / np.linalg.norm(mq_c)
            ts2 = q_c / np.linalg.norm(q_c)
            distance, gam0 = fastdtw(ts1.reshape(-1, 1), 
                                    ts2.reshape(-1, 1), 
                                    dist=euclidean)
            # gam0 is the warping path - it defines how the two time series are aligned to 
            # minimize the overall distance. It's a list of coordinate pairs that map points from ts1 to 
            # points in ts2.
            # Boundary Conditions of warping functions
            # 1. Starts at: (0, 0) - first points of both series
            # 2. Ends at: (len(ts1)-1, len(ts2)-1) - last points of both series
            # Normalize warping function to [0, 1]
            # print(f'For day {k}, warping function is {gam0}')
            gam_temp = (gam0 - gam0[0]) / (gam0[-1] - gam0[0])
            print(f'For day {k}, warping function is {gam_temp}')
            gam[k,:] = interp1d(t, gam_temp, kind='linear', bounds_error=False, fill_value='extrapolate')
            

            # Compute derivative of warping function
            gam_dev[k, :] = np.gradient(gam[k, :], 1 / (M - 1))

            # Warp the original function
            warped_time = (t[-1] - t[0]) * gam[k, :] + t[0]
            f_interp = interp1d(t, f[:, k], kind='linear', bounds_error=False, fill_value='extrapolate')
            f_temp[:, k] = f_interp(warped_time)

            # Compute SRVF of warped function
            grad_f = np.gradient(f_temp[:, k], binsize)
            q_temp[:, k] = grad_f / np.sqrt(np.abs(grad_f) + eps)

        # Store results
        # q_evolution.append(q_temp.copy())
        # f_evolution.append(f_temp.copy())
        q_evolution[r, :, :] = q_temp
        f_evolution[r, :, :] = f_temp

        # Compute the objective function
        diff_srvf = mq[:, r][:, np.newaxis] - q_temp
        srvf_term = np.sum(simpson(t, diff_srvf**2, axis=0))

        gam_dev_term = 1 - np.sqrt(gam_dev.T)
        warp_term = lambda_val * np.sum(simpson(t, gam_dev_term**2, axis=0))

        ds.append(srvf_term + warp_term)

        # Minimization Step - compute new mean
        mq[:, r+1] = np.mean(q_temp, axis=1)

        # Check convergence
        qun[r] = np.linalg.norm(mq[:, r+1] - mq[:,r]) / np.linalg.norm(mq[:, r])
        print(f'Iteration {r+1}: qun = {qun[r]:.6f}, ds = {ds[r+1]:.6f}')

        if qun[r] < tol or r >= MaxItr -1:
            break

        ## return mq[:, :r+2], q_evolution, f_evolution, ds


        r = r + 1
        # Sequential processing
        for k in range(N):
            q_c = q[:, k].T
            mq_c = mq[:, r+1].T
            
            ts1 = mq_c / np.linalg.norm(mq_c)
            ts2 = q_c / np.linalg.norm(q_c)
            
            distance, gam0 = fastdtw(ts1.reshape(-1, 1), 
                                    ts2.reshape(-1, 1), 
                                    dist=euclidean)
            
            gam[k, :] = (gam0 - gam0[0]) / (gam0[-1] - gam0[0])  # slight change on scale
            gam_dev[k, :] = np.gradient(gam[k, :], 1/(M-1))

        gamI = SqrtMeanInverse(gam)
        gamI_dev = np.gradient(gamI, 1/(M-1))

        # Interpolate mq
        new_t_mq = (t[-1] - t[0]) * gamI + t[0]
        mq_interp = interp1d(t, mq[:, r+1], kind='linear', axis=0, fill_value='extrapolate')
        mq[:, r+2] = (mq_interp(new_t_mq) * np.sqrt(gamI_dev)).T

        # Process each k
        for k in range(N):
            # Interpolate q
            new_t = (t[-1] - t[0]) * gamI + t[0]
            
            q_interp = interp1d(t, q_evolution[:, k, r+1], kind='linear', axis=0, fill_value='extrapolate')
            q_evolution[r+2, :, k] = (q_interp(new_t) * np.sqrt(gamI_dev)).T
            
            # Interpolate f
            f_interp = interp1d(t, f_evolution[:, k, r+1], kind='linear', axis=0, fill_value='extrapolate')
            f_evolution[r+2, :, k] = f_interp(new_t).T
            
            # Interpolate gam
            gam_interp = interp1d(t, gam[k, :], kind='linear', fill_value='extrapolate')
            gam[k, :] = gam_interp(new_t)

    return mq[:, r+2], q_evolution[r+2, :, :], f_evolution[r+2, :, :], q, f, ds


def monotonic_smooth_warping(gam0, ts1_length, ts2_length, method='spline'):
    """
    Create monotonic smooth warping function using PCHIP interpolation
    Preserves monotonicity of the warping path
    """
    warping_func = None
    # Convert warping path to arrays
    gam0_array = np.array(gam0)
    ts1_indices = gam0_array[:, 0] # indices from first time series
    ts2_indices = gam0_array[:, 1] # indices from second time series
    print(f'indices from first time series are {ts1_indices}')
    print(f'indices from second time series are {ts2_indices}')

    # Add small epsilon to handle numerical issues
    epsilon = 1e-6
    ts1_indices = ts1_indices + epsilon * np.arange(len(ts1_indices))
    ts2_indices = ts2_indices + epsilon * np.arange(len(ts2_indices))

    # Normalize to [0, 1] range
    t_norm = ts1_indices / (ts1_length - 1)
    warping_norm = ts2_indices / (ts2_length - 1)
    
    # Trim values > 1 to 1
    trimmed_t_norm = np.clip(t_norm, None, 1)  # Clip upper bound to 1
    # Remove duplicates and sort
    t_norm_unique, indices = np.unique(trimmed_t_norm, return_index=True)
    warping_norm_unique = warping_norm[indices]
    # handle right boundary: adjust to 1
    if warping_norm_unique[-1] != 1:
        warping_norm_unique[-1] = 1
    print(f'After normalization to [0, 1], t_norm is {t_norm_unique}')
    print(f'After normalization to [0, 1], warping_norm is {warping_norm_unique}')
    
    # Check if array has any duplicates
    has_duplicates = len(t_norm_unique) != len(np.unique(t_norm_unique))
    print(f"t_norm has duplicates: {has_duplicates}")

    has_duplicates = len(warping_norm_unique) != len(np.unique(warping_norm_unique))
    print(f"warping_norm has duplicates: {has_duplicates}")
    
    if method == 'linear':
        # # Create Linear interpolation function
        warping_func = interp1d(t_norm_unique, warping_norm_unique, 
                           kind='linear', fill_value=np.nan)
        
    elif method == 'quadratic':
        # # Create Quadratic interpolation function
        warping_func = interp1d(t_norm_unique, warping_norm_unique, 
                           kind='quadratic', fill_value=np.nan)
    
    elif method == 'cubic':
        # Create Cubic interpolation function
        warping_func = interp1d(t_norm_unique, warping_norm_unique, 
                           kind='cubic', fill_value=np.nan)
        # print('cubic done')
    
    elif method == 'spline':
        # Cubic spline. bc_type: Boundary condition type.
        warping_func = CubicSpline(t_norm_unique, warping_norm_unique,
                              bc_type='natural')
    
    elif method == 'savgol':
        # Apply Savitzky-Golay filter for smoothing
        if len(warping_norm_unique) > 11:  # Need enough points
            window_length = min(11, len(warping_norm_unique) - (1 - len(warping_norm_unique) % 2))
            warping_smooth = savgol_filter(warping_norm_unique, window_length, 3)
            warping_func = interp1d(t_norm_unique, warping_smooth, 
                               kind='cubic', fill_value=np.nan)
        else:
            warping_func = interp1d(t_norm_unique, warping_norm_unique, 
                               kind='linear', fill_value=np.nan)
    elif method == 'PchipInterpolator':
    # PCHIP preserves monotonicity: (PCHIP stands for Piecewise Cubic Hermite Interpolating Polynomial).
        warping_func = PchipInterpolator(t_norm_unique, warping_norm_unique)

    return warping_func, t_norm_unique, warping_norm_unique

if __name__ == "__main__":
    main()  # Run as script: python my_module.py

# Another script can import and use functions:
# from my_module import load_data, process_data