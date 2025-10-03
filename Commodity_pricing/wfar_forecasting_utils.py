def wfar_forecasting():

    # ------------------------------------WFAR ----------------------------------
    CAfd_o_forcvalFAR1315 = []

    for ind in range(train_sample + step, N):
        # 1. time warping
        f_wz = f[:, ind - step - wz + 1:ind - step]  # prices in the rolling window
        fn = np.zeros((M, wz))  # for warped prices
        
        for i in range(wz):
            # time warping with fixed warping function
            fn[:, i] = interp1d(t, f_wz[:, i], 
                            kind='linear', 
                            bounds_error=False, 
                            fill_value='extrapolate')(
                (t[-1] - t[0]) * KM_gam[ind - (train_sample + step) + i, :] + t[0]
            )
        
        # 2. FAR
        # 2.1.--------------------------- smoothing ----------------------
        CAlogall = fn  # M x wz
        # Note: You'll need to implement create_fourier_basis, smooth_basis, getcoef functions
        CAfourierb = create_fourier_basis([0, 1], CAnbasis, 1)  # create a functional data basis
        CAfourierfd = smooth_basis(electimescaled, CAlogall, CAfourierb)  # creat functional data object
        CAfourierfdcoef = getcoef(CAfourierfd)  # get coefficients
        
        # 2.2-------- forecasting by using FAR----------------------
        CAchat_FAR = np.zeros((totalK + 1, 1))  # 12x1
        CAsigma2hat_FAR = np.zeros((totalK + 1, 1))  # 12x1
        CAcepthat_FAR = np.zeros((CAnbasis, 1))  # 23 by 1
        
        # You'll need to implement LFARmle_cept function
        CAcepthat_FAR, CAchat_FAR, CAsigma2hat_FAR = LFARmle_cept(CAfourierfdcoef, step)
        
        # Forecasting(one step)
        CAcoef_forcFAR = np.zeros((CAnbasis, 1))
        
        CAcoef_forcFAR[0, 0] = CAcepthat_FAR[0, 0] + CAchat_FAR[0, 0] * CAfourierfdcoef[0, -1]
        
        for j in range(1, totalK + 1):
            CAcoef_forcFAR[2*j - 1, 0] = (CAcepthat_FAR[2*j - 1, 0] + 
                                        1/np.sqrt(2) * CAchat_FAR[j, 0] * CAfourierfdcoef[2*j - 1, -1])
            CAcoef_forcFAR[2*j, 0] = (CAcepthat_FAR[2*j, 0] + 
                                    1/np.sqrt(2) * CAchat_FAR[j, 0] * CAfourierfdcoef[2*j, -1])
        
        # You'll need to implement fd and eval_fd functions
        CAfd_forcFAR_1315 = fd(CAcoef_forcFAR, CAfourierb)  # forecasted curve
        CAfd_forcvalFAR_1315 = eval_fd(electimescaled, CAfd_forcFAR_1315)  # Mx1 discrete forecasted values
        
        # 2.3 warp back
        gam = KM_gam[ind - (train_sample + step) + step + wz - 1, :]  # -1 for 0-based indexing
        m = M
        x = np.arange(m) / (m - 1)  # uniform distribution points on (0,1)
        gamI = np.zeros((1, m))
        
        # interpolated values
        gamI[0, :] = interp1d(gam[0, :], x, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')(x)
        
        if np.isnan(gamI[0, m-1]):
            gamI[0, m-1] = 1
        else:
            for j in range(m):
                gamI[0, j] = gamI[0, j] / gamI[0, m-1]
        
        KM_gamI = gamI  # the inverse of the warping function matrix gam
        
        CAfd_o_forcvalFAR_1315_7 = interp1d(t, CAfd_forcvalFAR_1315, 
                                        kind='linear', 
                                        bounds_error=False, 
                                        fill_value='extrapolate')(
            (t[-1] - t[0]) * KM_gamI[0, :] + t[0]
        )
        
        CAfd_o_forcvalFAR1315.append(CAfd_o_forcvalFAR_1315_7)

    # Convert to numpy array
    CAfd_o_forcvalFAR1315 = np.array(CAfd_o_forcvalFAR1315).T

    # ------error RMSE between f and forcvalFAR----------- 
    t_index = np.arange(train_sample + step, N)  # t for forecasted prices
    CAfd_forcvalFAR = CAfd_o_forcvalFAR1315  # forecasted price matrix

    # Ensure dimensions match
    if CAfd_forcvalFAR.shape[1] != len(t_index):
        CAfd_forcvalFAR = CAfd_forcvalFAR[:, :len(t_index)]

    err_forcvalWFAR = CAfd_forcvalFAR - f[:, t_index]
    RMSEforcvalWFAR_KM = np.sqrt(np.mean(err_forcvalWFAR**2, axis=1))  # Root Mean Square Error
    eWFAR = err_forcvalWFAR.flatten()
    eWFARvec = np.mean(err_forcvalWFAR, axis=0)  # vectorized forecasted errors