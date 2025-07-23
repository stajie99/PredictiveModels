%% RMSE of WFAR, FAR, VAR, AR, AR*, SAR, ARX and ARX* models 
% Input: electricity price data of Nord Pool market in 2013/01/01-2017/12/31
% Output: WFAR_NP.mat including
%              1. RMSE_NP:  RMSE matrix (M by 8, M=24 i.e. hourly) of the above 8 models
%              2. DM: the DM statistics           
%              3. DMvec: the multivariate DM statistics 

% References:
% Uniejewski, B., Weron, R. and Ziel, F. (2018). Variance stabilizing transformations for electricity spot price forecasting, IEEE Transactions on Power Systems 33(2): 2219�C2229.
% Ziel F. and Weron R. (2018). Day-ahead electricity price forecasting with high dimensional structures: Univariate vs. multivariate models, Energy Economics 70: 396�C420.

% Creat by Jiejie Zhang, last modified on 2018.10.03

addpath(genpath('fdaM'))  
addpath(genpath('NPdata'))
%% input consolidated price data
load('electricity_prices.mat')
%% forecasting step, i.e. 1-day ahead
step=1;
%% take log10(.+1) to avoid the impact of extreme spikes and deal with 0 prices
fwhole=log10(fwhole+1);
%% no. of observations per day
M=24; 
flog = fwhole;
f = reshape(flog, M, []); % reshape vector to matrix
N    = size(flog, 1)/M; 
train_sample = 300;
wz   = 30;%180;%60;%   % rolling window size
%% prepare prices and dates for time warping
f_wz=f(:,train_sample-wz+1:train_sample);
N_forecast = N-train_sample-step+1;
[firstday_inwindow_dayno, firstday_inwindow_dayofweek] = weekday(datenum(fdatewhole(train_sample-wz+1),'dd/mm/yyyy'))
%% --------------------------time warping for fixed warping functions ------------------
%% default values for warping parameters
lambda = 0;
option_parallel = 0;
option_closepool = 0;
option_smooth = 0;
option_sparam = 25;
option_showplot = 0;
t    = [1:1:M]'; 
%% do time warping
%% gam is daily warping function
[~,~,~,~,~,gam,~,~] = time_warping1(f_wz,t,lambda,option_parallel,option_closepool,option_smooth,option_sparam,option_showplot); % gam is M by wz.
[KM_gam, ~] = KarcherMeansof_warpingfunctions(gam', firstday_inwindow_dayno); % KM_gam is 7 by M, estimated fixed warping functions

KM_gam = repmat(KM_gam, floor((N_forecast+wz)/7), 1); %repeat copies of a matrix into a ** by 1 block arrangement
KM_gam=[KM_gam; KM_gam(1:mod((N_forecast+wz),7),:)];  % KM_gam is N_forecast+wz by 1 now, containing the estimated fixed warping functions.
%%
% prepare for scaled time ticks, no. of basis, and totalK
        electime=1:M;
        electimescaled=zeros(M,1);
        for i=1:M
            electimescaled(i)=electime(i)/M;
        end
        %Create the fd objects by fourier transformation        
        CAnbasis = 23;
        totalK=(CAnbasis-1)/2;
%%  ------------------------------------WFAR ----------------------------------
CAfd_o_forcvalFAR1315 = [];
for ind=(train_sample+step):(N)
    %%%%%% fix rolling window wz %%%%%%%
        %%  1.time warping
        f_wz = f(:,ind-step-wz+1:ind-step); % prices in the rolling window
        fn=zeros(M,wz); % for warped prices
        for i=1:wz
            i;
            fn(:,i) = interp1(t, f_wz(:,i), (t(end)-t(1)).*KM_gam(ind-(train_sample+step)+i, :) + t(1))'; % time warping with fixed warping function
        end
       
        %% 2.FAR  
        % 2.1.--------------------------- smoothing ----------------------
        CAlogall=fn; % 24*wz        
        CAfourierb = create_fourier_basis([0,1],CAnbasis,1); % creat a functional data basis
        CAfourierfd=smooth_basis(electimescaled,CAlogall,CAfourierb);%23 * 300+ind-step
        % creat a functional data object, consisiting of a basis for expanding and a set of coefficients 
        CAfourierfdcoef=getcoef(CAfourierfd);%23x(300+ind-step)        
        
        % 2.2-------- forecasting by using FAR----------------------
        CAchat_FAR=zeros(totalK+1,1);%12x1
        CAsigma2hat_FAR=zeros(totalK+1,1);%12x1
        CAcepthat_FAR=zeros(CAnbasis,1);%23 by 1
        [CAcepthat_FAR,CAchat_FAR,CAsigma2hat_FAR]=LFARmle_cept(CAfourierfdcoef,step); % estimator from 300+ind-step
        % Forecasting(one step)
        CAcoef_forcFAR=zeros(CAnbasis,1);

        CAcoef_forcFAR(1,1)=CAcepthat_FAR(1,1)+CAchat_FAR(1,1)*CAfourierfdcoef(1,end); % forecast by one step
        for j=1:totalK
            CAcoef_forcFAR(2*j,1)=CAcepthat_FAR(2*j,1)+1/sqrt(2)*CAchat_FAR(j+1,1)*CAfourierfdcoef(2*j,end);
            CAcoef_forcFAR(2*j+1,1)=CAcepthat_FAR(2*j+1,1)+1/sqrt(2)*CAchat_FAR(j+1,1)*CAfourierfdcoef(2*j+1,end);
        end
        CAfd_forcFAR_1315=fd(CAcoef_forcFAR,CAfourierb);  % forecasted curve
        CAfd_forcvalFAR_1315=eval_fd(electimescaled,CAfd_forcFAR_1315);%24x1 discrete forecasted values? for RMSE
        %% 2.3 warp back
         % the inverse of the warping function matrix gam
        gam=KM_gam(ind-(train_sample+step)+step+wz,:);
        m = M;
        x = (0:m-1)/(m-1); % N uniform distribution points on (0,1)
        gamI = zeros(1,m); %train_sample*24
                gamI(1,:) = interp1(gam(1,:),x,x); % interpolated values of x(gam is sample points, x are values)
                                 %at specific query points using linear interpolation
                if isnan(gamI(1,m))
                gamI(1,m) = 1;
                else
                    for j=1:m
                    gamI(1,j) = gamI(1,j)./gamI(1,m);
                    end
                end
        KM_gamI=gamI; % the inverse of the warping function matrix gam, length(CAfd_forcvalFAR300) by 24    
        CAfd_o_forcvalFAR_1315_7 = interp1(t, CAfd_forcvalFAR_1315, (t(end)-t(1)).*KM_gamI + t(1))'; 
        CAfd_o_forcvalFAR1315 = [CAfd_o_forcvalFAR1315 CAfd_o_forcvalFAR_1315_7];
       
end
%%%%%------error RMSE between f and forcvalFAR-----------   %%%%%   
    t_index=(train_sample+step):(N);     % t for forcasted prices
    CAfd_forcvalFAR=CAfd_o_forcvalFAR1315;%(:,t_index); % forcasted price matrix
    err_forcvalWFAR = CAfd_forcvalFAR-f(:,t_index);
    RMSEforcvalWFAR_KM=sqrt(mean(err_forcvalWFAR.^2, 2)); % Root Mean Square Error
    eWFAR = err_forcvalWFAR(:) 
    eWFARvec = mean(err_forcvalWFAR, 1); % err_forcvalFAR(:) % vectorized forecasted errors

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%    FAR model

CAfourierb=create_fourier_basis([0,1],CAnbasis,1); % creat a functional data basis
CAfd_forcvalFAR300=[];
for ind=train_sample+step:N % step:43 step:277  no. of curves to forecast = 277-step+1
    %% --------------------------------Smoothing-----------------------------------
    %%%%%% fix rolling window wz %%%%%%%
    ind;
    f_wz=f(:,ind-step-wz+1:ind-step);
    CAlogall=f_wz; 
    
    % creat a fd (functional data) object, consisiting of a basis for expanding and a set of coefficients 
    CAfourierfd=smooth_basis(electimescaled,CAlogall,CAfourierb);%23 * 300+ind-step
    CAfourierfdcoef=getcoef(CAfourierfd);%23x(300+ind-step)
    %% ------------------------------forecasting by using FAR----------------------
    CAchat_FAR=zeros(totalK+1,1);%12x1
    CAsigma2hat_FAR=zeros(totalK+1,1);%12x1
    CAcepthat_FAR=zeros(CAnbasis,1);%23 by 1  
    [CAcepthat_FAR,CAchat_FAR,CAsigma2hat_FAR]=LFARmle_cept(CAfourierfdcoef,step); % estimator from 300+ind-step
    % Forecasting(one step)
    CAcoef_forcFAR=zeros(CAnbasis,1);   
    CAcoef_forcFAR(1,1)=CAcepthat_FAR(1,1)+CAchat_FAR(1,1)*CAfourierfdcoef(1,end); % forecast by one step
    for j=1:totalK
        CAcoef_forcFAR(2*j,1)=CAcepthat_FAR(2*j,1)+1/sqrt(2)*CAchat_FAR(j+1,1)*CAfourierfdcoef(2*j,end);
        CAcoef_forcFAR(2*j+1,1)=CAcepthat_FAR(2*j+1,1)+1/sqrt(2)*CAchat_FAR(j+1,1)*CAfourierfdcoef(2*j+1,end);
    end  
    CAfd_forcFAR_300=fd(CAcoef_forcFAR,CAfourierb);  % forecasted curves
    CAfd_forcvalFAR_300=eval_fd(electimescaled,CAfd_forcFAR_300);%24x1 discrete forecasted values 
    CAfd_forcvalFAR300 = [CAfd_forcvalFAR300 CAfd_forcvalFAR_300];%% 24*(ind-step+1) forecasted matrix
end
    t_index=(train_sample+step):N;
    err_forcvalFAR=CAfd_forcvalFAR300-f(:,t_index);%24x(277-step+1)ar
    eFAR = err_forcvalFAR(:);
    eFARvec = mean(err_forcvalFAR, 1); %err_forcvalFAR(:);
    RMSEforcvalFARNP=sqrt(mean(err_forcvalFAR.^2, 2));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%    VAR model
forecVARlog=[];
for ind=train_sample+step:N   
    %%%%%% fix rolling window wz %%%%%%%
    ind;
    CAlogall=f(:,ind-step-wz+1:ind-step);% 24 by wz 
    Xm=zeros(wz-step,M+1);  
    forecVARlog_d=zeros(M,1);
    for j=1:M
        yv=CAlogall(j,1+step:end)';
        Xm(:,1)=ones(wz-step,1);
        Xm(:,2:M+1)=CAlogall(:,1:end-step)';        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%2x1        
        forecVARlog_d(j,1)=[1 CAlogall(:,end)']*CAlogpricebeta;
    end
    forecVARlog=[forecVARlog forecVARlog_d];
end
t_index=(train_sample+step):N;
err_forcVAR=forecVARlog-f(:,t_index);%24x43
eVAR =err_forcVAR(:);
 eVARvec =mean(err_forcVAR, 1); %err_forcVAR(:);
RMSEforcvalVAR=sqrt(mean(err_forcVAR.^2,2)); %24x1

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%    AR model
forecARlog=[];
for ind=train_sample+step:N 
    ind;
    %  fixed rolling window
    CAlogall=f(:,ind-step-wz+1:ind-step); 
    %% prepare x and y
    Xm=zeros(wz-step,2); 
    forecARXlog_d=zeros(M,1);
    for j=1:M
        j;
        yv=CAlogall(j,1+step:end)';
        Xm(:,1)=ones(wz-step,1);
        Xm(:,2)=CAlogall(j,1:end-step)';% lag 1    
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%9x1
        forecARlog_d(j,1)=[1 CAlogall(j,end)]*CAlogpricebeta;
    end
    forecARlog=[forecARlog forecARlog_d];
end
t_forcindex=(train_sample+step):N;
err_forcAR=forecARlog-f(:,t_forcindex);%24x43
eAR = err_forcAR(:);
eARvec= mean(err_forcAR, 1); %err_forcARsea(:);
RMSEforcvalAR=sqrt(mean(err_forcAR.^2,2)); %24x1

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%    AR* model, the last known price of the previous day
forecARlog=[];
for ind=train_sample+step:N 
    ind
    %  fixed rolling window
    CAlogall=f(:,ind-step-wz+1:ind-step); 
    %% prepare x and y
    Xm=zeros(wz-step,2); 
    forecARXlog_d=zeros(M,1);
    for j=1:M
        j;
        yv=CAlogall(j,1+step:end)';
        Xm(:,1)=ones(wz-step,1);
        Xm(:,2)=CAlogall(M,1:end-step)';% lag 1    
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%9x1
        forecARlog_d(j,1)=[1 CAlogall(M,end)]*CAlogpricebeta;
    end
    forecARlog=[forecARlog forecARlog_d];
end
t_forcindex=(train_sample+step):N;
err_forcARstar=forecARlog-f(:,t_forcindex);%24x43
eARstar = err_forcARstar(:);
eARstarvec = mean(err_forcARstar, 1); %err_forcARsea(:);
RMSEforcvalARstar=sqrt(mean(err_forcARstar.^2,2)); %24x1

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%    SAR model
forecARsealog=[];
for ind=train_sample+step:N 
    ind;
    %  fixed rolling window
    CAlogall=f(:,ind-step-wz+1:ind-step); 
    %% prepare x and y
    length = wz-step-7+1;
    Xm=zeros(length,4); 
    forecARXlog_d=zeros(M,1);
    for j=1:M
        j;
        yv=CAlogall(j,7+step:end)';
        Xm(:,1)=ones(length,1);
        Xm(:,2)=CAlogall(j,7:end-step)';% lag 1
        Xm(:,3)=CAlogall(j,6:end-step-1)';% lag 2
        Xm(:,4)=CAlogall(j,1:end-step-6)';% lag 7      
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%9x1
        forecARsealog_d(j,1)=[1 CAlogall(j,end) CAlogall(j,end-1) CAlogall(j,end-6)]*CAlogpricebeta;
    end
    forecARsealog=[forecARsealog forecARsealog_d];
end
t_forcindex=(train_sample+step):N;
err_forcARsea=forecARsealog-f(:,t_forcindex);%24x43
 eSAR = err_forcARsea(:);
 eSARvec = mean(err_forcARsea, 1); %err_forcARsea(:);
RMSEforcvalSAR=sqrt(mean(err_forcARsea.^2,2)); %24x1

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%    ARX model
load('electricity_volumes.mat')
%% --1. build a dummy matrix for Sun, Mon and Sat------------------------
[first_indata_dayno, first_indata_dayofweek] = weekday(datenum(fdatewhole(1),'dd/mm/yyyy'))
DumMat = [0 0 1; 1 0 0; 0 0 0;0 0 0;0 0 0;0 0 0; 0 1 0]; %for Sun, Mon and Sat
DumMat = DumMat([first_indata_dayno:7 1:first_indata_dayno-1],:);
DumMat49=repmat(DumMat,floor(N/7),1);
DumMat49=[DumMat49; DumMat49(1:mod(N,7),:)]; 
%% -- 2. prepare the electricity load, the prices, and minimum of prices------------------------------------
CAforec_load=reshape(fvolumewhole,M,[]);%The forecasted load
CAforec_actload=CAforec_load;%The actual  load for out-of-sample forecast
CAforec_logload=log10(CAforec_actload);%24x343 for the common log of forecasted load
% % 1 March 2000 (761th) is Wednesday. 30 April 2000 is Sunday.
forecARXlog=[];
for ind=train_sample+step:N 
    ind;
    %  fixed rolling window
    %% 3. cut the needed the electricity load, the prices, and minimum of prices
    CAlogall =               f(:,ind-step-wz+1:ind-step); 
    mp       = min(CAlogall);% col.mean
    mCAlog   = mp';      %The min log price for previous day 
    CAlogld  = CAforec_logload(:,ind-step-wz+1:end);
    CADumMat =          DumMat49(ind-step-wz+1:end,:); % till the end for forecast
    %% 4. prepare x and y
    length = wz-step-7+1;
    Xm=zeros(length,9); 
    Xm(:,7:9)=CADumMat(7+step:wz,:); % Mon, Sat, Sun corresponds to the day of week of dependent variable Y:  468 onwards
    forecARXlog_d=zeros(M,1);
    for j=1:M
        j;
        yv     = CAlogall(j,7+step:end)';
        Xm(:,1)= ones(length,1);
        Xm(:,2)= CAlogall(j,7:end-step)';% lag 1
        Xm(:,3)= CAlogall(j,6:end-step-1)';% lag 2
        Xm(:,4)= CAlogall(j,1:end-step-6)';% lag 7
        Xm(:,5)= mCAlog(7:end-step);% lag 1
        Xm(:,6)= CAlogld(j,7+step:wz)';% now lag 1. Weron:lag 0/ load forecast
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%9x1
        
        forecARXlog_d(j,1)=[1 CAlogall(j,end) CAlogall(j,end-1) CAlogall(j,end-6) mCAlog(end) CAlogld(j,wz+step) CADumMat(wz+step,:)]*CAlogpricebeta;
    end
    forecARXlog=[forecARXlog forecARXlog_d];
end
t_forcindex=(train_sample+step):N;
err_forcARX=forecARXlog-f(:,t_forcindex);%24x43
 eARX = err_forcARX(:);
 eARXvec = mean(err_forcARX, 1);%err_forcARX(:);
RMSEforcvalARX=sqrt(mean(err_forcARX.^2,2)); %24x1

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%    ARX* model
% % 1 March 2000 (761th) is Wednesday. 30 April 2000 is Sunday.
forecARXlog=[];
for ind=train_sample+step:N 
    ind;
    %  fixed rolling window
    %% 3. cut the needed the electricity load, the prices, and minimum of prices
    CAlogall =  f(:,ind-step-wz+1:ind-step); 
    mCAlog   = CAlogall(end, :)';      %The last known log price for previous day 
    CAlogld  = CAforec_logload(:,ind-step-wz+1:end);
    CADumMat =          DumMat49(ind-step-wz+1:end,:); % till the end for forecast
    %% 4. prepare x and y
    length = wz-step-7+1;
    Xm=zeros(length,9); 
    Xm(:,7:9)=CADumMat(7+step:wz,:); % Mon, Sat, Sun corresponds to the day of week of dependent variable Y:  468 onwards
    forecARXlog_d=zeros(M,1);
    for j=1:(M-1)
        j;
        yv     = CAlogall(j,7+step:end)';
        Xm(:,1)= ones(length,1);
        Xm(:,2)= CAlogall(j,7:end-step)';% lag 1
        Xm(:,3)= CAlogall(j,6:end-step-1)';% lag 2
        Xm(:,4)= CAlogall(j,1:end-step-6)';% lag 7
        Xm(:,5)= mCAlog(7:end-step);% lag 1
        Xm(:,6)= CAlogld(j,7+step:wz)';% now lag 1. Weron:lag 0/ load forecast
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%9x1
        
        forecARXlog_d(j,1)=[1 CAlogall(j,end) CAlogall(j,end-1) CAlogall(j,end-6) mCAlog(end) CAlogld(j,wz+step) CADumMat(wz+step,:)]*CAlogpricebeta;
    end
    for j=M
        j;
        yv     = CAlogall(j,7+step:end)';
        Xm(:,1)= ones(length,1);
        Xm(:,2)= CAlogall(j,7:end-step)';% lag 1
        Xm(:,3)= CAlogall(j,6:end-step-1)';% lag 2
        Xm(:,4)= CAlogall(j,1:end-step-6)';% lag 7
        Xm(:,5)= mCAlog(7:end-step);% lag 1
        Xm(:,6)= CAlogld(j,7+step:wz)';% now lag 1. Weron:lag 0/ load forecast
        Xm = Xm(:, [1 2 3 4 6 7 8 9]);
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%9x1
        
        forecARXlog_d(j,1)=[1 CAlogall(j,end) CAlogall(j,end-1) CAlogall(j,end-6) CAlogld(j,wz+step) CADumMat(wz+step,:)]*CAlogpricebeta;
    end
    forecARXlog=[forecARXlog forecARXlog_d];
end
t_forcindex=(train_sample+step):N;
err_forcARX=forecARXlog-f(:,t_forcindex);%24x43
 eARXstar = err_forcARX(:);  % vectorize by column
 eARXstarvec = mean(err_forcARX, 1); %err_forcARX(:);  % vectorize by column
RMSEforcvalARXstar=sqrt(mean(err_forcARX.^2,2)); %24x1

%% combine RMSE matrix of WFAR model and alternative models
RMSE_NP = [RMSEforcvalWFAR_KM RMSEforcvalFARNP RMSEforcvalVAR RMSEforcvalAR RMSEforcvalARstar RMSEforcvalSAR RMSEforcvalARX RMSEforcvalARXstar];

%% Diebold Mariano test
DM= round([dmtestpvl(eFAR, eWFAR) dmtestpvl(eVAR, eWFAR) dmtestpvl(eAR, eWFAR) dmtestpvl(eARstar, eWFAR) dmtestpvl(eSAR, eWFAR) dmtestpvl(eARX, eWFAR) dmtestpvl(eARXstar, eWFAR)], 3)
DMvec= round([dmtestpvl(eFARvec', eWFARvec') dmtestpvl(eVARvec', eWFARvec') dmtestpvl(eARvec', eWFARvec') dmtestpvl(eARstarvec', eWFARvec') dmtestpvl(eSARvec', eWFARvec') dmtestpvl(eARXvec', eWFARvec') dmtestpvl(eARXstarvec', eWFARvec')], 3)
save('WFAR_NP.mat', 'RMSE_NP', 'DM', 'DMvec')
