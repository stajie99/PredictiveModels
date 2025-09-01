 function [KMgamma, gamI_cate] = KarcherMeansof_warpingfunctions(gam, daynumber_1stday)
%%% Actually it is not necessary to identify what day it is for the first,
%%% second,... day. Categorize as n*k+1, n*k+2,..., n*k+7 are important
%%%%%%%%%%%%%%%%%%%%  Karcher Mean of Warping functions, categorized by Mon, Tue till Sun.
% set the Rainbow color matrix and plot the original functions
[N,T] = size(gam); % N days
dayno = [daynumber_1stday:7 1:daynumber_1stday-1]';
Dayno = repmat(dayno, floor(N/7), 1);
Dayno = [Dayno; Dayno(1:mod(N,7),:)];
gam = [Dayno gam];
gamMon = [];
gamTue = [];
gamWed = [];
gamThu = [];
gamFri = [];
gamSat = [];
gamSun = [];
for i =1:N   
    if gam(i,1) == 1     %% '1' stand for 'Sunday','2'for 'Monday', ...,'7' for 'Satday'.
       gamSun = [gamSun; gam(i,2:end)];
    elseif gam(i,1) == 2
       gamMon = [gamMon; gam(i,2:end)];
    elseif gam(i,1) == 3
       gamTue = [gamTue; gam(i,2:end)];
    elseif gam(i,1) == 4
       gamWed = [gamWed; gam(i,2:end)];
    elseif gam(i,1) == 5
       gamThu = [gamThu; gam(i,2:end)];
    elseif gam(i,1) == 6
       gamFri = [gamFri; gam(i,2:end)];
    else
       gamSat = [gamSat; gam(i,2:end)];
    end
end

gamI_cate = zeros(7,T);
gamI_cate(1,:) = Jiejie_SqrtMean(gamSun);
gamI_cate(2,:) = Jiejie_SqrtMean(gamMon);
gamI_cate(3,:) = Jiejie_SqrtMean(gamTue);
gamI_cate(4,:) = Jiejie_SqrtMean(gamWed); % Karcher Mean of warping functions
gamI_cate(5,:) = Jiejie_SqrtMean(gamThu);
gamI_cate(6,:) = Jiejie_SqrtMean(gamFri);
gamI_cate(7,:) = Jiejie_SqrtMean(gamSat);

KMgamma=[gamI_cate(daynumber_1stday:7,:); gamI_cate(1:daynumber_1stday-1,:)];


