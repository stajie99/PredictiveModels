function gamKM = KarcherMean_warpingfunctions(gam)
%%%%%%%%%%%%%%%%%%%%  Karcher Mean of Warping functions, categorized by Mon, Tue till Sun.
% set the Rainbow color matrix and plot the original functions
gamMon = [];
gamTue = [];
gamWed = [];
gamThu = [];
gamFri = [];
gamSat = [];
gamSun = [];
for i =1:N   
    if mod(i,7) == 1
      gamMon(ceil(i/7),:) = gam(i,:);
    elseif mod(i,7) == 2
      gamTue(ceil(i/7),:) = gam(i,:);
    elseif mod(i,7) == 3
      gamWed(ceil(i/7),:) = gam(i,:);
    elseif mod(i,7) == 4
      gamThu(ceil(i/7),:) = gam(i,:);
    elseif mod(i,7) == 5
      gamFri(ceil(i/7),:) = gam(i,:);
    elseif mod(i,7) == 6
      gamSat(ceil(i/7),:) = gam(i,:);
    else
      gamSun(ceil(i/7),:) = gam(i,:);
    end
end
gamI_cate = zeros(7,M);
gamI_cate(1,:) = Jiejie_SqrtMean(gamMon);
gamI_cate(2,:) = Jiejie_SqrtMean(gamTue);
gamI_cate(3,:) = Jiejie_SqrtMean(gamWed); % Karcher Mean of warping functions
gamI_cate(4,:) = Jiejie_SqrtMean(gamThu);
gamI_cate(5,:) = Jiejie_SqrtMean(gamFri);
gamI_cate(6,:) = Jiejie_SqrtMean(gamSat);
gamI_cate(7,:) = Jiejie_SqrtMean(gamSun);
gamI_cate=gamI_cate';