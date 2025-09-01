%% WFAR plots for illustration

% 1. plot of daily log smoothed price curves--surface
% 2. plot of average curves on Mon--Sun of both NP and CA
% 3. plot of min, max and mean curve of both NP and CA
% 4. plots of time series of 17:00 and 18:00, ACF & CCF
% 5. plot of Karcher mean & the center
% 6. plot of Karcher Mean of Warping functions, categorized by Mon, Tue till Sun for both NP market and CA market

% Creat by Jiejie Zhang, last modified on 2018.10.03

%% NP
addpath(genpath('fdaM'))  
addpath(genpath('NPdata'))
year = [2013:2017];
M=24; % hourly points
load('electricity_prices.mat') % the original data of NP
step=1;
% take log10(.+1) 
fwhole=log10(fwhole+1); 
flog = fwhole;
N    = size(flog, 1)/M; %
fmatr = reshape(flog, M, N);

%% Plotting
[yy xx] = meshgrid(1:M, 1:0.5:N); 
y_tick  = 2:2:M;
y_label = {'2', '4', '6', '8', '10','12', '14', '16', '18', '20', '22', '24'}; 
y_lim   = [1, M];
tck2014 = 1+size(fdate2013,2);
tck2015 = 1+size(fdate2013,2)+size(fdate2014,2);
tck2016 = 1+size(fdate2013,2)+size(fdate2014,2)+size(fdate2015,2);
tck2017 = 1+size(fdate2013,2)+size(fdate2014,2)+size(fdate2015,2)+size(fdate2016,2);
x_tick  = 2*[1 tck2014 tck2015 tck2016 tck2017 N]-1;
x_label = {'2013.01.01','2014.01.01','2015.01.01','2016.01.01','2017.01.01', '2017.12.31'};
x_lim   = [1, 2*N-1];

%% 3d surface
figure(1);clf;
surf(interp2(fmatr,xx,yy)','EdgeColor','none')
colormap Jet 
colorbar
grid on  
set(gca,'XTick',x_tick,'XTickLabel',x_label,'XLim', x_lim)
set(gca,'YDir','Reverse','YTick',y_tick,'YTickLabel',y_label, 'YLim', y_lim)
ylabel('Time','FontSize',16,'FontWeight','bold');
xlabel('Date','FontSize',16,'FontWeight','bold');
zlabel('Log Price','FontSize',16,'FontWeight','bold');

%% CA %% 
addpath(genpath('fdaM'))  
addpath(genpath('CAdata'))
load('CA_hourly.dat')
step=1;
M=24; % hourly points
Total_obs=1037; 
start_point=461; 
N=Total_obs-start_point+1; %577 
flog=log10(CA_hourly(M*(start_point-1)+1:end,3)+1); % 577*24 by 1
fmatr = reshape(flog, M, []);
%% Plotting
[yy xx] = meshgrid(1:M, 1:0.5:N); 
y_tick  = 2:2:M;
y_label = {'2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24'}; 
y_lim   = [1, M];
x_tick  = 2*[1,93,185,276,367,459,551]-1;
x_label = {'1999.07.05','1999.10.05','2000.01.05','2000.04.05','2000.07.05','2000.10.05','2001.01.05'};
x_lim   = 2*[1, N]-1;
%% 3d surface
figure(1);clf;
surf(interp2(fmatr,xx,yy)','EdgeColor','none')
colormap Jet 
colorbar
grid on  
set(gca,'XTick',x_tick,'XTickLabel',x_label,'XLim', x_lim)
set(gca,'YDir','Reverse','YTick',y_tick,'YTickLabel',y_label, 'YLim', y_lim)
ylabel('Time','FontSize',16,'FontWeight','bold');
xlabel('Date','FontSize',16,'FontWeight','bold');
zlabel('Log Price','FontSize',16,'FontWeight','bold');
%%%%%%%%%%%%%%%%%%%%%%%%
% Average warped and unwarped curves
%% 1.NP
addpath(genpath('fdaM'))  
addpath(genpath('NPdata'))
year = [2013:2017];
M=24;
load('electricity_prices.mat')
step=1;
% take log10(.+1) 
fwhole=log10(fwhole+1); 
flog = fwhole;
N    = size(flog, 1)/M; %
fmatr = reshape(flog, M, N);
[first_dayno, first_dayofweek] = weekday(datenum(fdatewhole(1),'mm/dd/yyyy'))

t=[1:1:M]'; 
f0 = fmatr;
fMon=[];
fTue=[];
fWed=[];
fThu=[];
fFri=[];
fSat=[];
fSun=[];   
for i =1:N    
    if mod(i,7) == 1
      fTue(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 2
      fWed(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 3
      fThu(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 4
      fFri(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 5
      fSat(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 6
      fSun(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 0
      fMon(:,ceil(i/7)) = f0(:,i);
    end
end
desc_logf_cate=zeros(M,7);
desc_logf_cate(:,1) = mean(fMon,2);
desc_logf_cate(:,2) = mean(fTue,2);
desc_logf_cate(:,3) = mean(fWed,2);
desc_logf_cate(:,4) = mean(fThu,2);
desc_logf_cate(:,5) = mean(fFri,2);
desc_logf_cate(:,6) = mean(fSat,2);
desc_logf_cate(:,7) = mean(fSun,2);

x = [1:1:M];
x_tklabel = {' ', '2', ' ', '4', ' ', '6', ' ', '8', ' ', '10', ' ','12', ' ', '14', ' ', '16', ' ', '18', ' ', '20', ' ', '22', ' ', '24'};
ylim = [min(min(desc_logf_cate))-0.02, max(max(desc_logf_cate))+0.02];
h=[];
linestyle = {'-', ':', '-.', '--', '--', ':', '--'};
marker   = {'none', 'none', 'none', '*', 'o', 's', '+'};
figure(3);clf;
for i=1:7
    h(i)=plot(t,desc_logf_cate(:,i), 'linewidth',1.1, 'LineStyle', linestyle{i}, 'Marker', marker{i}, 'Color', 'k'); hold on;
    [yvalley, indv]=min(desc_logf_cate(:,i));
    [ypeak, indp]=findpeaks(desc_logf_cate(:,i));
    line([indv, indv], [ylim(1), yvalley],  'LineStyle', '--', 'Color', 'k');hold on;
    if i==6
        line([indp(1), indp(1)], [ylim(1), ypeak(1)],  'LineStyle', '--', 'Color', 'k');hold on;
    elseif size(ypeak,1)==1
       line([indp(1), indp(1)], [ylim(1), ypeak(1)],  'LineStyle', '--', 'Color', 'k');hold on;
    elseif size(ypeak,1)==2
       line([indp(1), indp(1)], [ylim(1), ypeak(1)],  'LineStyle', '--', 'Color', 'k');hold on;
       line([indp(2), indp(2)], [ylim(1), ypeak(2)],    'LineStyle', '--', 'Color', 'k');hold on;
    end
    
end
set(gca,'XTick',x, 'XTickLabel',x_tklabel, 'XLim', [1,M], 'YLim', ylim);
xlabel('Time','FontSize',14,'FontWeight','bold');
ylabel('Log Price','FontSize',14,'FontWeight','bold');
legend([h(1),h(2),h(3),h(4),h(5),h(6),h(7)],'Mon','Tue','Wed','Thu','Fri','Sat','Sun','location','northeast');

% 2. NP warped data
addpath(genpath('fdaM'))  
addpath(genpath('NPdata'))
load('electricity_prices.mat')
load('LogAlignedNPwhole.mat') % the results of doing time warping for all the Nord Pool data--in sample analysis
% take log10(.+1) 
M=24;
N    = size(fn, 2); %
fmatr = fn;
[first_dayno, first_dayofweek] = weekday(datenum(fdatewhole(1),'mm/dd/yyyy'))

t=[1:1:M]'; 
f0 = fmatr;
fMon=[];
fTue=[];
fWed=[];
fThu=[];
fFri=[];
fSat=[];
fSun=[];   
for i =1:N    
    if mod(i,7) == 1
      fTue(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 2
      fWed(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 3
      fThu(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 4
      fFri(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 5
      fSat(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 6
      fSun(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 0
      fMon(:,ceil(i/7)) = f0(:,i);
    end
end
desc_logf_cate=zeros(M,7);
desc_logf_cate(:,1) = mean(fMon,2);
desc_logf_cate(:,2) = mean(fTue,2);
desc_logf_cate(:,3) = mean(fWed,2);
desc_logf_cate(:,4) = mean(fThu,2);
desc_logf_cate(:,5) = mean(fFri,2);
desc_logf_cate(:,6) = mean(fSat,2);
desc_logf_cate(:,7) = mean(fSun,2);

x = [1:1:24];
x_tklabel = {' ', '2', ' ', '4', ' ', '6', ' ', '8', ' ', '10', ' ', '12', ' ', '14', ' ', '16', ' ', '18', ' ', '20', ' ', '22', ' ', '24'};
ylim = [min(min(desc_logf_cate))-0.02, max(max(desc_logf_cate))+0.02];
h=[];
linestyle = {'-', ':', '-.', '--', '--', ':', '--'};
marker   = {'none', 'none', 'none', '*', 'o', 's', '+'};
figure(3);clf;
for i=1:7
    h(i)=plot(t,desc_logf_cate(:,i), 'linewidth',1.1, 'LineStyle', linestyle{i}, 'Marker', marker{i}, 'Color', 'k');hold on;
     [yvalley, indv]=min(desc_logf_cate(:,i));
    [ypeak, indp]=findpeaks(desc_logf_cate(:,i));
    line([indv, indv], [ylim(1), yvalley], 'LineStyle', '--', 'Color', 'k');hold on;
    if size(ypeak,1)==1
       line([indp(1), indp(1)], [ylim(1), ypeak(1)], 'LineStyle', '--', 'Color', 'k');hold on;
    elseif size(ypeak,1)==2
       line([indp(1), indp(1)], [ylim(1), ypeak(1)], 'LineStyle', '--', 'Color', 'k');hold on;
       line([indp(2), indp(2)], [ylim(1), ypeak(2)], 'LineStyle', '--', 'Color', 'k');hold on;
    end
end
set(gca,'XTick',x, 'XTickLabel',x_tklabel, 'XLim', [1,24], 'YLim', ylim);
xlabel('Time','FontSize',14,'FontWeight','bold');
ylabel('Log Price','FontSize',14,'FontWeight','bold');
legend([h(1),h(2),h(3),h(4),h(5),h(6),h(7)],'Mon','Tue','Wed','Thu','Fri','Sat','Sun','location','northeast');



%% 3.CA
addpath(genpath('fdaM'))  
addpath(genpath('CAdata'))
load('CA_hourly.dat')
step=1;
M=24; % hourly points
Total_obs=1037;
start_point=461;
N=Total_obs-start_point+1; %577
t=[1:1:M]'; 
train_sample=300;
f0=log10(CA_hourly((24*(start_point-1)+1):end,3)+1); % 577*24 by 1
f0=reshape(f0,M,N);
fMon=[];
fTue=[];
fWed=[];
fThu=[];
fFri=[];
fSat=[];
fSun=[];   
for i =1:N    
    if mod(i,7) == 1
      fMon(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 2
      fTue(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 3
      fWed(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 4
      fThu(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 5
      fFri(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 6
      fSat(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 0
      fSun(:,ceil(i/7)) = f0(:,i);
    end
end
desc_logf_cate=zeros(M,7);
desc_logf_cate(:,1) = mean(fMon,2);
desc_logf_cate(:,2) = mean(fTue,2);
desc_logf_cate(:,3) = mean(fWed,2);
desc_logf_cate(:,4) = mean(fThu,2);
desc_logf_cate(:,5) = mean(fFri,2);
desc_logf_cate(:,6) = mean(fSat,2);
desc_logf_cate(:,7) = mean(fSun,2);
x = [1:1:24];
x_tklabel = {' ', '2', ' ', '4', ' ', '6', ' ', '8', ' ', '10', ' ', '12', ' ', '14', ' ', '16', ' ', '18', ' ', '20', ' ', '22', ' ', '24'};
ylim = [min(min(desc_logf_cate))-0.02, max(max(desc_logf_cate))+0.02];
h = [];
linestyle = {'-', ':', '-.', '--', '--', ':', '--'};
marker   = {'none', 'none', 'none', '*', 'o', 's', '+'};
figure(3);clf;
for i=1:7
    h(i)=plot(t,desc_logf_cate(:,i), 'linewidth',1.1, 'LineStyle', linestyle{i}, 'Marker', marker{i}, 'Color', 'k');hold on;
    [yvalley, indv]=min(desc_logf_cate(:,i));
    [ypeak, indp]=findpeaks(desc_logf_cate(:,i));
    line([indv, indv], [ylim(1), yvalley], 'Linestyle', '--','Color', 'k');hold on;
    if i== 6 | 7
         line([indp(1), indp(1)], [ylim(1), ypeak(1)],'Linestyle', '--','Color', 'k');hold on;
    elseif size(ypeak,1)==1
       line([indp(1), indp(1)], [ylim(1), ypeak(1)],'Linestyle', '--','Color', 'k');hold on;
    elseif size(ypeak,1)==2
       line([indp(1), indp(1)], [ylim(1), ypeak(1)],'Linestyle','--', 'Color', 'k');hold on;
       line([indp(2), indp(2)], [ylim(1), ypeak(2)],'Linestyle', '--','Color', 'k');hold on;
    end
end
set(gca,'XTick',x, 'XTickLabel',x_tklabel, 'XLim', [1,24], 'YLim', ylim);
xlabel('Time','FontSize',14,'FontWeight','bold');
ylabel('Log Price','FontSize',14,'FontWeight','bold');
legend([h(1),h(2),h(3),h(4),h(5),h(6),h(7)],'Mon','Tue','Wed','Thu','Fri','Sat','Sun','location','northeast');
% 4. warped
addpath(genpath('fdaM'))  
addpath(genpath('CAdata'))
load('elecAlignedLog1_577.mat')% the results of doing time warping for all the California data--in sample analysis
step=1;
M=24; % hourly points
N=size(fn, 2); %577
t=[1:1:M]'; 
f0=fn([14:M, 1:13], :); %([12:M,1:11],:);
fMon=[];
fTue=[];
fWed=[];
fThu=[];
fFri=[];
fSat=[];
fSun=[];   
for i =1:N    
    if mod(i,7) == 1
      fMon(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 2
      fTue(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 3
      fWed(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 4
      fThu(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 5
      fFri(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 6
      fSat(:,ceil(i/7)) = f0(:,i);
    elseif mod(i,7) == 0
      fSun(:,ceil(i/7)) = f0(:,i);
    end
end
desc_logf_cate=zeros(M,7);
desc_logf_cate(:,1) = mean(fMon,2);
desc_logf_cate(:,2) = mean(fTue,2);
desc_logf_cate(:,3) = mean(fWed,2);
desc_logf_cate(:,4) = mean(fThu,2);
desc_logf_cate(:,5) = mean(fFri,2);
desc_logf_cate(:,6) = mean(fSat,2);
desc_logf_cate(:,7) = mean(fSun,2);
x = [1:1:24];
x_tklabel = { ' ', '2', ' ', '4', ' ', '6', ' ', '8', ' ', '10', ' ', '12', ' ', '14', ' ', '16', ' ', '18', ' ', '20', ' ', '22', ' ', '24'};
ylim = [min(min(desc_logf_cate))-0.02, max(max(desc_logf_cate))+0.02];
h=[];
linestyle = {'-', ':', '-.', '--', '--', ':', '--'};
marker   = {'none', 'none', 'none', '*', 'o', 's', '+'};
figure(3);clf;
for i=1:7
    h(i)=plot(t,desc_logf_cate(:,i),'linewidth',1.1, 'LineStyle', linestyle{i}, 'Marker', marker{i}, 'Color', 'k');hold on;
    [yvalley, indv]=min(desc_logf_cate(:,i));
    [ypeak, indp]=findpeaks(desc_logf_cate(:,i));
     line([indv, indv], [ylim(1), yvalley], 'Linestyle', '--','Color', 'k');hold on;
    if size(ypeak,1)==2
    line([indp(2), indp(2)], [ylim(1), ypeak(2)],'Linestyle', '--','Color', 'k');hold on;
    end
end
set(gca,'XTick',x, 'XTickLabel',x_tklabel, 'XLim', [1,24], 'YLim', ylim);
xlabel('Time','FontSize',14,'FontWeight','bold');
ylabel('Log Price','FontSize',14,'FontWeight','bold');
legend([h(1),h(2),h(3),h(4),h(5),h(6),h(7)],'Mon','Tue','Wed','Thu','Fri','Sat','Sun','location','northeast');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Min, max, mean of NP and CA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   NP
addpath(genpath('fdaM'))  
addpath(genpath('NPdata'))
year = [2013:2017];
M=24; % hourly points
% the unwarped data
load('electricity_prices.mat')
step=1;
% % take log10(.+1) and vertically combined
fwhole=log10(fwhole+1);
flogmatr = reshape(fwhole, M, []);

% or %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   CA
addpath(genpath('fdaM'))  
% the unwarped data
load('CA_hourly.dat')
step=1;
M=24; % hourly points
Total_obs=1037; 
start_point=461; 
N=Total_obs-start_point+1; %577 
flog=log10(CA_hourly(M*(start_point-1)+1:end,3)+1); % 577*24 by 1
fmatr = reshape(flog, M, []);
flogmatr = fmatr;

%%  calculate min, max, mean
describef(:,1) = min(flogmatr,[],2);
describef(:,2) = max(flogmatr,[],2);
describef(:,3) = mean(flogmatr,2);
describef(:,4) = std(flogmatr,0,2);

%% Plotting
electime=1:M;
electimescaled=zeros(M,1);
for i=1:M
    electimescaled(i)=electime(i)/M;
end
x = electimescaled(2:2:end);
x_label = {'2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24'};
x_lim = [electimescaled(1), electimescaled(end)];
t=[1:1:M]';
%% Min curve
figure(11);clf;
plot(electimescaled,describef(:,1),'linewidth',2,'Color','b');  
set(gca, 'XTick',x, 'XTickLabel',x_label, 'XLim', x_lim);
xlabel('Time','FontSize',16,'FontWeight','bold');
ylabel('Log Price','FontSize',16,'FontWeight','bold');

%% Max curve
figure(12);clf;
plot(electimescaled,describef(:,2),'linewidth',2,'Color','b');   
set(gca, 'XTick',x, 'XTickLabel',x_label, 'XLim', x_lim);
xlabel('Time','FontSize',16,'FontWeight','bold');
ylabel('Log Price','FontSize',16,'FontWeight','bold');

%% Mean curve
figure(18);clf;
plot(electimescaled,describef(:,3),'linewidth',2,'Color','b');hold on;
plot(electimescaled,describef(:,3)+describef(:,4),'linewidth',2,'linestyle','--','Color','b');
plot(electimescaled,describef(:,3)-describef(:,4),'linewidth',2,'linestyle','--','Color','b');      
set(gca, 'XTick',x, 'XTickLabel',x_label, 'XLim', x_lim);
xlabel('Time','FontSize',16,'FontWeight','bold');
ylabel('Log Price','FontSize',16,'FontWeight','bold');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. time series of 17:00 and 18:00
%% NP.  
addpath(genpath('fdaM'))  
addpath(genpath('NPdata'))
year = [2013:2017];
M=24;
load('electricity_prices.mat')
step=1;
fwhole=log10(fwhole+1);
flog = fwhole;
N    = size(flog, 1)/M; %
flogmatr = reshape(flog, M, N);
f0 = flogmatr;
t=[1:1:M]'; 

day=[1:1:N];
x = [1,size(f2013,1)/M+1,(size(f2013,1)+size(f2014,1))/M+1,(size(f2013,1)+size(f2014,1)+size(f2015,1))/M+1,(size(f2013,1)+size(f2014,1)+size(f2015,1)+size(f2016,1))/M+1, N];
x_label = {'2013.01.01','2014.01.01','2015.01.01','2016.01.01','2017.01.01','2017.12.31'};
figure(1);clf;
plot(day,f0(17,:),'linewidth',1.5,'Color','b')
hold on
plot(day,f0(18,:),'r--','linewidth',1.5)
set(gca, 'XTick',x,  'XTickLabel',x_label, 'XLim', [1,N]);
xlabel('Date','FontSize',16,'FontWeight','bold');
ylabel('Log Price','FontSize',16,'FontWeight','bold');
legend('17:00','18:00');

% ACF AND CCF
figure(14);clf;
autocorr(f0(17,:),40);
xlabel('Lag','FontSize',14,'FontWeight','bold');
ylabel('Sample Autocorrelation','FontSize',14,'FontWeight','bold');

figure(14);clf;
crosscorr(f0(17,:),f0(18,:)); 
xlabel('Lag','FontSize',16,'FontWeight','bold');
ylabel('Sample Cross Correlation','FontSize',16,'FontWeight','bold');

%% CA. time series of 17:00 and 18:00
addpath(genpath('fdaM'))  
load('CA_hourly.dat')
step=1;
M=24; % hourly points
Total_obs=1037;
start_point=461;
N=Total_obs-start_point+1; %577
t=[1:1:M]'; 
train_sample=300;
f0=log10(CA_hourly((24*(start_point-1)+1):end,3)+1); % 577*24 by 1
f0=reshape(f0,M,N);
day=[1:1:577];
x = [1,93,185,276,367,459,551];
x_label = {'1999.07.05','1999.10.05','2000.01.05','2000.04.05','2000.07.05','2000.10.05','2001.01.05'};
figure(2);clf;
plot(day,f0(17,:),'linewidth',1.5,'Color','b')
hold on
plot(day,f0(18,:),'r--','linewidth',1.5)
set(gca, 'XTick',x,  'XTickLabel',x_label);
xlabel('Date','FontSize',16,'FontWeight','bold');
ylabel('Log Price','FontSize',16,'FontWeight','bold');
legend('17:00','18:00');

% ACF AND CCF
figure(14);clf;
autocorr(f0(17,:),40);
xlabel('Lag','FontSize',14,'FontWeight','bold');
ylabel('Sample Autocorrelation','FontSize',14,'FontWeight','bold');

figure(14);clf;
crosscorr(f0(17,:),f0(18,:)); 
xlabel('Lag','FontSize',16,'FontWeight','bold');
ylabel('Sample Cross Correlation','FontSize',16,'FontWeight','bold');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. plot of Karcher mean & the center
% 6. plot of Karcher Mean of Warping functions, categorized by Mon, Tue till Sun.
%% NP
addpath(genpath('fdaM'))  
addpath(genpath('NPdata'))
load('electricity_prices.mat')
step=1;
fwhole=log10(fwhole+1);
M=24;
flog = fwhole;
f = reshape(flog, M, []);
N    = size(flog, 1)/M; 
[first_dayno, first_dayofweek] = weekday(datenum(fdatewhole(1),'dd/mm/yyyy'))
t=[1:1:M]'; 
load('LogAlignedNPwhole.mat')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Karcher mean and Center
linestyle = {'-', ':', '-.', '--', '--', ':', '--'};
marker   = {'none', 'none', 'none', '*', 'o', 's', '+'};
figure(15);clf;
x = [1:1:24];
x_tklabel = { ' ', '2', ' ', '4', ' ', '6', ' ', '8', ' ', '10', ' ', '12', ' ', '14', ' ', '16', ' ', '18', ' ', '20', ' ', '22', ' ', '24'};
for i=1:N
   h3= plot(t, q0(:,i), 'Linewidth', 1, 'Color', [0.8,0.8,0.8]); hold on;
end
h2=plot(t, mqn, 'linewidth', 2, 'Color', 'k');
set(gca, 'XTick', x,'XTickLabel',x_tklabel, 'XLim', [0.97,24.01], 'YLim',[-0.6,0.7]);
legend([h2,h3],{'Center of Karcher mean','SRSFs'}, 'location','northeast');
xlabel('Time','FontSize',16,'FontWeight','bold');
ylabel('Log Price','FontSize',16,'FontWeight','bold');

%%%%%%%%%%%%%%%%%%%%  Karcher Mean of Warping functions, categorized by Mon, Tue till Sun.
if size(gam, 2)~=M
    gam = gam';
end
[~, gamI_cate]= KarcherMeansof_warpingfunctions(gam, first_dayno);
% set the Rainbow color matrix and plot the original functions
colmap = RainbowColorsQY(7);
colmap(4,:)=[0,1,1];
CM=colmap;
gamI_cate = gamI_cate';
figure(2); clf;
for i=1:N 
h8=plot((0:M-1)/(M-1), gam(i,:), 'linewidth', 1,'Color',[0.8,0.8,0.8]);hold on;
end
hold on;
h1=plot((0:M-1)/(M-1), gamI_cate(:,2),  'linewidth',1.1, 'LineStyle', linestyle{1}, 'Marker', marker{1}, 'Color', 'k');
h2=plot((0:M-1)/(M-1), gamI_cate(:,3),  'linewidth',1.1, 'LineStyle', linestyle{2}, 'Marker', marker{2}, 'Color', 'k');
h3=plot((0:M-1)/(M-1), gamI_cate(:,4),  'linewidth',1.1, 'LineStyle', linestyle{3}, 'Marker', marker{3}, 'Color', 'k');
h4=plot((0:M-1)/(M-1), gamI_cate(:,5),  'linewidth',1.1, 'LineStyle', linestyle{4}, 'Marker', marker{4}, 'Color', 'k');
h5=plot((0:M-1)/(M-1), gamI_cate(:,6),  'linewidth',1.1, 'LineStyle', linestyle{5}, 'Marker', marker{5}, 'Color', 'k');
h6=plot((0:M-1)/(M-1), gamI_cate(:,7),  'linewidth',1.1, 'LineStyle', linestyle{6}, 'Marker', marker{6}, 'Color', 'k');
h7=plot((0:M-1)/(M-1), gamI_cate(:,1),  'linewidth',1.1, 'LineStyle', linestyle{7}, 'Marker', marker{7}, 'Color', 'k');
legend([h1,h2,h3,h4,h5,h6,h7,h8],{'Mon','Tue','Wed','Thu','Fri','Sat','Sun','Warping functions'}, 'location','southeast');

%% CA
addpath(genpath('fdaM'))  
addpath(genpath('CAdata'))
load('CA_hourly.dat')
step=1;
M=24; 
Total_obs=1037; 
start_point=461; 
N=Total_obs-start_point+1;%577
%% prepare prices and dates
flog=log10(CA_hourly(M*(start_point-1)+1:end,3)+1); % 577*24 by 1
f = reshape(flog, M, []);
date = CA_hourly(M*(start_point-1)+1:end,1);% start_point:Total_obs
fdate = reshape(date, M, []);
f = [fdate(1,:); f]; 
t    = [1:1:M]'; 
[first_dayno, first_dayofweek] =  weekday(datenum(num2str(f(1,1)),'yyyymmdd'))
% load('LogAlignedCAwhole.mat')
load('elecAlignedLog1_577.mat')% the results of doing time warping for all the California data--in sample analysis
% f0=fn([14:M, 1:13], :); %([12:M,1:11],:);
qn = qn([14:M, 1:13], :);
mqn = mqn([14:M, 1:13]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Karcher mean and Center
colmap = RainbowColorsQY(32);
figure(15);clf;
x = [1:1:24];
x_tklabel = {' ', '2', ' ', '4', ' ', '6', ' ', '8', ' ', '10', ' ', '12', ' ', '14', ' ', '16', ' ', '18', ' ', '20', ' ', '22', ' ', '24'};
for i=1:N
   h3= plot(t, qn(:,i), 'Linewidth', 1, 'Color', [0.8,0.8,0.8]); hold on;
end
h2=plot(t, mqn, 'linewidth', 2, 'Color', 'k');
set(gca, 'XTick', x,'XTickLabel',x_tklabel, 'XLim', [0.97,24.01], 'YLim',[-0.7,0.8]);
legend([h2,h3],{'Center of Karcher mean','SRSFs'}, 'location','northeast');
xlabel('Time','FontSize',16,'FontWeight','bold');
ylabel('Log Price','FontSize',16,'FontWeight','bold');

%%%%%%%%%%%%%%%%%%%%  Karcher Mean of Warping functions, categorized by Mon, Tue till Sun.
if size(gam, 2)~=M
    gam = gam';
end
[~, gamI_cate]= KarcherMeansof_warpingfunctions(gam, first_dayno);
% set the Rainbow color matrix and plot the original functions
colmap = RainbowColorsQY(7);
colmap(4,:)=[0,1,1];
CM=colmap;
gamI_cate=gamI_cate';
linestyle = {'-', ':', '-.', '--', '--', ':', '--'};
marker   = {'none', 'none', 'none', '*', 'o', 's', '+'};
figure(2); clf;
for i=1:N 
h8=plot((0:M-1)/(M-1), gam(i,:), 'linewidth', 1,'Color',[0.8,0.8,0.8]);hold on;
end
hold on;
h1=plot((0:M-1)/(M-1), gamI_cate(:,2), 'linewidth',1.1, 'LineStyle', linestyle{1}, 'Marker', marker{1}, 'Color', 'k');
h2=plot((0:M-1)/(M-1), gamI_cate(:,3), 'linewidth',1.1, 'LineStyle', linestyle{2}, 'Marker', marker{2}, 'Color', 'k');
h3=plot((0:M-1)/(M-1), gamI_cate(:,4),  'linewidth',1.1, 'LineStyle', linestyle{3}, 'Marker', marker{3}, 'Color', 'k');
h4=plot((0:M-1)/(M-1), gamI_cate(:,5), 'linewidth',1.1, 'LineStyle', linestyle{4}, 'Marker', marker{4}, 'Color', 'k');
h5=plot((0:M-1)/(M-1), gamI_cate(:,6),  'linewidth',1.1, 'LineStyle', linestyle{5}, 'Marker', marker{5}, 'Color', 'k');
h6=plot((0:M-1)/(M-1), gamI_cate(:,7),  'linewidth',1.1, 'LineStyle', linestyle{6}, 'Marker', marker{6}, 'Color', 'k');
h7=plot((0:M-1)/(M-1), gamI_cate(:,1), 'linewidth',1.1, 'LineStyle', linestyle{7}, 'Marker', marker{7}, 'Color', 'k');
legend([h1,h2,h3,h4,h5,h6,h7,h8],{'Mon','Tue','Wed','Thu','Fri','Sat','Sun','Warping functions'}, 'location','southeast');
