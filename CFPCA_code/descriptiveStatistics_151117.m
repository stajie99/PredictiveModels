%DESCRIPTIVESTATISTICS runs descriptive statistics of the data set
%
%  Prints of single selected curves
%  Print of min, max, median, quantiles
%  Descriptive statistics: mean, standard deviation, min, max
%  Sample autocorrelation table and functions
%  3D yield curve prints
%  3D covariance surface
%  3D correlation surface

% load data and set parameters
load('Variables\inputs.mat');
addpath('Library') ;

STEP_SIZE_PRINTING          = 10;    % step size for printing of all curves
CURVE_WIDTH_NARROW          = 0.2;   % width of line in prints
CURVE_WIDTH_WIDE            = 2;     % width of mean line in prints
SPACE_BETWEEN_LEGEND_LABELS = 100;   % depth of entries in colorbar

COUNTRY_NAMES_LONG          = ['USDEFFR'; 'SONIA  '; 'EONIA  '; 'TONAR  '];

CUSTOM_COLOR_ORDER      = [0.1 0.1 1.0; 
                           1.0 0.0 0.0;
                           0.7 0.0 0.9;
                           1.0 0.8 0.0];
CUSTOM_COLOR_MAP_BLUE   = [0.1 0.1 0.5
                           0.1 0.1 0.6
                           0.1 0.1 0.7
                           0.1 0.1 0.8
                           0.2 0.2 0.9
                           0.4 0.4 1.0];
CUSTOM_COLOR_MAP_RED    = [0.5 0.0 0.0
                           0.6 0.0 0.0
                           0.7 0.0 0.0
                           0.8 0.0 0.0
                           0.9 0.2 0.2
                           1.0 0.2 0.2];
CUSTOM_COLOR_MAP_VIOLET = [0.5 0.0 0.7
                           0.6 0.0 0.7
                           0.6 0.0 0.8
                           0.7 0.0 0.8
                           0.7 0.2 0.9
                           0.75 0.4 1.0];
CUSTOM_COLOR_MAP_YELLOW = [0.8 0.6 0.0
                           0.8 0.7 0.0
                           0.8 0.8 0.0
                           0.9 0.8 0.0
                           1.0 0.8 0.2
                           1.0 0.9 0.4];
                       
% define size of output
fig                         = figure;
width                       = 16; % centimeters
height                      = 10.7; % centimeters
fig.PaperUnits              = 'centimeters';
fig.PaperPosition           = [0,0 width, height];
fig.PaperPositionMode       = 'manual';
set(fig, 'Units', 'normalized', 'Position', [0.1, 0.15, 0.8, 0.75]); 
                       
                       
                       
%---------------------------------------------------------------------------
% Prints of single selected curves
%---------------------------------------------------------------------------
% load('Variables\4groups_3PC_PCs.mat')
splineDataCell = cell(length(data),1);
for i=1:length(data)
    for j=1:length(data{i});
        splineDataCell{i}(j) = spap2(KNOT_SEQUENCE,SPLINE_ORDER,...
                                     MATURITIES,data{i}(j,:));
    end;
end;

% select dates
settleDatesPlot = datenum({'31-Jul-2012','31-May-2013','31-Mar-2014',...
    '30-Jan-2015'});
indexVector     = sort(datefind(settleDatesPlot,datenum(dateVector{i})),...
                       'ascend');

% create plot for selected dates
for j=1:length(settleDatesPlot)
    subplot(2,2,j)
    hold on
    set(gca, 'ColorOrder', CUSTOM_COLOR_ORDER)
    fnplt(splineDataCell{1}(indexVector(j)),CURVE_WIDTH_NARROW);
    fnplt(splineDataCell{2}(indexVector(j)),CURVE_WIDTH_NARROW);
    fnplt(splineDataCell{3}(indexVector(j)),CURVE_WIDTH_NARROW);
    fnplt(splineDataCell{4}(indexVector(j)),CURVE_WIDTH_NARROW);
   
    h1 = scatter(MATURITIES,data{1}(indexVector(j),:),10,...
                 'MarkerEdgeColor',[.1 .1 1],'MarkerFaceColor',[.1 .1 1]);
    h2 = scatter(MATURITIES,data{2}(indexVector(j),:),10,...
                 'MarkerEdgeColor',[1 0 0],  'MarkerFaceColor',[1 0 0]);
    h3 = scatter(MATURITIES,data{3}(indexVector(j),:),10,...
                 'MarkerEdgeColor',[.7 0 .9],'MarkerFaceColor',[.7 0 .9]);
    h4 = scatter(MATURITIES,data{4}(indexVector(j),:),10,...
                 'MarkerEdgeColor',[1 .8 0], 'MarkerFaceColor',[1 .8 0]);          
    hold off;   
    axis([0 30 -0.2 4]);
    set(gca, 'XTick',[0 5 10 15 20 25 30])
    set(gca, 'XTickLabel',{0,60, 120,180,240,300,360})
    set(gca, 'YTick',[0 1 2 3 4])
    xlabel('Matutity (months)');
    ylabel('Yield (%)');
    title(strcat(datestr(settleDatesPlot(j),'mm/dd/yyyy'),' Yield Curves'));
    if j==length(settleDatesPlot)
        legend([h1,h2,h3,h4],COUNTRY_NAMES_LONG(1,:),...
               COUNTRY_NAMES_LONG(2,:),COUNTRY_NAMES_LONG(3,:),...
               COUNTRY_NAMES_LONG(4,:),'Location','NorthWest');
    end;
end;
set(findall(fig,'-property','FontSize'),'FontSize',FONT_SIZE);
print(strcat('Figures\Descriptives\selected_Curves.png'),'-dpng','-r400');



%---------------------------------------------------------------------------
% Print of quantiles
%---------------------------------------------------------------------------
set(0, 'CurrentFigure', fig);
clf reset;
for i=1:length(data)
    splineDataMedian     = spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                                 quantile(data{i},0.50));
    splineData25Quantile = spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                                 quantile(data{i},0.25));
    splineData75Quantile = spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                                 quantile(data{i},0.75));
    splineDataMin        = spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                                 min(data{i}));
    splineDataMax        = spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                                 max(data{i}));
    % plot quantiles
    subplot(2,2,i)
    set(groot,'defaultAxesColorOrder', CUSTOM_COLOR_ORDER(i,:));
    fnplt(splineDataMedian,'-', 1.3);
    hold on
    set(groot,'defaultAxesColorOrder', CUSTOM_COLOR_ORDER(i,:));
    fnplt(splineData25Quantile,':',1.0);
    fnplt(splineData75Quantile,':',1.0);
    fnplt(splineDataMin,'-.',1.0);
    fnplt(splineDataMax,'-.',1.0);
    hold off;   
    axis([0 30 -0.2 4]);
    set(gca, 'XTick',[0 5 10 15 20 25 30])
    set(gca, 'XTickLabel',{0,60, 120,180,240,300,360})
    set(gca, 'YTick',[0 1 2 3 4])
    xlabel('Matutity (months)');
    ylabel('Yield (%)');
    title(COUNTRY_NAMES_LONG(i,:));
    clearvars sp50quant sp25quant sp75quant splineDataMedian 
    clearvars splineDataMin splineDataMax
end;
print(strcat('Figures\Descriptives\quantiles.png'),'-dpng','-r400');
clearvars i



%---------------------------------------------------------------------------
% Descriptive Statistics
%---------------------------------------------------------------------------
dataDescriptiveStats             = cell(length(data),1);
for i=1:length(data)
    dataDescriptiveStats{i}      = zeros(length(MATURITIES),4);
    dataDescriptiveStats{i}(:,1) = mean (data{i})';
    dataDescriptiveStats{i}(:,2) = std  (data{i})';
    dataDescriptiveStats{i}(:,3) = min  (data{i})';
    dataDescriptiveStats{i}(:,4) = max  (data{i})';
    xlswrite('Figures\Descriptives\descriptives.xls',...
        dataDescriptiveStats{i},COUNTRY_NAMES(i,:));
end;
clearvars i

% sample auto correclation table
autoCorrelation    = cell(length(data),1);
for i=1:length(data)
    consideredLags = [1, 5, 10];
    autoCorrelation{i}=zeros(length(MATURITIES),length(consideredLags));
    for m=1:length(consideredLags)
        for j=1:length(MATURITIES)
            autoCorrelationTemp = xcorr(data{i}(:,j),consideredLags(m),...
                                        'coeff');
            autoCorrelation{i}(j,m) = autoCorrelationTemp(end);
        end;
    end;
    xlswrite('Figures\Descriptives\autocorrelation.xls',...
             autoCorrelation{i},COUNTRY_NAMES_LONG(i,:),'B2:D22');
end;
clearvars i j m consideredLags

% sample auto correclation functions
consideredLags             = 60;     % up to 60 trading days
consideredMatrurityIndicis = [1,7,17];      %1,12,120 months
autocorr                   = zeros(length(data),...
                                   length(consideredMatrurityIndicis));
set(0, 'CurrentFigure', fig);
clf reset;
set(groot,'defaultAxesColorOrder', CUSTOM_COLOR_ORDER);
for i=1:length(data)
    subplot(2,2,i)
    hold all
    for m=1:length(consideredMatrurityIndicis)
        autoCorrelationTemp = xcorr(data{i}(:,consideredMatrurityIndicis...
                                    (m)),consideredLags,'coeff');
        % plot autocorrelation from lag 1 up to lag consideredLags
        ax = gca;
        ax.ColorOrderIndex = i;
        if m==1
            plot(1:consideredLags,autoCorrelationTemp(end-...
                                  consideredLags+1:end),':');
        elseif m==2
            plot(1:consideredLags,autoCorrelationTemp(end-...
                                  consideredLags+1:end),'--');
        else
            plot(1:consideredLags,autoCorrelationTemp(end-...
                                  consideredLags+1:end),'-');
        end;
        autocorr(i,m) = autoCorrelationTemp(end);
    end;
    hold off
    axis([0 60 0.6 1]);
    set(gca, 'XTick',[0 20 40 60])
    set(gca, 'YTick',[0.6 0.7 0.8 0.9 1.0])
    xlabel('Lag (days)');
    ylabel('autocorrelation');
    title(COUNTRY_NAMES_LONG(i,:));
end;
print(strcat('Figures\Descriptives\autocorrelation.png'),'-dpng','-r400');
clearvars i j m consideredMatrurities consideredLags



%--------------------------------------------------------------------------
% 3D Curve Prints
%--------------------------------------------------------------------------
set(0, 'CurrentFigure', fig);
clf reset;
stepSize = 20;
for i=1:length(data)
    subplot(2,2,i)
    surf(data{i})
    shading interp
    if i==1
        colormap(CUSTOM_COLOR_MAP_BLUE);
    elseif i==2
        colormap(CUSTOM_COLOR_MAP_RED);
    elseif i==3
        colormap(CUSTOM_COLOR_MAP_VIOLET);
    else
        colormap(CUSTOM_COLOR_MAP_YELLOW);
    end;
    freezeColors
    view(230,20)
    % set tick as months
    set(gca, 'XTick',[0  10  20 30])
    xlim([0 30]);
    set(gca, 'XTickLabel',{0,120,240,360})
    % select beginning of years for ticks
    ylim([0 716]);
    dateSelection=[127, 388, 649];
    set(gca, 'YTick',dateSelection);
    set(gca, 'YTickLabel',{dateVector{i}(dateSelection(1),1),...
        dateVector{i}(dateSelection(2),1),...
        dateVector{i}(dateSelection(3),1)});
    zlim([-0.2 3.5])
    set(gca, 'ZTick',[0 1 2 3 ])
    zlabel('Yield (%)');
    ylabel('Date')
    xlabel('Maturity')
    title(COUNTRY_NAMES_LONG(i,:));
end;
print(strcat('Figures\Descriptives\3DCurves.png'),'-dpng','-r400');



%--------------------------------------------------------------------------
% 3D Covariance Surface
%--------------------------------------------------------------------------
set(0, 'CurrentFigure', fig);
clf reset;
dataCovariance = cell(length(data),1);
for i=1:length(data)
    subplot(2,2,i)
    dataCovariance{i} = cov(data{i});
    surf(dataCovariance{i})
%     shading interp
    if i==1
        colormap(CUSTOM_COLOR_MAP_BLUE);
    elseif i==2
        colormap(CUSTOM_COLOR_MAP_RED);
    elseif i==3
        colormap(CUSTOM_COLOR_MAP_VIOLET);
    else
        colormap(CUSTOM_COLOR_MAP_YELLOW);
    end;
    freezeColors
    xlim([0 21]);
    ylim([0 21]);
    set(gca, 'XTick',[0  7  14 21])
    set(gca, 'XTickLabel',{0,12,96,360})
    set(gca, 'YTick',[0  7  14 21])
    set(gca, 'YTickLabel',{0,12,96,360})
    if i==4
       zlim([0 0.05]) 
    else
        zlim([-0.05 0.25])
    end;

    ylabel('Maturity')
    xlabel('Maturity')
    title(COUNTRY_NAMES_LONG(i,:));
end;
print(strcat('Figures\Descriptives\3DCovarianceSurf.png'),'-dpng','-r400');



%--------------------------------------------------------------------------
% 3D Correlation Surface
%--------------------------------------------------------------------------
set(0, 'CurrentFigure', fig);
clf reset;
dataCorr = cell(length(data),1);
for i=1:length(data)
    subplot(2,2,i)
    dataCorr{i} = corrcoef(data{i});
    surf(dataCorr{i})
%     shading interp
    if i==1
        colormap(CUSTOM_COLOR_MAP_BLUE);
    elseif i==2
        colormap(CUSTOM_COLOR_MAP_RED);
    elseif i==3
        colormap(CUSTOM_COLOR_MAP_VIOLET);
    else
        colormap(CUSTOM_COLOR_MAP_YELLOW);
    end;
    freezeColors
%     view(230,20)
    % set tick as months
    xlim([0 21]);
    ylim([0 21]);
    
    set(gca, 'XTick',[0  7  14 21])
    set(gca, 'XTickLabel',{0,12,96,360})
    set(gca, 'YTick',[0  7  14 21])
    set(gca, 'YTickLabel',{0,12,96,360})
    ylabel('Maturity')
    xlabel('Maturity')
    title(COUNTRY_NAMES_LONG(i,:));
end;
print(strcat('Figures\Descriptives\3DCorrelationSurf.png'),'-dpng','-r400');
close;