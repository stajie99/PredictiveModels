function [] = printPcsOnly(fig,i,scores,fileName)
% PRINTPCSONLY: creates and saves print of residual yield curves and 
% principal components scores
%
%   Arguments
%   FIG      a figure
%   I        the group index
%   SCORES   the principal component scores
%   PCSAVEARRAY the eigenvalues in the first row and the corresponding
%            eigenfunctions in the subsequent rows
%   FILENAME the complete path and name to save the figure
%
%   Result
%   saved image at given fileName
    
% load data and set parameters
global data dateVector NUMBER_PRINCIPAL_COMPONENTS ...
       MATURITIES splineDataMeanCorrected FONT_SIZE

% define size of output
width=16; % centimeters
height=5; % centimeters

set(0, 'CurrentFigure', fig);
clf reset;
fig.PaperUnits              = 'centimeters';
fig.PaperPosition           = [0,0 width, height];
fig.PaperPositionMode       = 'manual';
set(fig, 'Units', 'normalized', 'Position', [0, 0, 1, 1]); 

STEP_SIZE_PRINTING          = 10;    % step size for printing of all curves
CURVE_WIDTH_NARROW          = 0.5;   % width of line in prints
% CURVE_WIDTH_WIDE            = 2.0;   % width of mean line in prints
GREY                        = [0.7 0.7 0.7];

X_MIN                       = min(MATURITIES);
X_MAX                       = max(MATURITIES);
Y_MIN                       = -1.5;
Y_MAX                       = 1.5;

dataLength                  = length(data{i});

SPACE_BETWEEN_LEGEND_LABELS = floor(length(data{i})/5);
colorBarSettings            = varycolor(ceil(dataLength/...
                                  STEP_SIZE_PRINTING)+1);



%---------------------------------------------------------------------------
% plot PCs
%---------------------------------------------------------------------------
% subplot of mean corrected splines
subplot(1,NUMBER_PRINCIPAL_COMPONENTS+2,1,'Position',...
        [0.05, 0.15, 0.15, 0.75]);
set(gca, 'ColorOrder', colorBarSettings);

hold all;
for j=1:STEP_SIZE_PRINTING:dataLength;
    fnplt(splineDataMeanCorrected{i}(j),CURVE_WIDTH_NARROW)
end;
plot([X_MIN X_MAX],[0 0],'Color', GREY);
hold off;
axis([X_MIN X_MAX Y_MIN Y_MAX]);
title(' Residual Yield Curves');
ylabel('Yield');
set(gca,'XTick',[X_MIN, 10, 20, 30]);
set(gca,'XTickLabel',{num2str(X_MIN*12),'120','240','360'});
set(gca,'YTick',[-1,0,1]);
set(gca,'TickDir','out')
set(gcf, 'Colormap', colorBarSettings);



% subplot used for color bar
subplot(1,NUMBER_PRINCIPAL_COMPONENTS+2,2,'Position',...
        [0.25, 0.15, 0.1, 0.75]);
axis off;
datesForLegend = datestr(dateVector{i}(1:SPACE_BETWEEN_LEGEND_LABELS:...
                                       dataLength,:),'mm/yyyy');
v = (1:SPACE_BETWEEN_LEGEND_LABELS:dataLength) / dataLength;
set(gcf, 'Colormap', colorBarSettings);
set(gca,'TickDir','out')
colorbar('Ticks',v,'TickLabels',datesForLegend,'Location',...
         'westoutside','AxisLocation','out');



% subplots describing PCs
for n=1:NUMBER_PRINCIPAL_COMPONENTS
    % subplot scores time series
    pos = [0.2*(n+1), 0.15, 0.15, 0.75];
    subplot(1,NUMBER_PRINCIPAL_COMPONENTS+2,n+2,'Position',pos);
    [~,xi] = ksdensity(scores(:,n));
    yScores=max(-xi(1),xi(end));
    
    plot(zeros(length(dateVector{i}),1),'Color', GREY);
    hold on;
    plot(scores(:,n),'k-');
    hold off; 
    dateSelection=[127, 388, 649]; %beginning of years
    xlim([0 716]);
    set(gca, 'XTick',dateSelection);
    set(gca, 'XTickLabel',...
        {datestr(dateVector{i}(dateSelection(1),:),'mmmyy'),...
         datestr(dateVector{i}(dateSelection(2),:),'mmmyy'),...
         datestr(dateVector{i}(dateSelection(3),:),'mmmyy')});
    
    ylim([-yScores yScores]);
    set(gca,'TickDir','out')
    title(strcat('PC', num2str(n),' Scores'));
    if n==1
        ylabel('PC Score');
    end;
end;

set(findall(fig,'-property','FontSize'),'FontSize',FONT_SIZE);
print(fileName,'-dpng','-r400');
end

