function [] = printIntervals(fig,inter,fileName,i,nTrainingPeriods,lag)
%PRINTINTERVALS prints the intervlas used to calibrate the LAR models
%   
%   Arguments
%   FIG      a ficure
%   inter    a cell containing the used intervals for the LAR forecast
%   FILENAME a string containing the location and file name
%   I        group index
%   NTRAININGPERIODS the number of training periods
%   LAG      the lag in the LAR prediction
%
%   Returns
%   saves the print to the given location   

% load data and set parameters 
global dateVector FONT_SIZE

nPeriods                             = length(inter{i});
nPredictionPeriodes                  = nPeriods-nTrainingPeriods-(lag-1);
NUMBER_PRINCIPAL_COMPONENTS_FUNCTION = size(inter{i},2);

CURVE_WIDTH_NARROW                   = 0.6;   % width of line in prints
SPACE_BETWEEN_LEGEND_LABELS_Y        = floor((nPredictionPeriodes-1)/2);
SPACE_BETWEEN_LEGEND_LABELS_X        = SPACE_BETWEEN_LEGEND_LABELS_Y*2;

% define size of output
width=16; % centimeters
height=5; % centimeters
set(0, 'CurrentFigure', fig);
clf reset;
fig.PaperUnits              = 'centimeters';
fig.PaperPosition           = [0,0 width, height];
fig.PaperPositionMode       = 'manual';
set(fig, 'Units', 'normalized', 'Position', [0.1, 0.15, 0.8, 0.75]); 

dateSelectionX  = mod(nPeriods,SPACE_BETWEEN_LEGEND_LABELS_X):...
                  SPACE_BETWEEN_LEGEND_LABELS_X:nPeriods;
datesForLegendX = datestr(dateVector{i}(dateSelectionX,:),'mmmyyyy');
dateSelectionY  = nTrainingPeriods+lag:SPACE_BETWEEN_LEGEND_LABELS_Y:...
                  nPeriods;
datesForLegendY = datestr(dateVector{i}(dateSelectionY,:),'mmmyyyy');
        


for n=1:NUMBER_PRINCIPAL_COMPONENTS_FUNCTION
    subplot(1,NUMBER_PRINCIPAL_COMPONENTS_FUNCTION,n) 
    hold all;
    
    for s=nTrainingPeriods+lag:nPeriods
        % plot interval
        x = [s-inter{i}(s,n)-lag+1,s-lag];
        y = [s,s];
        plot(x,y,'b','LineWidth',CURVE_WIDTH_NARROW)
        
        %plot break
        x = [s-inter{i}(s,n)-lag-4,s-inter{i}(s,n)-lag+1];
        y = [s,s];
        plot(x,y,'Color',[1 0.5 0],'LineWidth',CURVE_WIDTH_NARROW)
    end;
    
    hold off;
    
    xlim([1 nPeriods]);
    set(gca,'XTick',dateSelectionX);
    set(gca,'XTickLabel',datesForLegendX);    
    
    ylim([nTrainingPeriods+lag-1 nPeriods])
    set(gca,'YTick',dateSelectionY);
    set(gca,'YTickLabel',datesForLegendY);
    set(gca,'TickDir','out')        

    title(strcat('Intervals PC',num2str(n)));  
end;

set(findall(fig,'-property','FontSize'),'FontSize',FONT_SIZE)
print(fileName,'-dpng','-r400'); 

end

