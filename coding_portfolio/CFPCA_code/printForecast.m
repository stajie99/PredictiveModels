function [] = printForecast(fig,scores,scoresPred,i,fileName,...
                            nTrainingPeriods,lag,dns)
%PRINTFORECAST prints DNS factors or the scores prediction in comparision 
%to the DNS factors or the actual scores
%   
%   Arguments
%   FIG      a ficure
%   SCORES   a cell containing the DNS factors or principal component scores 
%            time series
%   SCORESPRED a cell containing the forecasted DNS factor time series or 
%            principal component scores time series
%   I        group index
%   FILENAME a string containing the location and file name
%   NTRAININGPERIODS the number of training periods
%   LAG      the lag in the AR prediction
%   DNS      an indicator variable for the description of the graphs
%
%   Returns
%   save the print to the given location   

% load data and set parameters 
global dateVector FONT_SIZE

nPeriods                             = length(scores{i});
nPredictionPeriodes                  = nPeriods-nTrainingPeriods-(lag-1);
NUMBER_PRINCIPAL_COMPONENTS_FUNCTION = size(scores{i},2);

GREY                                 = [0.7 0.7 0.7];
CURVE_WIDTH_NARROW                   = 0.3;   % width of line in prints
SPACE_BETWEEN_LEGEND_LABELS          = floor((nPredictionPeriodes-1)/2);

% define size of output
width=16; % centimeters
height=5; % centimeters
set(0, 'CurrentFigure', fig);
clf reset;
fig.PaperUnits              = 'centimeters';
fig.PaperPosition           = [0,0 width, height];
fig.PaperPositionMode       = 'manual';
set(fig, 'Units', 'normalized', 'Position', [0.1, 0.15, 0.8, 0.75]); 

dateSelection  = nTrainingPeriods+lag:SPACE_BETWEEN_LEGEND_LABELS:nPeriods;
datesForLegend = datestr(dateVector{i}(dateSelection,:),'mmmyyyy');
        


for n=1:NUMBER_PRINCIPAL_COMPONENTS_FUNCTION
    subplot(1,NUMBER_PRINCIPAL_COMPONENTS_FUNCTION,n) 
    plot([1 nPredictionPeriodes], [0 0],'Color',GREY,'LineWidth',...
         CURVE_WIDTH_NARROW);
    hold on;  
    plot(scores{i}(end-nPredictionPeriodes+1:end,n),'Color',[1 0.5 0],...
        'LineWidth',CURVE_WIDTH_NARROW);   
    plot(scoresPred{i}(end-nPredictionPeriodes+1:end,n),'b--',...
        'LineWidth',CURVE_WIDTH_NARROW);
    hold off;
    xlim([1 nPredictionPeriodes]);
    ylim([-1.2*max(abs(scoresPred{i}(:,n))) ...
          1.2*max(abs(scoresPred{i}(:,n)))]);
    set(gca, 'XTick',dateSelection-(nTrainingPeriods+lag)+1);
    set(gca, 'XTickLabel',datesForLegend);
    set(gca,'TickDir','out')        
    if dns==1
        title(strcat('Beta',num2str(n),' Forecast'));  
        if n==1
            ylabel('Beta')
        end;
    else
        title(strcat('Scores PC',num2str(n),' Forecast'));  
        if n==1
            ylabel('PC Score');
        end;
    end; 
end;

set(findall(fig,'-property','FontSize'),'FontSize',FONT_SIZE)
print(fileName,'-dpng','-r400'); 

end