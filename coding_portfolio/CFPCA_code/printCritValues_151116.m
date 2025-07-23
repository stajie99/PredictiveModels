%PRINTCRITVALUES print the critical values for the CFPC models

% load and define variables
load('Variables\LAR_all.mat');
addpath('Library');

CURVE_WIDTH_NARROW          = 1.0;    % width of line in prints

% define size of output
fig=figure;
width=16; % centimeters
height=5; % centimeters
fig.PaperUnits              = 'centimeters';
fig.PaperPosition           = [0,0 width, height];
fig.PaperPositionMode       = 'manual';
set(fig, 'Units', 'normalized', 'Position', [0.1, 0.15, 0.8, 0.75]); 

xticks  = 2:3:NUMBER_INTERVALS;
xlabels = {'6w','18w','30w','42w','54w','66w','78w','90w'};

for i=1:NUMBER_GROUPS    
    fileName = strcat('Figures\LAR\lar1_',num2str(...
        NUMBER_PREDICTION_DATES),'_ahead',num2str(AHEAD_SHORT),...
        '_CFPC_int',num2str(NUMBER_INTERVALS),'_criticalVal_',...
        COUNTRY_NAMES(i,:),'.png');
    set(0, 'CurrentFigure', fig);
    clf reset;
    hold on;
    
    plot(criticalValues{i,1},'k-','LineWidth',CURVE_WIDTH_NARROW);
    plot(criticalValues{i,2},'k--','LineWidth',CURVE_WIDTH_NARROW);
    plot(criticalValues{i,3},'k:','LineWidth',CURVE_WIDTH_NARROW);
    
    hold off;
    xlim([2 NUMBER_INTERVALS]);
    set(gca,'XTick',2:3:NUMBER_INTERVALS);
    set(gca,'XTickLabel',xlabels);
    legend('PC1', 'PC2','PC3','Location','northeast');
    set(findall(fig,'-property','FontSize'),'FontSize',FONT_SIZE);
    print(fileName,'-dpng','-r400'); 
    
    for n=1:NUMBER_PRINCIPAL_COMPONENTS
        x = cfpcScores{i}(1:NUMBER_TRAINING_DATES-AHEAD_SHORT,n);
        y = cfpcScores{i}(AHEAD_SHORT+1:NUMBER_TRAINING_DATES,n);
        trueParameter = localLogLikeEst(x,y);
        [AHEAD_SHORT i n trueParameter']
    end;
end;

% set(0, 'CurrentFigure', fig);
% clf reset;    
% 
% for i=1:NUMBER_GROUPS  
%     subplot(1,4,i);
%     hold on;
%     plot(criticalValues{i,1}(2:end),'k-','LineWidth',CURVE_WIDTH_NARROW);
%     plot(criticalValues{i,2}(2:end),'k--','LineWidth',CURVE_WIDTH_NARROW);
%     plot(criticalValues{i,3}(2:end),'k:','LineWidth',CURVE_WIDTH_NARROW);
%     
%     hold off;
%     xlim([1 NUMBER_INTERVALS])
%     if i==NUMBER_GROUPS
%         legend('PC1', 'PC2','PC3','Location','northeast');
%     end;
%     title(COUNTRY_NAMES_LONG(i,:));   
% end;
% set(findall(fig,'-property','FontSize'),'FontSize',FONT_SIZE);
% fileName = strcat('Figures\LAR\lar1_',num2str(...
%     NUMBER_PREDICTION_DATES),'_ahead',num2str(AHEAD_SHORT),...
%     '_CFPC_int',num2str(NUMBER_INTERVALS),'_criticalVal.png');
% print(fileName,'-dpng','-r400'); 

for i=1:NUMBER_GROUPS    
    fileName = strcat('Figures\LAR\lar_',num2str(...
        NUMBER_PREDICTION_DATES),'_ahead',num2str(AHEAD_LONG),...
        '_CFPC_int',num2str(NUMBER_INTERVALS),'_criticalVal_',...
        COUNTRY_NAMES(i,:),'.png');
    set(0, 'CurrentFigure', fig);
    clf reset;
    hold on
    
    plot(aheadCriticalValues{i,1}(2:end),'k-','LineWidth',...
        CURVE_WIDTH_NARROW);
    plot(aheadCriticalValues{i,2}(2:end),'k--','LineWidth',...
        CURVE_WIDTH_NARROW);
    plot(aheadCriticalValues{i,3}(2:end),'k:','LineWidth',...
        CURVE_WIDTH_NARROW);
    
    hold off;
    xlim([1 NUMBER_INTERVALS]);
    set(gca,'XTick',2:3:NUMBER_INTERVALS);
    set(gca,'XTickLabel',xlabels);
    legend('PC1', 'PC2','PC3','Location','northeast');
    set(findall(fig,'-property','FontSize'),'FontSize',FONT_SIZE);
    print(fileName,'-dpng','-r400'); 
    
    for n=1:NUMBER_PRINCIPAL_COMPONENTS
        x = cfpcScores{i}(1:NUMBER_TRAINING_DATES-AHEAD_LONG,n);
        y = cfpcScores{i}(AHEAD_LONG+1:NUMBER_TRAINING_DATES,n);
        trueParameter = localLogLikeEst(x,y);
        [AHEAD_LONG i n trueParameter']
    end;
end;
