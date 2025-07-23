function [] = printPcOnly(fig,PcSaveArray,fileName)
% PRINTPCONLY: creates and saves print of principal components curves
%
%   Arguments
%   FIG      a figure
%   PCSAVEARRAY the eigenvalues in the first row and the corresponding
%            eigenfunctions in the subsequent rows
%   FILENAME the complete path and name to save the figure
%
%   Result
%   saved image at given fileName
    
% load data and set parameters
global FONT_SIZE KNOT_SEQUENCE MATURITIES SPLINE_ORDER 

% define size of output
width=16; % centimeters
height=5; % centimeters

set(0, 'CurrentFigure', fig);
clf reset;
fig.PaperUnits              = 'centimeters';
fig.PaperPosition           = [0,0 width, height];
fig.PaperPositionMode       = 'manual';

CURVE_WIDTH_NARROW          = 1.0;   % width of line in prints
% CURVE_WIDTH_WIDE            = 2.0;   % width of mean line in prints
GREY                        = [0.7 0.7 0.7];
set(fig, 'Units', 'normalized', 'Position', [0, 0, 1, 1]); 
    
NUMBER_PRINCIPAL_COMPONENTS_FUNCTION = size(PcSaveArray,2);

eigenvectors                = PcSaveArray(2:end,:)';

X_MIN                       = min(MATURITIES);
X_MAX                       = max(MATURITIES);
Y_MIN                       = -1.5;
Y_MAX                       = 1.5;



%---------------------------------------------------------------------------
% plot PCs
%---------------------------------------------------------------------------
% subplots describing PCs
for n=1:NUMBER_PRINCIPAL_COMPONENTS_FUNCTION
    % subplot PC spline
    subplot(1,NUMBER_PRINCIPAL_COMPONENTS_FUNCTION,n);
    splinePc=spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,eigenvectors(n,:));
    plot([X_MIN X_MAX],[0 0],'Color',GREY);
    hold on;
    fnplt(splinePc,'k-', CURVE_WIDTH_NARROW);
    axis([X_MIN X_MAX Y_MIN Y_MAX]);
    set(gca,'XTick',[X_MIN, 10, 20, 30]);
    set(gca,'XTickLabel',{num2str(X_MIN*12),'120','240','360'});
    set(gca,'YTick',[-1,0,1]);
    set(gca,'TickDir','out');
    
    title(strcat('PC',num2str(n)));
    hold off;
end;

set(findall(fig,'-property','FontSize'),'FontSize',FONT_SIZE);
print(fileName,'-dpng','-r400');
end