%MODELESTIMATION computes the diefferent models and saves the results to
%files
%
%  Dynamic Nelson-Siegel Model
%  PCA (multivariate) of separate datasets
%  PCA (multivariate) of combined dataset
%  CPC model (multivariate)
%  FPCAs of separate data sets using spline basis
%  CFPCA using spline basis
%  print summary

% load data and set parameters
global data dataMeanCorrected dateVector FONT_SIZE KNOT_SEQUENCE ...
       MATURITIES SPLINE_ORDER  NUMBER_PRINCIPAL_COMPONENTS
global splineData splineDataMeanCorrected splineDataMean COUNTRY_NAMES_LONG

load('Variables\inputs.mat');
addpath('Library') ;

COUNTRY_NAMES_LONG = ['USDEFFR'; 'SONIA  '; 'EONIA  '; 'TONAR  '];
LAMBDA_T           = 0.0609;
GREY               = [0.7 0.7 0.7];
NUMBER_GROUPS      = length(data);
fig                = figure;
STEP_SIZE_PRINTING = 10;    

% spline estimation
splineData              = cell(NUMBER_GROUPS,1);
splineDataMean          = cell(NUMBER_GROUPS,1);
splineDataMeanCorrected = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS
    for j=1:length(data{i});
        splineData{i}(j) = spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                                 data{i}(j,:));
    end;
    splineDataMean{i}=spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                            mean(data{i}));

    for j=1:length(data{i});
       splineDataMeanCorrected{i}(j) = spap2(KNOT_SEQUENCE,SPLINE_ORDER,...
                                             MATURITIES,...
                                             dataMeanCorrected{i}(j,:));
    end;
end;

% calculate the covariance matrice for each group
covarianceMatrices=cell(NUMBER_GROUPS,1);
for i=1:NUMBER_GROUPS;
    covarianceMatrices{i}=cov(data{i});
end;
nFgAlgorithm=[length(data{1}),length(data{2}),length(data{3}),...
              length(data{4})];
  
          
          
%---------------------------------------------------------------------------
% Dynamic Nelson-Siegel Model
%---------------------------------------------------------------------------          
dnsScores           = cell(NUMBER_GROUPS,1);
dnsSaveCell         = cell(NUMBER_GROUPS,1);
correlationsFactors = zeros(NUMBER_GROUPS,3);

% Construct a matrix of the factor loadings
dnsFactorLoadings = [ones(size(MATURITIES'.*12)) ...
    (1 - exp(-LAMBDA_T*MATURITIES'.*12))./(LAMBDA_T*MATURITIES'.*12) ...
    ((1 - exp(-LAMBDA_T*MATURITIES'.*12))./(LAMBDA_T*MATURITIES'.*12) ...
     - exp(-LAMBDA_T*MATURITIES'.*12))];

for i=1:NUMBER_GROUPS
    dnsSaveCell{i}          = zeros(length(MATURITIES)+1,3);
    dnsSaveCell{i}(2:end,:) = dnsFactorLoadings;
end;

% define size of output
width=16; % centimeters
height=5; % centimeters
set(0, 'CurrentFigure', fig);
clf reset;
fig.PaperUnits              = 'centimeters';
fig.PaperPositionMode       = 'manual';
fig.PaperPosition           = [0,0 width, height];
set(fig, 'Units', 'normalized', 'Position', [0, 0, 1, 1]); 

% subplots describing DNS loadings
for m=1:3
    subplot(1,3,m);
    plot([min(MATURITIES'.*12) max(MATURITIES'.*12)], [0 0],'Color', GREY);
    hold on;
    plot(MATURITIES'.*12,dnsFactorLoadings(:,m),'k-')
    hold off;
    axis([min(MATURITIES'.*12) max(MATURITIES'.*12) -1.5 1.5]);    
    set(gca,'XTick',[min(MATURITIES'.*12), 120, 240, max(MATURITIES'.*12)]);
    set(gca,'YTick',[-1,0,1]);
    if m==1 
        ylabel('Loadings');
    end;
    title(strcat('Beta ',num2str(m)));
    set(findall(fig,'-property','FontSize'),'FontSize',FONT_SIZE);
    set(gca,'TickDir','out');   
end;
print('Figures\Models_simple\DNS_Beta_Loadings.png','-dpng','-r400'); 

% plot models
for i=1:NUMBER_GROUPS
    set(0, 'CurrentFigure', fig);
    clf reset;
    set(fig, 'Units', 'normalized', 'Position', [0, 0, 1, 1]); 
    
    SPACE_BETWEEN_LEGEND_LABELS = floor(length(data{i})/5);
    colorBarSettings            = varycolor(ceil(length(data{i})/...
                                  STEP_SIZE_PRINTING)+1);
    
    % Preallocate the Betas
    dnsScores{i} = zeros(size(data{i},1),3);
    % Loop through and fit each end of month yield curve
    for j = 1:size(data{i},1)
        dnsScores{i}(j,:) = regress(data{i}(j,:)',dnsFactorLoadings);
    end;
    
    % subplot of mean corrected splines
    subplot(1,5,1,'Position',[0.05, 0.15, 0.15, 0.75]) 
    set(gca, 'ColorOrder', colorBarSettings);

    hold all;
    for j=1:STEP_SIZE_PRINTING:length(data{i});
        fnplt(splineData{i}(j),0.5)
    end;
    plot([min(MATURITIES) max(MATURITIES)],[0 0],'Color', GREY);
    hold off;
    axis([min(MATURITIES) max(MATURITIES) -1 4]);
    title(' Yield Curves');
    ylabel('Yield');
    set(gca,'XTick',[min(MATURITIES), 10, 20, 30]);
    set(gca,'XTickLabel',{num2str(min(MATURITIES)*12),'120','240','360'});
    set(gca,'YTick',[-1,0,1,2,3,4]);
    set(gca,'TickDir','out')
    set(gcf, 'Colormap', colorBarSettings);
    
    % subplot used for color bar
    subplot(1,5,2,'Position',[0.25, 0.15, 0.1, 0.75])
    axis off
    datesForLegend = datestr(dateVector{i}(1:SPACE_BETWEEN_LEGEND_LABELS:...
                                           length(data{i}),:),'mm/yyyy');
    v = (1:SPACE_BETWEEN_LEGEND_LABELS:length(data{i})) / length(data{i});
    set(gcf, 'Colormap', colorBarSettings);
    set(gca,'TickDir','out')
    colorbar('Ticks',v,'TickLabels',datesForLegend,'Location',...
             'westoutside','AxisLocation','out');
    
    % Plot Betas
    for m=1:3
        pos = [0.2*(m+1), 0.15, 0.15, 0.75];
        subplot(1,5,m+2,'Position',pos);   
        plot(zeros(length(dateVector{i}),1),'Color', GREY);
        hold on
        plot(dnsScores{i}(:,m),'k-');
        % plot level, slope, curvature
        if m==1
            levelFac = data{i}(:,21);
            plot(levelFac,'--','Color',[1 0.5 0],'LineWidth',0.3)
            yieldMax = max(abs(levelFac));
        elseif m==2
            slopeFac = -(data{i}(:,21)-data{i}(:,1));
            plot(slopeFac,'--','Color',[1 0.5 0],'LineWidth',0.3); 
            yieldMax = max(abs(slopeFac));
        else
            curvatureFac = 2*data{i}(:,7)-((data{i}(:,21)+data{i}(:,1)));
            plot(curvatureFac,'--','Color',[1 0.5 0],'LineWidth',0.3); 
            yieldMax = max(abs(curvatureFac));
        end;
        hold off; 
        [~,xi]        = ksdensity(dnsScores{i}(:,m));
        yScores       = max(max(-xi(1),xi(end)),yieldMax);
        dateSelection = [127, 388, 649]; %beginning of years
        xlim([0 716]);
        set(gca, 'XTick',dateSelection);
        set(gca, 'XTickLabel',...
            {datestr(dateVector{i}(dateSelection(1),:),'mmmyy'),...
             datestr(dateVector{i}(dateSelection(2),:),'mmmyy'),...
             datestr(dateVector{i}(dateSelection(3),:),'mmmyy')});
        ylim([-yScores yScores]);
        set(gca,'TickDir','out');
        title(strcat('Beta', num2str(m)));
        if m==1
            ylabel('Beta Scores');
        end;
    end;  
    set(findall(fig,'-property','FontSize'),'FontSize',FONT_SIZE)
    
    fileName = strcat('Figures\Models_simple\DNS_Betas_',...
                      COUNTRY_NAMES(i,:),'.png');
    print(fileName,'-dpng','-r400');   
    
    correlationsFactors(i,1) = corr(levelFac,dnsScores{i}(:,1));
    correlationsFactors(i,2) = corr(slopeFac,dnsScores{i}(:,2));
    correlationsFactors(i,3) = corr(curvatureFac,dnsScores{i}(:,3));
end;
xlswrite('Figures\Models_simple\DNS_Factor_Correlations.xls',...
    correlationsFactors)
save('Variables\model_dns.mat','dnsSaveCell','dnsScores');



%---------------------------------------------------------------------------
% PCA (multivariate) of separate datasets
%---------------------------------------------------------------------------
pcaScores   = cell(NUMBER_GROUPS,1);
pcaSaveCell = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS
    % calculate multivaritate PCs
    [pcaEigenvector, pcaScoresTemp, pcaEigenvalues] = ...
        pca(data{i},'Centered',true);
    % save PCs into pcaSaveCell, 1st row: explained variation, 
    % then: eigenvectors
    pcaScores{i}   = pcaScoresTemp(:,1:3);
    pcaSaveCell{i} = zeros(length(MATURITIES)+1,...
                           NUMBER_PRINCIPAL_COMPONENTS);
    for j=1:NUMBER_PRINCIPAL_COMPONENTS
        pcaSaveCell{i}(:,j) = [pcaEigenvalues(j)/sum(pcaEigenvalues); ...
                               pcaEigenvector(:,j)];
    end;
  
    % plot PCs
    fileName = strcat('Figures\Models\',num2str(NUMBER_GROUPS),'groups_',...
                      num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_PCA_',COUNTRY_NAMES(i,:),'.png');
    printPc(fig,i,pcaScores{i},pcaSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_PCA_',COUNTRY_NAMES(i,:),'.png');
    printPcSimple(fig,i,pcaScores{i},pcaSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      '_PCA_PC_',COUNTRY_NAMES(i,:),'.png');
    printPcOnly(fig,pcaSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      '_PCA_PCS_',COUNTRY_NAMES(i,:),'.png');
    printPcsOnly(fig,i,pcaScores{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'Expl_Variation_PCA.xls');
    xlswrite(fileName,pcaSaveCell{i}(1,:),COUNTRY_NAMES_LONG(i,:));
end;
save('Variables\model_pca.mat','pcaSaveCell','pcaScores');



%---------------------------------------------------------------------------
% PCA (multivariate) of combined dataset
%---------------------------------------------------------------------------
combPcaScores   = cell(NUMBER_GROUPS,1);
combPcaSaveCell = cell(NUMBER_GROUPS,1);

% call pcacov based on cov of data matrix to calculate the PCs
[combPcaEigenvector, combPcaEigenvalues, combPcaVariationExpl] = ...
    pcacov([covarianceMatrices{1}; covarianceMatrices{2}; ...
            covarianceMatrices{3}; covarianceMatrices{4}]);

for i=1:NUMBER_GROUPS
    % save PCs
    combPcaSaveCell{i} = zeros(length(MATURITIES)+1,...
                               NUMBER_PRINCIPAL_COMPONENTS);
    for j=1:NUMBER_PRINCIPAL_COMPONENTS
        combPcaSaveCell{i}(:,j) = [combPcaVariationExpl(j)/100; ...
                              combPcaEigenvector(:,j)];
    end;
 
    % Scores
    combPcaScores{i} = dataMeanCorrected{i}*combPcaEigenvector;

    % plot PCs
    fileName = strcat('Figures\Models\',num2str(NUMBER_GROUPS),'groups_',...
                      num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_PCA_comb_',COUNTRY_NAMES(i,:),'.png');
    printPc(fig,i,combPcaScores{i},combPcaSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_PCA_comb_',COUNTRY_NAMES(i,:),'.png');
    printPcSimple(fig,i,combPcaScores{i},combPcaSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      '_PCA_comb_PCS_',COUNTRY_NAMES(i,:),'.png');
    printPcsOnly(fig,i,combPcaScores{i},fileName);
end;

fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                  'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                  '_PCA_comb_PC.png');
printPcOnly(fig,combPcaSaveCell{1},fileName);

fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                  'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                  'Expl_Variation_PCA_comb.xls');
xlswrite(fileName,combPcaSaveCell{1}(1,:));

save('Variables\model_pca_comb.mat','combPcaSaveCell','combPcaScores');



%---------------------------------------------------------------------------
% CPC model (multivariate)
%---------------------------------------------------------------------------
% use the FG algorithm to calculate the common eigenvectors 
% of the covariance matrices
[B,cpcAlgorithmMethod,cpcAlgorithmDiff] = FG(covarianceMatrices,...
                                             nFgAlgorithm,PRESICION_FG_ALG);

%calculate the eigenvalues 
lambda        = cell(NUMBER_GROUPS,1);
for i=1:NUMBER_GROUPS;
    covar     = B'*covarianceMatrices{i}*B;
    lambda{i} = diag(covar);
end;

% sort eigenvectors
sumLambda        = zeros(size(lambda{1}));
for i=1:NUMBER_GROUPS;
    sumLambda    = sumLambda+lambda{i};
end;

[~,sortIndices]  = sort(sumLambda,'descend');
B1               = B(:,sortIndices);
cpcEigenvectors  = B1(:,1:NUMBER_PRINCIPAL_COMPONENTS);
sumEigenvectors  = sum(cpcEigenvectors);
cpcEigenvectors(:,sumEigenvectors < 0) ...
                 = - cpcEigenvectors(:,sumEigenvectors < 0);

cpcEigenvalues   = cell(NUMBER_GROUPS,1);
cpcVariationExpl = cell(NUMBER_GROUPS,1);
cpcSaveCell      = cell(NUMBER_GROUPS,1);
cpcScores        = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS;
    % sort eigenvalues and assign explained variation 
    cpcEigenvalues{i}       = lambda{i}(sortIndices);
    cpcVariationExpl{i}     = cpcEigenvalues{i}(1:...
        NUMBER_PRINCIPAL_COMPONENTS)./sum(cpcEigenvalues{i});
    
    % save results
    cpcSaveCell{i}          = zeros(length(MATURITIES)+1,...
                                    NUMBER_PRINCIPAL_COMPONENTS);
    for j=1:NUMBER_PRINCIPAL_COMPONENTS
        cpcSaveCell{i}(:,j) = [cpcVariationExpl{i}(j);cpcEigenvectors(:,j)];
    end;
    
    % calculate cpcScores
    cpcScores{i} = dataMeanCorrected{i}*cpcEigenvectors;
    
    %plot CPCs
    fileName = strcat('Figures\Models\',num2str(NUMBER_GROUPS),'groups_',...
                      num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_CPC_',COUNTRY_NAMES(i,:),'.png');
    printPc(fig,i,cpcScores{i},cpcSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_CPC_',COUNTRY_NAMES(i,:),'.png');
    printPcSimple(fig,i,cpcScores{i},cpcSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      '_CPC_PCS_',COUNTRY_NAMES(i,:),'.png');
    printPcsOnly(fig,i,cpcScores{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                  'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                  'Expl_Variation_CPC.xls');
    xlswrite(fileName,cpcSaveCell{i}(1,:),COUNTRY_NAMES_LONG(i,:));
end;

fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                  'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                  '_CPC_PC.png');
printPcOnly(fig,cpcSaveCell{1},fileName); 

save('Variables\model_cpc.mat','cpcSaveCell','cpcScores',...
     'cpcAlgorithmMethod');


 
%---------------------------------------------------------------------------
% FPCAs of separate data sets using Spline Basis
%---------------------------------------------------------------------------
fpcaScores   = cell(NUMBER_GROUPS,1);
fpcaSaveCell = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS
    % calculate functional PCs
    [fpcaEigenvectors, fpcaEigenvalues, fpcaVariationExpl, ...
    fpcaScores{i}] = fpca(dataMeanCorrected{i});

    fpcaSaveCell{i} = zeros(length(MATURITIES)+1,...
                            NUMBER_PRINCIPAL_COMPONENTS);
    for j=1:NUMBER_PRINCIPAL_COMPONENTS
        fpcaSaveCell{i}(:,j) = [fpcaVariationExpl(j);fpcaEigenvectors(j,:)'];
    end;
    
    fileName = strcat('Figures\Models\',num2str(NUMBER_GROUPS),'groups_',...
                    num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                    'PCs_FPCA_',COUNTRY_NAMES(i,:),'.png');
    printPc(fig,i,fpcaScores{i},fpcaSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_FPCA_',COUNTRY_NAMES(i,:),'.png');
    printPcSimple(fig,i,fpcaScores{i},fpcaSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      '_FPCA_PC_',COUNTRY_NAMES(i,:),'.png');
    printPcOnly(fig,fpcaSaveCell{i},fileName); 
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      '_FPCA_PCS_',COUNTRY_NAMES(i,:),'.png');
    printPcsOnly(fig,i,fpcaScores{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'Expl_Variation_FPCA.xls');
    xlswrite(fileName,fpcaSaveCell{i}(1,:),COUNTRY_NAMES_LONG(i,:));
end;
save('Variables\model_fpca.mat','fpcaSaveCell','fpcaScores');



%---------------------------------------------------------------------------
% FPCA of combined dataset using Spline Basis
%---------------------------------------------------------------------------
[combFpcaEigenvectors, combFpcaEigenvalues, combFpcaVariationExpl, ...
    combFpcaScoresTemp] = fpca([dataMeanCorrected{1};dataMeanCorrected{2};...
                            dataMeanCorrected{3};dataMeanCorrected{4}]);
%save PCs
combFpcaSaveCell   = cell(NUMBER_GROUPS,1);
combFpcaScores     = cell(NUMBER_GROUPS,1);

combFpcaSaveArray  = zeros(length(MATURITIES)+1,NUMBER_PRINCIPAL_COMPONENTS);
for j=1:NUMBER_PRINCIPAL_COMPONENTS
   combFpcaSaveArray(:,j) = [combFpcaVariationExpl(j);...
                             combFpcaEigenvectors(j,:)'];
end

%plot FPCs
for i=1:NUMBER_GROUPS
    combFpcaSaveCell{i}   = combFpcaSaveArray;
    combFpcaScores{i}     = combFpcaScoresTemp((i-1)*length(data{i})+1:...
                                               i*length(data{i}),:);
    fileName = strcat('Figures\Models\',num2str(NUMBER_GROUPS),'groups_',...
                      num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_FPCA_comb_',COUNTRY_NAMES(i,:),'.png');
    printPc(fig,i,combFpcaScores{i},combFpcaSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_FPCA_comb_',COUNTRY_NAMES(i,:),'.png');
    printPcSimple(fig,i,combFpcaScores{i},combFpcaSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      '_FPCA_comb_PCS_',COUNTRY_NAMES(i,:),'.png');
    printPcsOnly(fig,i,combFpcaScores{i},fileName);
end;

fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
               'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
               '_FPCA_comb_PC.png');
printPcOnly(fig,combFpcaSaveCell{1},fileName); 

fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
               'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
               'Expl_Variation_FPCA_comb.xls');
xlswrite(fileName,combFpcaSaveCell{1}(1,:));
save('Variables\model_fpca_comb.mat','combFpcaSaveCell','combFpcaScores');
   
 

%---------------------------------------------------------------------------
% CFPCA using Spline Basis
%---------------------------------------------------------------------------
% calculation of the Gram matrix wMatrix
wMatrix     = Gram(KNOT_SEQUENCE,SPLINE_ORDER);
wMatrixRoot = chol(wMatrix);

% calculation of coefficient matrices
coefMatrix  = cell(NUMBER_GROUPS,1);
eigenMatrix = cell(NUMBER_GROUPS,1);
for i=1:NUMBER_GROUPS
    % estimation of the coefficient matrix using least square B-splines
    coefMatrix{i}          = zeros(length(dataMeanCorrected{i}),...
                                   length(MATURITIES));
    for j=1:size(dataMeanCorrected{i},1)
        coefMatrix{i}(j,:) = fnbrk(splineDataMeanCorrected{i}(j),'c');
    end;       
    
    % preparing eigenanalysis
    eigenMatrix{i} = wMatrixRoot*(coefMatrix{i}'*coefMatrix{i}./...
                     (size(coefMatrix{i},1)+1))*wMatrixRoot';
    eigenMatrix{i} = (eigenMatrix{i} + eigenMatrix{i}')./2;
end;

%use the FG algorithm to calculate the common eigenvectors of eigenMatrix
[B,cfpcAlgorithmMethod,cfpcAlgorithmDiff] = FG(eigenMatrix,nFgAlgorithm,...
                                               PRESICION_FG_ALG);

% calculate the eigenvalues 
lambda        = cell(NUMBER_GROUPS,1);
for i=1:NUMBER_GROUPS;
    covar     = B'*eigenMatrix{i}*B;
    lambda{i} = diag(covar);
end;

% sort eigenvectors
sumLambda        = zeros(size(lambda{1}));
for i=1:NUMBER_GROUPS;
    sumLambda    = sumLambda+lambda{i};
end;
[~,sortIndices]      = sort(sumLambda,'descend');
B1                   = B(:,sortIndices);
cfpcEigenvectorsU = B1(:,1:NUMBER_PRINCIPAL_COMPONENTS);
sumEigenvectorsU      = sum(cfpcEigenvectorsU);
cfpcEigenvectorsU(:,sumEigenvectorsU < 0) ...
                     = - cfpcEigenvectorsU(:,sumEigenvectorsU < 0);

cfpcEigenvalues      = cell(NUMBER_GROUPS,1);
cfpcVariationExpl    = cell(NUMBER_GROUPS,1);
cfpcSaveCell         = cell(NUMBER_GROUPS,1);
cfpcScores           = cell(NUMBER_GROUPS,1);

cfpcEigenvectorsTemp = wMatrixRoot\cfpcEigenvectorsU;
cfpcEigenvectors     = zeros(length(MATURITIES),NUMBER_PRINCIPAL_COMPONENTS);
                             
for j=1:NUMBER_PRINCIPAL_COMPONENTS
    cfpcEigenvectors(:,j) = fnval(spmak(KNOT_SEQUENCE,...
                                  cfpcEigenvectorsTemp(:,j)'),MATURITIES);
end;
                                   
for i=1:NUMBER_GROUPS;
    % sort eigenvalues and assign explained variation 
    cfpcEigenvalues{i}       = lambda{i}(sortIndices);
    cfpcVariationExpl{i}     = cfpcEigenvalues{i}(1:...
        NUMBER_PRINCIPAL_COMPONENTS)./sum(cfpcEigenvalues{i});

    % calculate the scores and the eigenvectors
    cfpcScores{i}        = coefMatrix{i}*wMatrixRoot'*cfpcEigenvectorsU; 
    cfpcSaveCell{i}      = zeros(length(MATURITIES)+1,...
                                 NUMBER_PRINCIPAL_COMPONENTS);
    for j=1:NUMBER_PRINCIPAL_COMPONENTS
       % save results
       cfpcSaveCell{i}(:,j)  = [cfpcVariationExpl{i}(j);...
                                cfpcEigenvectors(:,j)];
    end;
    
    %plot CFPCs
    fileName = strcat('Figures\Models\',num2str(NUMBER_GROUPS),'groups_',...
                      num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_CFPC_',COUNTRY_NAMES(i,:),'.png');
    printPc(fig,i,cfpcScores{i},cfpcSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'PCs_CFPC_',COUNTRY_NAMES(i,:),'.png');
    printPcSimple(fig,i,cfpcScores{i},cfpcSaveCell{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      '_CFPC_PCS_',COUNTRY_NAMES(i,:),'.png');
    printPcsOnly(fig,i,cfpcScores{i},fileName);
    
    fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'Expl_Variation_CFPC.xls');
    xlswrite(fileName,cfpcSaveCell{i}(1,:),COUNTRY_NAMES_LONG(i,:));
end;

fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                  'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                  '_CFPC_PC.png');
printPcOnly(fig,cfpcSaveCell{1},fileName); 

save('Variables\model_cfpc.mat','cfpcSaveCell','cfpcScores',...
     'cfpcAlgorithmMethod');
close;
% save(strcat('Variables\4groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
%     'PC_models_all.mat'));



%---------------------------------------------------------------------------
% Print summary
%---------------------------------------------------------------------------
% define size of output
width=16; % centimeters
height=6; % centimeters

fig=figure;
fig.PaperUnits              = 'centimeters';
fig.PaperPosition           = [0,0 width, height];
fig.PaperPositionMode       = 'manual';
CURVE_WIDTH_SUMMARY         = 0.5;   % width of line in prints
% subplot PC spline
for j=1:NUMBER_PRINCIPAL_COMPONENTS
    subplot(1,NUMBER_PRINCIPAL_COMPONENTS,j);
    plot([min(MATURITIES) max(MATURITIES)],[0 0],'Color',GREY);
    hold on;
    splinePc=spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                   dnsSaveCell{1}(2:end,j));
    fnplt(splinePc,'b--', CURVE_WIDTH_SUMMARY);
    
    splinePc=spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                   cpcSaveCell{1}(2:end,j));
    fnplt(splinePc,'r--', CURVE_WIDTH_SUMMARY);
    
    splinePc=spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                   combFpcaSaveCell{1}(2:end,j));
    fnplt(splinePc,'g--', CURVE_WIDTH_SUMMARY);
    
    splinePc=spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                   cfpcSaveCell{1}(2:end,j));
    fnplt(splinePc,'k-', CURVE_WIDTH_SUMMARY);
    hold off;
    
    axis([min(MATURITIES) max(MATURITIES) -1.5 1.5]);
    set(gca,'XTick',[min(MATURITIES), 10, 20, 30]);
    set(gca,'XTickLabel',{num2str(min(MATURITIES)*12),'120','240','360'});
    set(gca,'YTick',[-1,0,1]);
    set(gca,'TickDir','out');      
    title(strcat('PC/Factor',num2str(j)));             
end;     
fileName = strcat('Figures\Models_simple\',num2str(NUMBER_GROUPS),...
                      'groups_',num2str(NUMBER_PRINCIPAL_COMPONENTS),...
                      'Summary_PC_Factors.png');
print(fileName,'-dpng','-r400');        
close;