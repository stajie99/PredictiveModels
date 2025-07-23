%FORECASTAR1 runs AR(1) forecasts for the DNS, CPC, combFPCA, CFPC for 1-
%day ahead and 5-days-ahead and saves the results to an excel file as well 
%as prints the forecasted scores
%
%  provides descriptive statistics of factors and scores
%  forecasts for DNS factors
%  forecasts for CPC scores
%  forecasts for combFPCA scores
%  forecasts for CFPC scores

% load and define variables
global data FONT_SIZE dateVector 
global NUMBER_GROUPS COUNTRY_NAMES_LONG

load('Variables\inputs.mat');
load('Variables\model_dns.mat');
load('Variables\model_cpc.mat');
load('Variables\model_fpca_comb.mat');
load('Variables\model_cfpc.mat');
addpath('Library') ;

COUNTRY_NAMES_LONG = ['USDEFFR'; 'SONIA  '; 'EONIA  '; 'TONAR  '];
LAMBDA_T           = 0.0609;
GREY               = [0.7 0.7 0.7];

NUMBER_GROUPS      = length(data);
fig                = figure;

NUMBER_TRAINING_DATES   = 455;
NUMBER_PREDICTION_DATES = length(data{1})-NUMBER_TRAINING_DATES;
AHEAD_SHORT             = 1;
AHEAD_LONG              = 5;



% --------------------------------------------------------------------------
% Descriptive statistics of factors and scores
% --------------------------------------------------------------------------    
scores       = cell(4,1);
scores{1}    = dnsScores;
scores{2}    = cpcScores;
scores{3}    = combFpcaScores;
scores{4}    = cfpcScores;
namesFactors = ['DNS  '; 'CPC  '; 'cFPCA'; 'CFPC '];

scoresDescriptive   = cell(length(scores),1);

for j=1:length(scores) 
    for i=1:NUMBER_GROUPS
    scoresDescriptive{i,j} = zeros(NUMBER_GROUPS,size(scores{j}{i},2)*5);
         for m=1:size(scores{j}{i},2)
            scoresDescriptive{j}(i,(m-1)*5+1) = mean(scores{j}{i}(:,m));
            scoresDescriptive{j}(i,(m-1)*5+2) = std(scores{j}{i}(:,m));
            autoCorrelationTemp  = xcorr(scores{j}{i}(:,m),....
                                   AHEAD_SHORT,'coeff');
            scoresDescriptive{j}(i,(m-1)*5+3) = autoCorrelationTemp(end);
            clearvars autoCorrelationTemp
            autoCorrelationTemp  = xcorr(scores{j}{i}(:,m),....
                                   AHEAD_LONG,'coeff');
            scoresDescriptive{j}(i,(m-1)*5+4) = autoCorrelationTemp(end);
            clearvars autoCorrelationTemp
         end;
    end;
    xlswrite('Figures\Forecast\scoresDescriptives.xls',...
             scoresDescriptive{j},namesFactors(j,:));
end;
clearvars scores namesFactors



% -------------------------------------------------------------------------
% Forecast for DNS factors
% -------------------------------------------------------------------------           
% AHEAD_SHORT
dnsPredScores = cell(NUMBER_GROUPS,1);
dnsPredCurves = cell(NUMBER_GROUPS,1);
dnsPredModels = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS
    dnsPredScores{i} = zeros(length(data{i}),NUMBER_PRINCIPAL_COMPONENTS);
    dnsPredCurves{i} = zeros(size(data{i}));
    dnsPredModels{i} = zeros(length(data{i}),NUMBER_PRINCIPAL_COMPONENTS,3);
    
    for n=1:NUMBER_PRINCIPAL_COMPONENTS
        for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_SHORT  
            x = dnsScores{i}(1:s-1,n);
            y = dnsScores{i}(2:s,n);
            dnsPredModels{i}(s,n,:)           = localLogLikeEst(x,y);
            dnsPredScores{i}(s+AHEAD_SHORT,n) = dnsPredModels{i}(s,n,1)+...
                           dnsPredModels{i}(s,n,2) *dnsScores{i}(s,n);  
        end;
    end;
    
    % calculate predicted yield for given maturities
    for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_SHORT
        dnsPredCurves{i}(s+AHEAD_SHORT,:) = dnsPredScores{i}...
            (s+AHEAD_SHORT,:)*dnsSaveCell{i}(2:end,:)';
    end;
end;

fileNameDetail = strcat('pred',num2str(NUMBER_PREDICTION_DATES),...
                        '_ahead',num2str(AHEAD_SHORT),'_DNS');
[dnsPredRmse,dnsPredMad] = arErrorMeasuresDns(dnsSaveCell,...
                              dnsScores,dnsPredScores,dnsPredCurves,...
                              NUMBER_TRAINING_DATES,AHEAD_SHORT);
                          
% export error measures                    
fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_Errors.xls');               
xlswrite(fileName,dnsPredRmse{1},'RMS_Scores');
xlswrite(fileName,dnsPredRmse{2},'RMS_Maturities');
xlswrite(fileName,dnsPredRmse{3},'RMS_Curves');  
xlswrite(fileName,dnsPredMad{1},'MAD_Scores');
xlswrite(fileName,dnsPredMad{2},'MAD_Maturities'); 
xlswrite(fileName,dnsPredMad{3},'MAD_Curves');  

% plot results
for i=1:NUMBER_GROUPS
    fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_',...
                      COUNTRY_NAMES(i,:),'.png');
    printForecast(fig,dnsScores,dnsPredScores,i,fileName,...
                  NUMBER_TRAINING_DATES,AHEAD_SHORT,1);
end;

% save results
fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
save(fileName,'dnsPredModels','dnsPredScores','dnsPredCurves',...
              'dnsPredRmse','dnsPredMad');

          
          
% AHEAD_LONG
dnsPredAheadScores = cell(NUMBER_GROUPS,1);
dnsPredAheadCurves = cell(NUMBER_GROUPS,1);
dnsPredAheadModels = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS
    dnsPredAheadScores{i} = zeros(length(data{i}),...
                                  NUMBER_PRINCIPAL_COMPONENTS);
    dnsPredAheadCurves{i} = zeros(size(data{i}));
    dnsPredAheadModels{i} = zeros(length(data{i}),...
                                  NUMBER_PRINCIPAL_COMPONENTS,3);
    
    for n=1:NUMBER_PRINCIPAL_COMPONENTS
        for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_LONG
            x = dnsScores{i}(1:s-AHEAD_LONG,n);
            y = dnsScores{i}(1+AHEAD_LONG:s,n);
            dnsPredAheadModels{i}(s,n,:)          = localLogLikeEst(x,y);
            dnsPredAheadScores{i}(s+AHEAD_LONG,n) = dnsPredAheadModels{i}...
                (s,n,1)+dnsPredAheadModels{i}(s,n,2)*dnsScores{i}(s,n);  
        end;
    end;
    
    % calculate predicted yield for given maturities
    for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_LONG
        dnsPredAheadCurves{i}(s+AHEAD_LONG,:) = dnsPredAheadScores{i}...
            (s+AHEAD_LONG,:)*dnsSaveCell{i}(2:end,:)';
    end;
end;
                
fileNameDetail = strcat('pred',num2str(NUMBER_PREDICTION_DATES),...
                        '_ahead',num2str(AHEAD_LONG),'_DNS');
% fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
% load(fileName);

[dnsPredAheadRmse,dnsPredAheadMad] = arErrorMeasuresDns(dnsSaveCell,...
                        dnsScores,dnsPredAheadScores,dnsPredAheadCurves,...
                        NUMBER_TRAINING_DATES,AHEAD_LONG);

% export error measures                        
fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_Errors.xls');               
xlswrite(fileName,dnsPredAheadRmse{1},'RMS_Scores');
xlswrite(fileName,dnsPredAheadRmse{2},'RMS_Maturities');
xlswrite(fileName,dnsPredAheadRmse{3},'RMS_Curves');  
xlswrite(fileName,dnsPredAheadMad{1},'MAD_Scores');
xlswrite(fileName,dnsPredAheadMad{2},'MAD_Maturities'); 
xlswrite(fileName,dnsPredAheadMad{3},'MAD_Curves');  

% plot results
for i=1:NUMBER_GROUPS
    fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_',...
                      COUNTRY_NAMES(i,:),'.png');
    printForecast(fig,dnsScores,dnsPredAheadScores,i,fileName,...
                  NUMBER_TRAINING_DATES,AHEAD_LONG,1);
end;

% save results
fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
save(fileName,'dnsPredAheadModels','dnsPredAheadScores',...
              'dnsPredAheadCurves','dnsPredAheadRmse','dnsPredAheadMad');


% -------------------------------------------------------------------------
% Scores forecast for CPC
% -------------------------------------------------------------------------            
% AHEAD_SHORT
cpcPredScores = cell(NUMBER_GROUPS,1);
cpcPredCurves = cell(NUMBER_GROUPS,1);
cpcPredModels = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS
    cpcPredScores{i} = zeros(length(data{i}),NUMBER_PRINCIPAL_COMPONENTS);
    cpcPredCurves{i} = zeros(size(data{i}));
    cpcPredModels{i} = zeros(length(data{i}),NUMBER_PRINCIPAL_COMPONENTS,3);
    
    for n=1:NUMBER_PRINCIPAL_COMPONENTS
        for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_SHORT  
            x = cpcScores{i}(1:s-1,n);
            y = cpcScores{i}(2:s,n);
            cpcPredModels{i}(s,n,:)           = localLogLikeEst(x,y);
            cpcPredScores{i}(s+AHEAD_SHORT,n) = cpcPredModels{i}(s,n,1)+...
                           cpcPredModels{i}(s,n,2)*cpcScores{i}(s,n);  
        end;
    end;
    
    % calculate predicted yield for given maturities
    for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_SHORT
        cpcPredCurves{i}(s+AHEAD_SHORT,:) = cpcPredScores{i}...
            (s+AHEAD_SHORT,:)*cpcSaveCell{i}(2:end,:)'+mean(data{i});
    end;
end;

fileNameDetail = strcat('pred',num2str(NUMBER_PREDICTION_DATES),...
                        '_ahead',num2str(AHEAD_SHORT),'_CPC');
% fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
% load(fileName);

[cpcPredRmse,cpcPredMad] = arErrorMeasures(cpcSaveCell,...
    cpcScores,cpcPredScores,cpcPredCurves,...
    NUMBER_TRAINING_DATES,AHEAD_SHORT);
      
% export error measures                    
fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_Errors.xls');               
xlswrite(fileName,cpcPredRmse{1},'RMS_Scores');
xlswrite(fileName,cpcPredRmse{2},'RMS_Maturities');
xlswrite(fileName,cpcPredRmse{3},'RMS_Curves');  
xlswrite(fileName,cpcPredMad{1},'MAD_Scores');
xlswrite(fileName,cpcPredMad{2},'MAD_Maturities'); 
xlswrite(fileName,cpcPredMad{3},'MAD_Curves');  

% plot results
for i=1:NUMBER_GROUPS
    fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_',...
                      COUNTRY_NAMES(i,:),'.png');
    printForecast(fig,cpcScores,cpcPredScores,i,fileName,...
                  NUMBER_TRAINING_DATES,AHEAD_SHORT,0);
end;

% save results
fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
save(fileName,'cpcPredModels','cpcPredScores','cpcPredCurves',...
              'cpcPredRmse','cpcPredMad');
     
          
          
% AHEAD_LONG
cpcPredAheadScores = cell(NUMBER_GROUPS,1);
cpcPredAheadCurves = cell(NUMBER_GROUPS,1);
cpcPredAheadModels = cell(NUMBER_GROUPS,1); 

for i=1:NUMBER_GROUPS
    cpcPredAheadScores{i} = zeros(length(data{i}),...
                                  NUMBER_PRINCIPAL_COMPONENTS);
    cpcPredAheadCurves{i} = zeros(size(data{i}));
    cpcPredAheadModels{i} = zeros(length(data{i}),...
                                  NUMBER_PRINCIPAL_COMPONENTS,3);
    
    for n=1:NUMBER_PRINCIPAL_COMPONENTS
        for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_LONG
            x = cpcScores{i}(1:s-AHEAD_LONG,n);
            y = cpcScores{i}(1+AHEAD_LONG:s,n);
            cpcPredAheadModels{i}(s,n,:)          = localLogLikeEst(x,y);
            cpcPredAheadScores{i}(s+AHEAD_LONG,n) = cpcPredAheadModels{i}...
                (s,n,1)+cpcPredAheadModels{i}(s,n,2)*cpcScores{i}(s,n);  
        end;
    end;
    
    % calculate predicted yield for given maturities
    for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_LONG
        cpcPredAheadCurves{i}(s+AHEAD_LONG,:) = cpcPredAheadScores{i}...
            (s+AHEAD_LONG,:)*cpcSaveCell{i}(2:end,:)'+mean(data{i});
    end;
end;                            
                
fileNameDetail = strcat('pred',num2str(NUMBER_PREDICTION_DATES),...
                        '_ahead',num2str(AHEAD_LONG),'_CPC');
% fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
% load(fileName);

[cpcPredAheadRmse,cpcPredAheadMad] = arErrorMeasures(cpcSaveCell,...
    cpcScores,cpcPredAheadScores,cpcPredAheadCurves,...
    NUMBER_TRAINING_DATES,AHEAD_LONG);


% export error measures                        
fileName       = strcat('Figures\Forecast_fast\',fileNameDetail,'_Errors.xls');               
xlswrite(fileName,cpcPredAheadRmse{1},'RMS_Scores');
xlswrite(fileName,cpcPredAheadRmse{2},'RMS_Maturities');
xlswrite(fileName,cpcPredAheadRmse{3},'RMS_Curves');  
xlswrite(fileName,cpcPredAheadMad{1},'MAD_Scores');
xlswrite(fileName,cpcPredAheadMad{2},'MAD_Maturities'); 
xlswrite(fileName,cpcPredAheadMad{3},'MAD_Curves');  

% plot results
for i=1:NUMBER_GROUPS
    fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_',...
                      COUNTRY_NAMES(i,:),'.png');
    printForecast(fig,cpcScores,cpcPredAheadScores,i,fileName,...
                  NUMBER_TRAINING_DATES,AHEAD_LONG,0);
end;

% save results
fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
save(fileName,'cpcPredAheadModels','cpcPredAheadScores',...
              'cpcPredAheadCurves','cpcPredAheadRmse','cpcPredAheadMad');


          
% --------------------------------------------------------------------------
% Scores forecast for FPCA comb
% --------------------------------------------------------------------------                         
% AHEAD_SHORT
combFpcaPredScores = cell(NUMBER_GROUPS,1);
combFpcaPredCurves = cell(NUMBER_GROUPS,1);
combFpcaPredModels = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS
    combFpcaPredScores{i} = zeros(length(data{i}),...
                                  NUMBER_PRINCIPAL_COMPONENTS);
    combFpcaPredCurves{i} = zeros(size(data{i}));
    combFpcaPredModels{i} = zeros(length(data{i}),...
                                  NUMBER_PRINCIPAL_COMPONENTS,3);
    
    for n=1:NUMBER_PRINCIPAL_COMPONENTS
        for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_SHORT  
            x = combFpcaScores{i}(1:s-1,n);
            y = combFpcaScores{i}(2:s,n);
            combFpcaPredModels{i}(s,n,:)           = localLogLikeEst(x,y);
            combFpcaPredScores{i}(s+AHEAD_SHORT,n) = ...
                combFpcaPredModels{i}(s,n,1)+combFpcaPredModels{i}(s,n,2)...
                *combFpcaScores{i}(s,n);  
        end;
    end;
    
    % calculate predicted yield for given maturities
    for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_SHORT
        combFpcaPredCurves{i}(s+AHEAD_SHORT,:) = combFpcaPredScores{i}...
            (s+AHEAD_SHORT,:)*combFpcaSaveCell{i}(2:end,:)'+mean(data{i});
    end;
end;                   

fileNameDetail = strcat('pred',num2str(NUMBER_PREDICTION_DATES),...
                        '_ahead',num2str(AHEAD_SHORT),'_FPCA_comb');
% fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');                    
% load(fileName);

[combFpcaPredRmse,combFpcaPredMad] = arErrorMeasures(combFpcaSaveCell,...
    combFpcaScores,combFpcaPredScores,combFpcaPredCurves,...
    NUMBER_TRAINING_DATES,AHEAD_SHORT);

% export error measures                    
fileName       = strcat('Figures\Forecast_fast\',fileNameDetail,'_Errors.xls');               
xlswrite(fileName,combFpcaPredRmse{1},'RMS_Scores');
xlswrite(fileName,combFpcaPredRmse{2},'RMS_Maturities');
xlswrite(fileName,combFpcaPredRmse{3},'RMS_Curves');  
xlswrite(fileName,combFpcaPredMad{1},'MAD_Scores');
xlswrite(fileName,combFpcaPredMad{2},'MAD_Maturities'); 
xlswrite(fileName,combFpcaPredMad{3},'MAD_Curves');  

% plot results
for i=1:NUMBER_GROUPS
    fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_',...
                      COUNTRY_NAMES(i,:),'.png');
    printForecast(fig,combFpcaScores,combFpcaPredScores,i,fileName,...
                  NUMBER_TRAINING_DATES,AHEAD_SHORT,0);
end;

% save results
fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
save(fileName,'combFpcaPredModels','combFpcaPredScores',...
              'combFpcaPredCurves','combFpcaPredRmse','combFpcaPredMad');


          
% AHEAD_LONG
combFpcaPredAheadScores = cell(NUMBER_GROUPS,1);
combFpcaPredAheadCurves = cell(NUMBER_GROUPS,1);
combFpcaPredAheadModels = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS
    combFpcaPredAheadScores{i} = zeros(length(data{i}),...
                                       NUMBER_PRINCIPAL_COMPONENTS);
    combFpcaPredAheadCurves{i} = zeros(size(data{i}));
    combFpcaPredAheadModels{i} = zeros(length(data{i}),...
                                       NUMBER_PRINCIPAL_COMPONENTS,3);
    
    for n=1:NUMBER_PRINCIPAL_COMPONENTS
        for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_LONG
            x              = combFpcaScores{i}(1:s-AHEAD_LONG,n);
            y              = combFpcaScores{i}(1+AHEAD_LONG:s,n);
            combFpcaPredAheadModels{i}(s,n,:)          = ...
                localLogLikeEst(x,y);
            combFpcaPredAheadScores{i}(s+AHEAD_LONG,n) = ...
                combFpcaPredAheadModels{i}(s,n,1)+...
                combFpcaPredAheadModels{i}(s,n,2)*combFpcaScores{i}(s,n);  
        end;
    end;
    
    % calculate predicted yield for given maturities
    for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_LONG
        combFpcaPredAheadCurves{i}(s+AHEAD_LONG,:) = ...
            combFpcaPredAheadScores{i}(s+AHEAD_LONG,:)*...
            combFpcaSaveCell{i}(2:end,:)'+mean(data{i});
    end;
end;
                
fileNameDetail = strcat('pred',num2str(NUMBER_PREDICTION_DATES),...
                        '_ahead',num2str(AHEAD_LONG),'_FPCA_comb');
% fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
% load(fileName);

[combFpcaPredAheadRmse,combFpcaPredAheadMad] = arErrorMeasures(...
    combFpcaSaveCell,combFpcaScores,combFpcaPredAheadScores,...
    combFpcaPredAheadCurves,NUMBER_TRAINING_DATES,AHEAD_LONG);

% export error measures                        
fileName       = strcat('Figures\Forecast_fast\',fileNameDetail,...
                        '_Errors.xls');               
xlswrite(fileName,combFpcaPredAheadRmse{1},'RMS_Scores');
xlswrite(fileName,combFpcaPredAheadRmse{2},'RMS_Maturities');
xlswrite(fileName,combFpcaPredAheadRmse{3},'RMS_Curves');  
xlswrite(fileName,combFpcaPredAheadMad{1},'MAD_Scores');
xlswrite(fileName,combFpcaPredAheadMad{2},'MAD_Maturities'); 
xlswrite(fileName,combFpcaPredAheadMad{3},'MAD_Curves');  

% plot results
for i=1:NUMBER_GROUPS
    fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_',...
                      COUNTRY_NAMES(i,:),'.png');
    printForecast(fig,combFpcaScores,combFpcaPredAheadScores,i,fileName,...
                            NUMBER_TRAINING_DATES,AHEAD_LONG,0);
end;

% save results
fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
save(fileName,'combFpcaPredAheadModels','combFpcaPredAheadScores',...
              'combFpcaPredAheadCurves','combFpcaPredAheadRmse',...
              'combFpcaPredAheadMad');
          


% --------------------------------------------------------------------------
% Scores forecast for CFPC
% --------------------------------------------------------------------------            
% AHEAD_SHORT
cfpcPredScores = cell(NUMBER_GROUPS,1);
cfpcPredCurves = cell(NUMBER_GROUPS,1);
cfpcPredModels = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS
    cfpcPredScores{i} = zeros(length(data{i}),NUMBER_PRINCIPAL_COMPONENTS);
    cfpcPredCurves{i} = zeros(size(data{i}));
    cfpcPredModels{i} = zeros(length(data{i}),...
                              NUMBER_PRINCIPAL_COMPONENTS,3);
    
    for n=1:NUMBER_PRINCIPAL_COMPONENTS
        for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_SHORT  
            x = cfpcScores{i}(1:s-1,n);
            y = cfpcScores{i}(2:s,n);
            cfpcPredModels{i}(s,n,:)           = localLogLikeEst(x,y);
            cfpcPredScores{i}(s+AHEAD_SHORT,n) = cfpcPredModels{i}(s,n,1)...
                +cfpcPredModels{i}(s,n,2)*cfpcScores{i}(s,n);  
        end;
    end;
    
    % calculate predicted yield for given maturities
    for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_SHORT
        cfpcPredCurves{i}(s+AHEAD_SHORT,:) = cfpcPredScores{i}...
            (s+AHEAD_SHORT,:)*cfpcSaveCell{i}(2:end,:)'+mean(data{i});
    end;
end;

fileNameDetail = strcat('pred',num2str(NUMBER_PREDICTION_DATES),...
                        '_ahead',num2str(AHEAD_SHORT),'_CFPC');
% fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
% load(fileName);

[cfpcPredRmse,cfpcPredMad] = arErrorMeasures(cfpcSaveCell,cfpcScores,...
    cfpcPredScores,cfpcPredCurves,NUMBER_TRAINING_DATES,...
    AHEAD_SHORT);

% export error measures                    
fileName       = strcat('Figures\Forecast_fast\',fileNameDetail,'_Errors.xls');               
xlswrite(fileName,cfpcPredRmse{1},'RMS_Scores');
xlswrite(fileName,cfpcPredRmse{2},'RMS_Maturities');
xlswrite(fileName,cfpcPredRmse{3},'RMS_Curves');  
xlswrite(fileName,cfpcPredMad{1},'MAD_Scores');
xlswrite(fileName,cfpcPredMad{2},'MAD_Maturities'); 
xlswrite(fileName,cfpcPredMad{3},'MAD_Curves');  

% plot results
for i=1:NUMBER_GROUPS
    fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_',...
                      COUNTRY_NAMES(i,:),'.png');
    printForecast(fig,cfpcScores,cfpcPredScores,i,fileName,...
                            NUMBER_TRAINING_DATES,AHEAD_SHORT,0);
end;

% save results
fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
save(fileName,'cfpcPredModels','cfpcPredScores','cfpcPredCurves',...
              'cfpcPredRmse','cfpcPredMad');


          
% AHEAD_LONG
cfpcPredAheadScores = cell(NUMBER_GROUPS,1);
cfpcPredAheadCurves = cell(NUMBER_GROUPS,1);
cfpcPredAheadModels = cell(NUMBER_GROUPS,1);

for i=1:NUMBER_GROUPS
    cfpcPredAheadScores{i} = zeros(length(data{i}),...
                                   NUMBER_PRINCIPAL_COMPONENTS);
    cfpcPredAheadCurves{i} = zeros(size(data{i}));
    cfpcPredAheadModels{i} = zeros(length(data{i}),...
                                   NUMBER_PRINCIPAL_COMPONENTS,3);
    
    for n=1:NUMBER_PRINCIPAL_COMPONENTS
        for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_LONG
            x = cfpcScores{i}(1:s-AHEAD_LONG,n);
            y = cfpcScores{i}(1+AHEAD_LONG:s,n);
            cfpcPredAheadModels{i}(s,n,:)          = localLogLikeEst(x,y);
            cfpcPredAheadScores{i}(s+AHEAD_LONG,n) = ...
                cfpcPredAheadModels{i}(s,n,1)+...
                cfpcPredAheadModels{i}(s,n,2)*cfpcScores{i}(s,n);  
        end;
    end;
    
    % calculate predicted yield for given maturities
    for s=NUMBER_TRAINING_DATES:length(data{i})-AHEAD_LONG
        cfpcPredAheadCurves{i}(s+AHEAD_LONG,:) = cfpcPredAheadScores{i}...
            (s+AHEAD_LONG,:)*cfpcSaveCell{i}(2:end,:)'+mean(data{i});
    end;
end;      
                 
fileNameDetail = strcat('pred',num2str(NUMBER_PREDICTION_DATES),...
                        '_ahead',num2str(AHEAD_LONG),'_CFPC');
% fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
% load(fileName);

[cfpcPredAheadRmse,cfpcPredAheadMad] = arErrorMeasures(cfpcSaveCell,...
    cfpcScores,cfpcPredAheadScores,cfpcPredAheadCurves,...
    NUMBER_TRAINING_DATES,AHEAD_LONG);

% export error measures                        
fileName       = strcat('Figures\Forecast_fast\',fileNameDetail,'_Errors.xls');               
xlswrite(fileName,cfpcPredAheadRmse{1},'RMS_Scores');
xlswrite(fileName,cfpcPredAheadRmse{2},'RMS_Maturities');
xlswrite(fileName,cfpcPredAheadRmse{3},'RMS_Curves');  
xlswrite(fileName,cfpcPredAheadMad{1},'MAD_Scores');
xlswrite(fileName,cfpcPredAheadMad{2},'MAD_Maturities'); 
xlswrite(fileName,cfpcPredAheadMad{3},'MAD_Curves');  

% plot results
for i=1:NUMBER_GROUPS
    fileName = strcat('Figures\Forecast_fast\',fileNameDetail,'_',...
                      COUNTRY_NAMES(i,:),'.png');
    printForecast(fig,cfpcScores,cfpcPredAheadScores,i,fileName,...
                            NUMBER_TRAINING_DATES,AHEAD_LONG,0);
end;

% save results
fileName       = strcat('Variables_fast\',fileNameDetail,'.mat');
save(fileName,'cfpcPredAheadModels','cfpcPredAheadScores','cfpcPredAheadCurves',...
              'cfpcPredAheadRmse','cfpcPredAheadMad');

save(strcat('Variables_fast\pred',num2str(NUMBER_PREDICTION_DATES),...
            '_all.mat'));



% --------------------------------------------------------------------------
% Descriptive statistics of forecasted scores
% --------------------------------------------------------------------------    
% load('Variables_fast\pred261_ahead1_DNS.mat')
% load('Variables_fast\pred261_ahead1_CPC.mat')
% load('Variables_fast\pred261_ahead1_FPCA_comb.mat')
% load('Variables_fast\pred261_ahead1_CFPC.mat')

models    = cell(4,1);
models{1} = dnsPredModels;
models{2} = cpcPredModels;
models{3} = combFpcaPredModels;
models{4} = cfpcPredModels;

namesFactors = ['DNS  '; 'CPC  '; 'cFPCA'; 'CFPC '];
Ar1ParameterDescriptives = cell(length(models),1);

for m=1:length(models) 
    for i=1:size(models{m},1)
        Ar1ParameterDescriptives{m,i} = zeros(size(models{m},1),...
                                              size(models{m},1)*5);
        for n=1:size(models{m}{i},2)
            Ar1ParameterDescriptives{m}(i,(n-1)*5+1) = ...
                mean(models{m}{i}(NUMBER_TRAINING_DATES:length(data{i})-...
                    AHEAD_SHORT,n,1));
            Ar1ParameterDescriptives{m}(i,(n-1)*5+2) = ...
                std(models{m}{i}(NUMBER_TRAINING_DATES:length(data{i})-...
                    AHEAD_SHORT,n,1));            
            Ar1ParameterDescriptives{m}(i,(n-1)*5+3) = ...
                mean(models{m}{i}(NUMBER_TRAINING_DATES:length(data{i})-...
                    AHEAD_SHORT,n,2));
            Ar1ParameterDescriptives{m}(i,(n-1)*5+4) = ...
                std(models{m}{i}(NUMBER_TRAINING_DATES:length(data{i})-...
                    AHEAD_SHORT,n,2));
         end;
    end;
    xlswrite('Figures\Forecast_fast\scoresPred1Descriptives.xls',...
             Ar1ParameterDescriptives{m},namesFactors(m,:));
end;
clearvars models namesFactors modelsAlpha0 modelsAlpha1 ...
          Ar1ParameterDescriptives m i n