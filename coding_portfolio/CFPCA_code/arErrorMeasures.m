function [rmse,mad] = arErrorMeasures(saveCell,scores,predScores,...
                                      predCurves,nTrainingPeriods,steps)
%ARERRORMEASURES return the RMSE and MAD of forecasted principal component
%scores
%   
%   Arguments
%   SAVECELL a cell containing the largest eigenvalues in the first row and the
%            eigenvalues
%   SCORES   a cell containing the scores time series
%   PREDSCORES a cell containing the forecasted score time series
%   PREDCURVES a cell of the calculated yield curves based on the predicted
%              scores
%   NTRAININGPERIODS the number of training periods
%   STEPS    the used lag in the forecasting procedure
%
%   Returns
%   RMSE    a cell containing the root mean square error of the principal
%           component scores, of the yield curves evaluated at the
%           maturities, and of the whole yield curve
%   MAD     a cell containing the mean absolute deviation of the principal
%           component scores, of the yield curves evaluated at the
%           maturities, and of the whole yield curve

% load data and set parameters 
global data

nPeriods                             = length(scores{1});
nPredictionPeriods                   = nPeriods-nTrainingPeriods-(steps-1);
NUMBER_PRINCIPAL_COMPONENTS_FUNCTION = size(saveCell{1},2);
dataLength                           = length(scores);

% initialize error measures
rmse       = cell(3,1);
mad        = cell(3,1);

rmse{1}    = zeros(dataLength,NUMBER_PRINCIPAL_COMPONENTS_FUNCTION);
rmse{2}    = zeros(dataLength,size(data{1},2));
rmse{3}    = zeros(dataLength,1);
mad{1}     = zeros(dataLength,NUMBER_PRINCIPAL_COMPONENTS_FUNCTION);
mad{2}     = zeros(dataLength,size(data{1},2));
mad{3}     = zeros(dataLength,1);

for i=1:dataLength;
    for n=1:NUMBER_PRINCIPAL_COMPONENTS_FUNCTION;
        % calculate RMSE for Scores
        rmse{1}(i,n) = ...
            sqrt(sum((scores{i}(nTrainingPeriods+steps:end,n)-...
                  predScores{i}(nTrainingPeriods+steps:end,n)...
                      ).^2)/nPredictionPeriods);
        
        % calculate MAD for Scores
        mad{1}(i,n)  = ...
            sumabs(scores{i}(nTrainingPeriods+steps:end,n)-...
               predScores{i}(nTrainingPeriods+steps:end,n))/...
            nPredictionPeriods;    
    end;
    
    % calculate error measures for curve matrurities and whole curves
    mseMat   = zeros(1,size(data{i},2));
    adMat    = zeros(1,size(data{i},2));
    
    for j=1:nPredictionPeriods
        % calculate predicted yield for given maturities
        predCurves{i}(nTrainingPeriods+steps+(j-1),:) = ...
            predScores{i}(nTrainingPeriods+steps+(j-1),:)*...
            saveCell{i}(2:end,:)' + mean(data{i});
        
        % % calculate MSE for different maturities
        mseMat = mseMat +  ((data{i}(nTrainingPeriods+steps+(j-1),:)...
                    -predCurves{i}(nTrainingPeriods+steps+(j-1),:)).^2);
        adMat  = adMat + abs(data{i}(nTrainingPeriods+steps+(j-1),:)...
                    -predCurves{i}(nTrainingPeriods+steps+(j-1),:));
    end;
    
    % calculate MAD and RMSE for maturities
    rmse{2}(i,:) = sqrt(mseMat/nPredictionPeriods);
    mad{2}(i,:)  = adMat/nPredictionPeriods;  
    
    % calculate RMSE and MAD for yield curves
    rmse{3}(i)   = sqrt(sum(mseMat)/(nPredictionPeriods*length(mseMat)));
    mad{3}(i)    = sum(adMat)/(nPredictionPeriods*length(adMat));
end;
end

