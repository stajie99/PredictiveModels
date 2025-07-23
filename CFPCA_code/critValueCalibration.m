function criticalValues = critValueCalibration(scores,i,n,intervals,...
                                               nTrainingDates,step)
%CRITVALUECALIBRATION uses monte carlo simulation to find critical values
%for the LAR forecasts
%
%   Arguments
%   SCORES           a cell containing the scores time series
%   I                the group index
%   N                the index of the component
%   INTERVALS        the set of the lengths of the intervals
%   NTRAININGPERIODS the number of training periods
%   STEP             the lag
% 
%   Returns
%   CRITICAL VALUES  a set of critical values

% load data and set parameters 
NUMBER_INTERVALS      = length(intervals);   
SAMPLE_LENGTH         = intervals(end)+step;
NUMBER_SAMPLES        = 10000;
POWER                 = 0.5;



% create samples
% find parameters for monte carlo simulation
x = scores{i}(1:nTrainingDates-1,n);
y = scores{i}(2:nTrainingDates,n);
modParameter = localLogLikeEst(x,y);

cfpcScoresSampleStart = modParameter(1)/(1-modParameter(2));   

% simulate innovations following N(0,sigma^2)
epsilon          = randn(SAMPLE_LENGTH,NUMBER_SAMPLES).*modParameter(3);

% create sample scores for critical value calibartion 
cfpcScoresSample = zeros(SAMPLE_LENGTH,NUMBER_SAMPLES);
cfpcScoresSample(1,:) = modParameter(1)+modParameter(2)*...
                        cfpcScoresSampleStart+epsilon(1,:);
for m=2:SAMPLE_LENGTH
    cfpcScoresSample(m,:) = modParameter(1)+modParameter(2)*...
                            cfpcScoresSample(m-1,:)+epsilon(m,:); 
end;

x = scores{i}(1:nTrainingDates-step,n);
y = scores{i}(step+1:nTrainingDates,n);
trueParameter = localLogLikeEst(x,y);



% initialize local likelihood estimation for calibration
locMaxLikeEst = zeros(NUMBER_INTERVALS,3,NUMBER_SAMPLES);
adaptiveEst   = zeros(NUMBER_INTERVALS,3,NUMBER_SAMPLES);
locMaxLikeFun = zeros(NUMBER_INTERVALS,NUMBER_SAMPLES);

T        = zeros(NUMBER_INTERVALS,NUMBER_SAMPLES);
D        = zeros(NUMBER_INTERVALS,NUMBER_SAMPLES);
R        = zeros(NUMBER_INTERVALS,NUMBER_SAMPLES);
ExpD     = zeros(NUMBER_INTERVALS,1);
ExpR     = zeros(NUMBER_INTERVALS,1);

% local likelihood estimation of simulated samples      
for k=1:NUMBER_INTERVALS
    % initialize estimators and local ML function
    curIntLength     = intervals(k);

    % calculate local maximum likelihood estimator at 
    % SAMPLE_LENGTH           
    for N=1:NUMBER_SAMPLES
        x = cfpcScoresSample(SAMPLE_LENGTH-curIntLength-...
                                  step+1:SAMPLE_LENGTH-step,N);
        y = cfpcScoresSample(SAMPLE_LENGTH-curIntLength+1:...
                                  SAMPLE_LENGTH,N);
        locMaxLikeEst(k,:,N) = localLogLikeEst(x,y);
        locMaxLikeFun(k,N)   = localLogLikeFun(locMaxLikeEst(k,:,N)',x,y);
        R(k,N)               = (abs(locMaxLikeFun(k,N)-...
                                localLogLikeFun(trueParameter,x,y)))^POWER;
    end;

    ExpR(k) = mean(R(k,:));
end;



% Critical value calibration
% accept smallest interval as local homogeneity is assumed
criticalValues  = ones(NUMBER_INTERVALS,1)*1000;
adaptiveEst(1,:,:) = locMaxLikeEst(1,:,:);

% for larger intervals do binary search to find the critical values
for k=2:NUMBER_INTERVALS
    curIntLength   = intervals(k);

    [i n k curIntLength] % status
    upperCritValue = criticalValues(k);
    lowerCritValue = 0.1;
    ExpD(k)        = 2*ExpR(k);
    for N=1:NUMBER_SAMPLES            
        x      = cfpcScoresSample(SAMPLE_LENGTH-curIntLength-...
                                  step+1:SAMPLE_LENGTH-step,N);
        y      = cfpcScoresSample(SAMPLE_LENGTH-curIntLength+1:...
                                  SAMPLE_LENGTH,N);
        T(k,N) = (abs(locMaxLikeFun(k,N)-localLogLikeFun(...
                           adaptiveEst(k-1,:,N)',x,y)))^POWER;
    end;

    % find critical values which lead to estimates fulfilling 
    % the condition ExpD{i,n}(k) <=R(k)
    while ((upperCritValue-lowerCritValue)>1.0e-4 || ExpD(k)>ExpR(k)) 

      midCritValue = mean([lowerCritValue;upperCritValue]);
      for N=1:NUMBER_SAMPLES
          if T(k,N)<=midCritValue
              adaptiveEst(k,:,N) = locMaxLikeEst(k,:,N);
              D(k,N)         = 0; % definition of D
          else                   
%                       adaptiveEst{k-1}(:,N) = locMaxLikeEst{k-1}(:,N);
              adaptiveEst(k,:,N) = locMaxLikeEst(k-1,:,N);
              D(k,N)             = (abs(locMaxLikeFun(k,N)-...
                                    localLogLikeFun(adaptiveEst(k,:,N)',...
                                    x,y)))^POWER;      
          end;
      end;

      ExpD(k)            = mean(D(k,:));

      %set new boundaries for binary search
      if ExpD(k)>ExpR(k)
         lowerCritValue = midCritValue;
      else
         upperCritValue = midCritValue;
      end;
    end;

    % set critical values to last approximation
    criticalValues(k) = midCritValue;
end;

%       for testing: save results
        fileName = strcat('Figures\LAR\testCriticalValues_',...
                          num2str(NUMBER_INTERVALS),'int_samp',...
                          num2str(NUMBER_SAMPLES),'_',num2str(step),'_',...
                          num2str(i),num2str(n),'.xls');
        xlswrite(fileName,'c','crit', 'A1');
        xlswrite(fileName,criticalValues,'crit', 'A2');

        xlswrite(fileName,'D','crit', 'B1');  
        xlswrite(fileName,ExpD,'crit', 'B2');

        xlswrite(fileName,'R','crit', 'C1');
        xlswrite(fileName,ExpR,'crit', 'C2');

        xlswrite(fileName,'M','crit', 'E1');
        xlswrite(fileName,modParameter','crit', 'E2');

        xlswrite(fileName,'T','crit', 'E3');
        xlswrite(fileName,trueParameter','crit', 'E4');


        xlswrite(fileName,T','T');
        xlswrite(fileName,D','D');
        xlswrite(fileName,R','R');

        ada = zeros(NUMBER_SAMPLES,NUMBER_INTERVALS);
        for k=1:NUMBER_INTERVALS
            ada(:,k) = adaptiveEst(k,2,:);
        end;
        xlswrite(fileName,ada,'adaptiveEst');

        xlswrite(fileName,ada,'adaptiveEst');

        lML = zeros(NUMBER_SAMPLES,NUMBER_INTERVALS);
        for k=1:NUMBER_INTERVALS
            lML(:,k) = locMaxLikeEst(k,2,:);
        end;
        xlswrite(fileName,lML,'locML');

        plot(criticalValues);

        fileName = strcat('Variables\criticalValues_',num2str(step),'_samp',...
                          num2str(NUMBER_SAMPLES),'.mat');
        save(fileName,'criticalValues');
end