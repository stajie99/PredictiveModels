function logLikeEst = localLogLikeEst(x,y)
%LOCALLOGLIKEEST calculates local log likelihood estimate
%
% Inputs
%   X   array of observed data
%   Y   array of forecasted data

dataLength = size(x,1);
x          = [ones(dataLength,1) x];

logLikeAlpha = (x'*x)\x'*y;
logLikeSigma = sqrt(1/dataLength*(y-x*logLikeAlpha)'*(y-x*logLikeAlpha));

logLikeEst   =[logLikeAlpha
               logLikeSigma];
end