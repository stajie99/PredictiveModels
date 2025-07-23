function logLike = localLogLikeFun(a,x,y)
%LOCALLOGLIKEFUN calculates local log likelihood function
%
% Inputs
%   A   an array of the paramteres in the form a(1) = alpha_0, a(2) =
%   alpha_1, and a(3) = standard deviation
%   X   array of observed data
%   Y   array of forecasted data

dataLength = size(x,1);
x          = [ones(dataLength,1) x];

logLike = -dataLength/2*log(2*pi)-dataLength*log(a(3))-1/(2*(a(3)^2))*...
          (y-x*a(1:2))'*(y-x*a(1:2));
end