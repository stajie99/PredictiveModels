function gamI = invertGamma(gam)

N = length(gam);  % max(size(gam))
x = (0:N-1)/(N-1); % N uniform distribution points on (0,1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamI = interp1(gam,x,x); % interpolated values of x=x(gam)%%  gamI and gam is about symetric about y=x.
                         % at specific query points x using linear interpolation



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
if isnan(gamI(N-1))
     gamI(N-1) = 0.950;
end
if isnan(gamI(N))
    gamI(N) = 1;
else
    gamI = gamI./gamI(N);
end

