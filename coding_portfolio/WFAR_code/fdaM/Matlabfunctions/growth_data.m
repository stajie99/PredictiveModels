% Simulation of WFAR based on what Prof Marron (2011)'s did 
% functions f
% warping functions gam

function [f, gam]=growth_data(N, t)
f = zeros(size(t, 2),N);
z = normrnd(1,0.25,N,2);
a = linspace(-1,1, N);
gam = zeros(size(t,2), N);
for i = 1:N
    if a(1, i)~= 0
    gam(:,i) = 6.*(exp(a(1, i).*(t+3)/6) - 1)/(exp(a(1, i) - 1)) - 3;
    else
    gam(:,i) = 0:1/(size(t,2)-1):1;
    end
end

for i = 1:N
    t = gam(:,i);
    f(:,i) = z(i, 1).*exp(-(t-1.5).^2/2) + z(i, 2).*exp(-(t+1.5).^2/2);
end

