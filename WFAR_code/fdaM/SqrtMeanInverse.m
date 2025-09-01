function gamI = SqrtMeanInverse(gam)
 
[n,T] = size(gam);
dT = 1/(T-1);
psi = zeros(n,T-1);
for i=1:n
    psi(i,:) = sqrt(diff(gam(i,:))/dT+eps); % the SRSFs of the initial warping functions
                                            % 2-norm of psi equal to 1
end
%%Compute Karcher Mean of warping functions/ relative phase
%% Find direction
mnpsi = mean(psi);
dqq = sqrt(sum((psi' - mnpsi'*ones(1,n)).^2,1));
[~, min_ind] = min(dqq);
mu = psi(min_ind,:);    % "mean" of rows of gam. Initialize "mean"
t = 1; % step size 
clear vec;
maxiter = 20;
lvm = zeros(1,maxiter);
vec = zeros(n,T-1);
for iter = 1:maxiter
    for i=1:n
        v = psi(i,:) - mu;
        dot1 = simps(linspace(0,1,T-1),mu.*psi(i,:)); % integration of mu*psi(i,:)/ inner product of two warping func.
        if dot1 > 1
            dot_limited = 1;
        elseif dot1 < -1
            dot_limited = -1;
        else
            dot_limited = dot1;
        end
        len = acos(dot_limited); % F-R distance between any two warping functions
        if len > 0.0001            
            vec(i,:) = (len/sin(len))*(psi(i,:) - cos(len)*mu); % shooting vector
        else
            vec(i,:) = zeros(1,T-1);
        end
        
    end
    vm = mean(vec);  % average direction
    lvm(iter) = sqrt(sum(vm.*vm)*dT); % length of vm
    mu = cos(t*lvm(iter))*mu + (sin(t*lvm(iter))/lvm(iter))*vm; % t is step size
    if lvm(iter) < 1e-6 || iter >= maxiter % stop criterion
        break
    end
end

gam_mu = [0 cumsum(mu.*mu)]/T; % mu is 1*(p-1) on (0, <2). gam_mu is 1*p on (0, 1)
gamI = invertGamma(gam_mu);   % interpolation 
