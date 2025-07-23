function [B,method,diff]=FG(arrays,n,tol)
%FG calculates a simultaneous diagonailation of a list of arrays
%
%   Arguments
%   ARRAYS   A List of arrays with same dimension 
%   N        Weight vector
%   TOL      The precision for the algorithm
%
%   Returns
%   B        common orthogonal array yielding the simultaneous
%            diagonalization
%   METHOD   A string with indicates the initial array used for B  

p              = size(arrays{1},1);
MAX_ITERATIONS = 100;



% start F algorithm with initial array I
BStart                = eye(p);
[B,nIterations, diff] = F(arrays,n,tol,BStart,MAX_ITERATIONS);

if nIterations<MAX_ITERATIONS
    method      = 'Identity';
else  
    % restart F algorithm with a random array
    BStart                = orth(rand(p));
    [B,nIterations, diff] = F(arrays,n,tol,BStart,MAX_ITERATIONS);
    if nIterations<MAX_ITERATIONS
        method            = 'Random';
    else
        % restart F algorithm with a matrix of eigenvectos of the average
        % array Average of the arrays
        Average         = zeros(size(arrays{1}));
        for i=1:length(arrays)
            Average     = Average+arrays{i};
        end;
        Average               = Average/length(arrays);
        [BStart,~]            = eig(Average);
        [B,nIterations, diff] = F(arrays,n,tol,BStart,MAX_ITERATIONS);
        
        % evaluate number of iteratins
        if nIterations<MAX_ITERATIONS
            method      = 'Average';
        else
            method      = 'noConvergence';
        end;
    end;
end;



