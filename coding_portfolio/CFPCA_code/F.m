function [B,cIteration,improvement] = F(arrays,n,tol,Bstart,maxIterations)
%F F-step of the FG algorithm
%  Every pair of two columns of the current approximation B is rotated such
%  that the corresponding equation (3.11) in Flury (1984) is satisfied
%
%   Arguments
%   ARRAYS   A List of arrays with same dimension 
%   N        Weight vector
%   TOL      The precision for the algorithm
%   BSTART   The initial matrix for the algorithm
%   MAXITERATIONS 
%            The number of maximal iterations before the execution of
%            the algorithm is stopped
%
%   Returns
%   B        common orthogonal array yielding the simultaneous
%            diagonalization
%   CITERATION
%            The number of iterations used
%
%   Literature
%   Flury, BN & Gautschi, W 1986, An Algorithm for Simultaneous Orthogonal 
%   Transformation of Several Positive Definite Symmetric Matrices to 
%   Nearly Diagonal Form, SIAM Journal on Scientific Computing, 
%   vol. 7, no. 1, pp. 169-84.
% 
%   Clarkson, DB 1988, Remark AS 71: A Remark on Algorithm AS 211. 
%   The F-G Diagonalization Algorithm, Applied Statistics, 
%   vol. 37, no. 1, p. 147.
%
%   Flury, BN 1988, Common Principal Components and Related Multivariate  
%   Models, Wiley series in probability and mathematical statistics, 
%   John Wiley & Sons, New York.

% set parameters
k           = length(arrays);
p           = size(arrays{1},1);



% Step F0: initialize arrays and parameters
B           = Bstart;
cIteration  = 0;
improvement = 1;
F           = cell(k,1);

while improvement>tol && cIteration<maxIterations
    for i=1:k
        F{i} = B'*arrays{i}*B;
    end;
    
    % Step F1: store current approximation of B
    Bold       = B;
    cIteration = cIteration+1;
   
    % Step F2
    for m=1:p-1
        
        % Step F21: select columns
        for j=m+1:p
            T        = cell(k,1);
            for i=1:k
                T{i} = F{i}([m,j],[m,j]);
            end;
            
            % Step F22: perform G algorithm to get Jacobi rotaion matrix Q
            Q = G(T,n,tol);
            c = Q(1,1);
            s = Q(2,1);
            
            % Step F23: update F using rotation values c and s
            for i=1:k
                F{i}([m,j],[m,j])=...
                    [c^2*T{i}(1,1)+2*c*s*T{i}(1,2)+s^2*T{i}(2,2)   ...
                    c*s*(T{i}(2,2)-T{i}(1,1))+(c^2-s^2)*T{i}(1,2); ...
                    c*s*(T{i}(2,2)-T{i}(1,1))+(c^2-s^2)*T{i}(1,2)  ...
                    s^2*T{i}(1,1)-2*c*s*T{i}(1,2)+c^2*T{i}(2,2)];
                
                for l=1:p
                    if l~=j && l~=m
                        H1            =  c*F{i}(m,l) + s*F{i}(j,l);
                        H2            = -s*F{i}(m,l) + c*F{i}(j,l);
                        F{i}(l,[m,j]) = [H1 H2];
                        F{i}([m,j],l) = F{i}(l,[m,j])';                        
                    end;
                end;
            end;
            % Step F24: apply Jaboci rotation and update B
            for l=1:p
                  B1     =  B(l,m);
                  B2     =  B(l,j);
                  B(l,m) =  c*B1+s*B2;
                  B(l,j) = -s*B1+c*B2;
            end;
        end;
    end;
    
    % use modified Gram-Schimdt to re-orthogonalizie B
    [B, ~]=gs_m(B ); 
    
% Step F3: calcualte change of B
improvement=norm(Bold-B);
end
end






