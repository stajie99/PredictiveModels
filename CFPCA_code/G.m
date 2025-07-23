function Q=G(T,n,tol)
% G G-step of the FG algorithm by Flury and Gautschi
%   An orthogonal (2x2) array which solves the two-dimensional analogue of
%   equation (3.11), stated in euqatioin (3.14) in Flury (1988) is found
%   via iteration
%   This matrix determins the rotation of a pair of vectors currently being
%   adjusted in the F-step

%   Arguments
%   T        An orthogonal (2x2) array 
%   N        Weight vector
%   TOL      The precision for the algorithm
%
%   Returns
%   Q        Rotaion matrix
%   
%   Literature
%   Flury, BN & Gautschi, W 1986, An Algorithm for Simultaneous Orthogonal 
%   Transformation of Several Positive Definite Symmetric Matrices to 
%   Nearly Diagonal Form, SIAM Journal on Scientific Computing, 
%   vol. 7, no. 1, pp. 169-84.
%
%   Flury, BN 1988, Common Principal Components and Related Multivariate  
%   Models, Wiley series in probability and mathematical statistics, 
%   John Wiley & Sons, New York.

MAX_ITERATIONS_G = 5; 



% Step G0: initialize Q with I_2
Q           = eye(2); 
improvement = 1;                                
cIteration  = 0;                                  

while improvement>tol && cIteration<MAX_ITERATIONS_G
    
    % Step G1: store current Q
    cIteration=cIteration+1;
    Qold=Q; 
    U=zeros(2);
    
    % Step G2: compute the d_ij = Q_j^T T_i Q_j using the current Q
    for i=1:length(T);
         di1=Q(:,1)'*T{i}*Q(:,1);
         di2=Q(:,2)'*T{i}*Q(:,2);
         U=U+n(i)*(di1-di2)/(di1*di2)*T{i};
    end; 
    
    % Step G3: computation of unique rotation
    [Q,~]=eig(U);
    if abs(Q(1,1))==min(min(abs(Q)))           
       Q=[Q(:,2) Q(:,1)];                      % chose cos to be maximal
    end;    
    Q=[sign(Q(1,1))*Q(:,1) sign(Q(2,2))*Q(:,2)]; % and of positive sign 
    
    % Step G4: calcualte change of Q
    improvement=norm(Qold-Q);
end;