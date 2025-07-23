function [valuesEigenvectors,eigenvalues,variationExpl, scores] = ...
         fpca(dataMatrix)
% FPCA calculates the functional principal components of a given data
% matrix using a B-spline representation
%
%   Arguments
%   DATAMATRIX
%            Data array which is already mean corrected
%
%   Returns
%   VALUESEIGENVECTORS
%            An array containing the first NUMBER_PRINCIPAL_COMPONENTS 
%            functional values of the eigenvectors
%   EIGENVALUES
%            A vector of length NUMBER_PRINCIPAL_COMPONENTS which stores
%            the largest NUMBER_PRINCIPAL_COMPONENTS eigenvalues
%   VARIATIONEXPL
%            A vector of length NUMBER_PRINCIPAL_COMPONENTS which stores
%            the explained variation of each of the first 
%            NUMBER_PRINCIPAL_COMPONENTS components
%   SCORES   The proncipal components scores
%
%   Literature
%   Ramsay, JO & Silverman, BW 2005, Functional data analysis, 
%   Springer series in statistics, 2nd edn, Springer, New York.
%
%   Ramsay, JO, Hooker, G & Graves, S 2009, Functional data analysis with R 
%   and MATLAB, Use R!, Springer, New York.  

global KNOT_SEQUENCE MATURITIES SPLINE_ORDER NUMBER_PRINCIPAL_COMPONENTS



% estimation of the coefficient matrix using least square B-splines
coefMatrix=zeros(length(dataMatrix),length(MATURITIES));
for j=1:size(dataMatrix,1)
    splineTemp      = spap2(KNOT_SEQUENCE,SPLINE_ORDER,MATURITIES,...
                           dataMatrix(j,:));
    coefMatrix(j,:) = fnbrk(splineTemp,'c');
end

% calculation of the Gram matrix wMatrix
wMatrix     = Gram(KNOT_SEQUENCE,SPLINE_ORDER);
wMatrixRoot = chol(wMatrix);

% create matrix for eigenanalysis
eigenMatrix = wMatrixRoot*(coefMatrix'*coefMatrix./...
                              (size(coefMatrix,1)+1))*wMatrixRoot';
eigenMatrix = (eigenMatrix + eigenMatrix')./2;

% eigenanalysis
[eigenvectors, eigenvalues] = eig(eigenMatrix);
[eigenvalues, sortIndices]  = sort(diag(eigenvalues),'descend');
eigenvectors                = eigenvectors(:,sortIndices);

% select eigenvectors and eigenvalues
sumEigenvalues      = sum(eigenvalues);
eigenvalues         = eigenvalues(1:NUMBER_PRINCIPAL_COMPONENTS);
eigenvectors        = eigenvectors(:,1:NUMBER_PRINCIPAL_COMPONENTS);
sumEigenvectors     = sum(eigenvectors);
eigenvectors(:,sumEigenvectors < 0) = -eigenvectors(:,sumEigenvectors < 0);

variationExpl       = eigenvalues(1:NUMBER_PRINCIPAL_COMPONENTS)./...
                      sumEigenvalues;
                          
%calculate the scores and the eigenvectors 
scores              = coefMatrix * wMatrixRoot' * eigenvectors; 
eigenvectors        = wMatrixRoot\eigenvectors;

valuesEigenvectors  = zeros(NUMBER_PRINCIPAL_COMPONENTS, length(MATURITIES));
for j=1:NUMBER_PRINCIPAL_COMPONENTS
// Evaluate spline function
   valuesEigenvectors(j,:) = fnval(spmak(KNOT_SEQUENCE,...
                                         eigenvectors(:,j)'),MATURITIES);
end;
end