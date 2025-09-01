function [intervalhat,cept_hat,c_hat,sigma2_hat,compare] = LFARestsimu_cept(coefM,CvalueLFAR,LFARinterval,step)
%Get the LFAR fitted model following the procedure on p19. K is even and
%m_n is set to be totalK
%INPUT: coefM is coef matrix;
%       p is the step or lag
%OUTPUT: 
%        intervalhat is the index of the optimal interval

ncurve=size(coefM,2);%no. of curves
totalK=(size(coefM,1)-1)/2;
nbasis=size(coefM,1);
K=length(LFARinterval);
intervalhat=1;
compare=zeros(K,1);
c_hat=zeros(totalK+1,1);
sigma2_hat=zeros(totalK+1,1);
cept_hat=zeros(nbasis,1);

num=LFARinterval(1);
%Get the MLE on 1st interval and set it as the 1st optimal one.
[cept_hat,c_hat,sigma2_hat,LLFmaxLFAR]=LFARmle_cept(coefM(:,ncurve-num+1:ncurve),step);
c_hat=c_hat;
intind=1;

for k=2:K
    num=LFARinterval(k);%the number of observations put into consideration
    [cept_hattemp,c_hattemp,sigma2_hattemp,LLFtemp]=LFARmle_cept(coefM(:,ncurve-num+1:ncurve),step);

    LLFhat=LFARllf_cept(cept_hat,c_hat,sigma2_hat,coefM(:,ncurve-num+1:ncurve),step,totalK);
    compare(k)=sqrt(abs(LLFtemp-LLFhat));
    if abs(compare(k))>CvalueLFAR(k)
        break;
    else
        intind=intind+1;
        c_hat=c_hattemp;
        sigma2_hat=sigma2_hattemp;
        cept_hat=cept_hattemp;
        intervalhat=k;
    end
end