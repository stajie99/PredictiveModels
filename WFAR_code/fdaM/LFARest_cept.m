function [cept_hat,c_hat,sigma2_hat,intervalhat,compare] = LFARest_cept(coefM,CvalueLFAR,LFARinterval,step,i)
%Get the LFAR fitted model following the procedure on p19. K is even and
%m_n is set to be totalK
%INPUT: coefM is coef matrix;
%       LFARinterval is {7d,28d,49d,70d,91d,112d,133d,154d,175d,196d,
%217d,238d,259d,280d,301d,322d,343d}, i.e. K=17 candidate intervals in total
%       CvalueLFAR is the 16 critical values corresponding to the 16 intervals except the 1st interval.
%       p is the step or lag
%       i is the i-th generated process for calculating Cvalues
%OUTPUT: 
%        intervalhat is the index of the optimal interval

global LLFM c_hatM sigma2_hatM cept_hatM
ncurve=size(coefM,2);%no. of curves
totalK=(size(coefM,1)-1)/2;

K=length(LFARinterval);
intervalhat=1;
compare=zeros(K,1);

num=LFARinterval(1);
%Get the MLE on 1st interval and set it as the 1st optimal one.
%[c_hat,sigma2_hat,LLFmaxLFAR]=LFARmle(coefM(:,ncurve-num+1:ncurve));%optimal estimates on 1st interval we set m_n to be totalK and no selection
c_hat=c_hatM(:,1,i);
sigma2_hat=sigma2_hatM(:,1,i);
cept_hat=cept_hatM(:,1,i);
LLFmaxLFAR=LLFM(1,i);
for k=2:K
    num=LFARinterval(k);%the number of observations put into consideration
    %[c_hattemp,sigma2_hattemp,LLFtemp]=LFARmle(coefM(:,ncurve-num+1:ncurve));
    LLFtemp=LLFM(k,i);
    
    LLFhat=LFARllf_cept(cept_hat,c_hat,sigma2_hat,coefM(:,ncurve-num+1:ncurve),step,totalK);
    compare(k)=sqrt(abs(LLFtemp-LLFhat));
    if abs(compare(k))>CvalueLFAR(k)
        break;
    else
        %c_hat=c_hattemp;
        %sigma2_hat=sigma2_hattemp;
        c_hat=c_hatM(:,k,i);
        sigma2_hat=sigma2_hatM(:,k,i);
        cept_hat=cept_hatM(:,k,i);
        
        intervalhat=k;
    end
end

