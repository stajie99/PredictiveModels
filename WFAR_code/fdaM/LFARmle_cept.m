function [cept_hatmle,c_hatmle,sigma2_hatmle,LLFmle]=LFARmle_cept(coefM,step)
%LFARmle calculate functional MLE estimates c_hat, sigma2_hat and the
%corresponding LLF. K is even. m_n is fixed as m_npre. Since the data are
%in finite space, we will only use multi-normal density w.r.t. Lebesgue
%measure.
%INPUT: coefM is the coef matrix of the curves and the dimension is nbasisxncurves;
%       totalK is (nbasis-1)/2 for fourier expansion.
%OUTPUT: all of the results, c_hatmle and sigma2_hatmle, LLFmle have the
%closed form now.
%      

nrep=size(coefM,2);%no. of curves %PLEASE NOTE THAT nrep CHANGES OVER TIME,YOU NEED TO DOUBLE-CHECK FOR nrep
totalK=(size(coefM,1)-1)/2;
nbasis=size(coefM,1);
c_hatmle=zeros(totalK+1,1);
cept_hatmle=zeros(nbasis,1);
sigma2_hatmle=zeros(totalK+1,1);

upper=0;
lower=0;
for i=step+1:nrep
    upper=upper+coefM(1,i)*coefM(1,i-step);
    lower=lower+coefM(1,i-step)^2;
end
c_hatmle(1)=(sum(coefM(1,step+1:nrep))*sum(coefM(1,1:nrep-step))-(nrep-step)*upper)/((sum(coefM(1,1:nrep-step)))^2-(nrep-step)*lower);    %\hat{c}_0 is universal
cept_hatmle(1)=(sum(coefM(1,step+1:nrep))-c_hatmle(1)*sum(coefM(1,1:nrep-step)))/(nrep-step);
sigma2temp=0;
for i=step+1:nrep
    sigma2temp=sigma2temp+(coefM(1,i)-cept_hatmle(1)-c_hatmle(1)*coefM(1,i-step))^2;
end
sigma2_hatmle(1)=sigma2temp/(nrep-step); %sigma^2_0
%There is no change for c_0 and sigma^2_0

U=zeros(totalK,1);
L=zeros(totalK,1);
M=zeros(totalK,1);
for k=1:totalK
    for i=step+1:nrep
        U(k)=U(k)+coefM(2*k,i-step)*coefM(2*k,i)+coefM(2*k+1,i-step)*coefM(2*k+1,i);
        L(k)=L(k)+coefM(2*k,i-step)^2+coefM(2*k+1,i-step)^2;
        M(k)=M(k)+coefM(2*k,i)^2+coefM(2*k+1,i)^2;
    end
    c_hatmle(k+1)=sqrt(2)*(U(k)-(sum(coefM(2*k,1:nrep-step))*sum(coefM(2*k,step+1:nrep))+sum(coefM(2*k+1,1:nrep-step))*sum(coefM(2*k+1,step+1:nrep)))/(nrep-step))/(L(k)-((sum(coefM(2*k+1,1:nrep-step)))^2+(sum(coefM(2*k,1:nrep-step)))^2)/(nrep-step));
    cept_hatmle(2*k)=(sum(coefM(2*k,step+1:nrep))-c_hatmle(k+1)*sum(coefM(2*k,1:nrep-step))/sqrt(2))/(nrep-step);
    cept_hatmle(2*k+1)=(sum(coefM(2*k+1,step+1:nrep))-c_hatmle(k+1)*sum(coefM(2*k+1,1:nrep-step))/sqrt(2))/(nrep-step);
    for i=step+1:nrep
        sigma2_hatmle(k+1)=sigma2_hatmle(k+1)+(coefM(2*k,i)-cept_hatmle(2*k)-c_hatmle(k+1)*coefM(2*k,i-step)/sqrt(2))^2+(coefM(2*k+1,i)-cept_hatmle(2*k+1)-c_hatmle(k+1)*coefM(2*k+1,i-step)/sqrt(2))^2;
    end
    sigma2_hatmle(k+1)=sigma2_hatmle(k+1)/(2*(nrep-step));
end
LLFmle=LFARllf_cept(cept_hatmle,c_hatmle,sigma2_hatmle,coefM,step,totalK);








