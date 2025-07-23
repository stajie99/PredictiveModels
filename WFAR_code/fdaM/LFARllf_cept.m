function LLF=LFARllf_cept(cept_hat,c_hat,sigma2_hat,coefM,step,m)
%Calculate LLF of coefM for the corresponding c_hat and sigma2_hat

%c_hat=c_hat(c_hat~=0);
%sigma2_hat=sigma2_hat(sigma2_hat~=0);

nrep=size(coefM,2);
totalK=m;

%if(totalK==0)
%    LLF=-(nrep-1)/2*log(2*pi)-(nrep-1)*log(sqrt(sigma2_hat(1)))-1/(2*sigma2_hat(1))*(sum(coefM(1,2:nrep).^2)+c_hat(1)^2*sum(coefM(1,1:nrep-1).^2)-2*c_hat(1)*coefM(1,1:nrep-1)*coefM(1,2:nrep)');
%else
    term1=0;term2=0;
    for i=step+1:nrep
        term1=term1+(coefM(1,i)-cept_hat(1)-c_hat(1)*coefM(1,i-step))^2;
    end
    
    for k=1:totalK
        for i=step+1:nrep
            term2=term2+((coefM(2*k,i)-cept_hat(2*k)-c_hat(k+1)*coefM(2*k,i-step)/sqrt(2))^2+(coefM(2*k+1,i)-cept_hat(2*k+1)-c_hat(k+1)*coefM(2*k+1,i-step)/sqrt(2))^2)/sigma2_hat(k+1);
        end
    end
    LLF=-(2*totalK+1)*(nrep-step)*log(2*pi)/2-(nrep-step)*log(sqrt(sigma2_hat(1)))-(nrep-step)*sum(log(sigma2_hat(2:totalK+1)))-1/(2*sigma2_hat(1))*term1-0.5*term2;
%end
