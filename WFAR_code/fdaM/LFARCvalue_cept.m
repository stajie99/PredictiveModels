function [Cvalue,LRcompare,intervalhattemp]=LFARCvalue_cept(coefgene,ksiLFAR,npro,LFARinterval,step)
%Calculate critical values for LFAR
%INPUT: coefgene is the generated coef matrix 23x343x1000
%       ksiLFAR is the bound for calculating critical values
%       npro is the number of generated processes
%       p is the lag
%       LFARinterval is {7d,28d,49d,70d,91d,112d,133d,154d,175d,196d,217d,238d,259d,280d,301d,322d,343d}, 
%i.e. K=17 candidate intervals in total
%RETURN: Cvalue is 17x1 vector

global LLFM
totalK=11;
K=length(LFARinterval);
Cvalue=zeros(K,1);
%totalK=(size(coefgene,1)-1)/2;

CV_start=20;
Cvalue=ones(K,1)*1000000;
% Cvalue(1)=1000000000;
% Cvalue(2)=11;
% Cvalue(3)=8;
% Cvalue(4)=10000;
% Cvalue(5)=5000000;
% Cvalue(6)=3800000;
% Cvalue(7)=4300000;
% Cvalue(8)=3200000;
% Cvalue(9)=2600000;
%Cvalue(10)=48;
%Cvalue(11)=500;
%Cvalue(12)=500;
%Cvalue(13)=500;
%Cvalue(14)=500;
%Cvalue(15)=500;
%Cvalue(16)=500;
%Cvalue(17)=500;
dist=0.1;

for k=2:K %now we need to calculate Cvalue(k) for k=2,...,K
    Cvalue(k)=CV_start;
    LHS=zeros(1,K-k+1);%we need to test every interval from k to K
    LRcompare=zeros(npro,K-k+1);
    bound=ksiLFAR*(k-1)/(K-1);%the RHS of the inequality on p23
    %bound=ksiLFAR(k)*(k-1)/(K-1);
    
    while(LHS<=bound)
        Cvalue(k)=Cvalue(k)-dist;
        %noted=0;
        LHS=zeros(1,K-k+1);%we need to test every interval from k to K
        LRcompare=zeros(npro,K-k+1);
        if(Cvalue(k)<=0)
            break
        else
            for q=k:K
            %for q=k
                num=LFARinterval(q);
                intervalhattemp=zeros(npro,K-k+1);
                for i=1:npro
                    %w=0;
                    y=coefgene(:,:,i);%for i-th process 23x343
                    ncurve=size(y,2);
                    [cept_hattemp,c_hattemp,sigma2_hattemp,intervalhattemp(i,q-k+1)]=LFARest_cept(y,Cvalue,LFARinterval(1:q),step,i);%LFAR MLE
                    
                    %if(intervalhattemp~=q)
                    %    w=1;
                    %end
                
                    LLFhat=LFARllf_cept(cept_hattemp,c_hattemp,sigma2_hattemp,y(:,ncurve-num+1:ncurve),step,totalK);
            
                    %[c_hatMLE,sigma2_hatMLE,LLFMLE]=LFARmle(y(:,ncurve-num+1:ncurve));
                    LLFMLE=LLFM(q,i);
                    LRcompare(i,q-k+1)=abs(sqrt(abs(LLFMLE-LLFhat)));
                    [i,k,q,Cvalue(k),intervalhattemp(i,q-k+1),LRcompare(i,q-k+1),bound]
                end
                LHScheck=mean(LRcompare(:,q-k+1));
                if(LHScheck>bound)
                    %noted=1;
                    break;
                end
            end
            %if(noted==1)
            %    break;
            %else
               LHS=mean(LRcompare,1)
               %oh=LRcompare(:,1)';
               %oh(oh>Cvalue(k))
               %find(oh)
               %checkinterval=intervalhattemp(:,1)';
               %checkinterval(checkinterval~=k)
            %end
            
            
            %mark=zeros(K-k+1,1);
            %if(intervalhattemp(:,1)==k)
            %    mark(1)=1;
            %    for m=2:K-k+1
            %        if(intervalhattemp(:,m)==m+k-1)
            %            mark(m)=1;
            %        else
            %            mark(m)=0;
            %        end
            %    end
            %    if(mark==1)
            %        flag=1;
            %    else
            %        flag=0
            %        mark
            %        break;
            %    end
            %end
                        
        end
    end
    Cvalue(k)=Cvalue(k)+dist;
end