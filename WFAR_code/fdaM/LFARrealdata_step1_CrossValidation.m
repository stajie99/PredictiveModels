%%%real data results in paper: comparion btw diff models.
avepricematrix=reshape(CAhourly(:,3),24,1037);
%avepricematrix=reshape(averprice,24,1037);
CAlogall=log(avepricematrix(:,461:803));%24x343

electime=1:24;
electimescaled=zeros(24,1);
for i=1:24
    electimescaled(i)=electime(i)/24;
end
%Create the fd objects by fourier transformation
CAnbasis=23;
CAfourierb=create_fourier_basis([0,1],CAnbasis,1);
eval_penalty(CAfourierb,int2Lfd(0))

CAfourierfd=smooth_basis(electimescaled,CAlogall,CAfourierb);
plot(CAfourierfd)
title('CAlog_fd new')

CAfourierfdcoef=getcoef(CAfourierfd);%23x343
ncurve=size(CAfourierfdcoef,2);%343

% CAsub=CAlogall(:,1:300);
% CAsubcoef=CAfourierfdcoef(:,1:300);
% fdsub=fd(CAsubcoef,CAfourierb);
% plot(fdsub)
% axis([0,1,1.5,5.5])

LFARinterval=[14,30,45,60,90,120,180,240,300];
ninter=length(LFARinterval);
step=1;

% ceptsub=zeros(CAnbasis,1);
% chatsub=zeros((CAnbasis-1)/2+1,1);
% sigma2hatsub=zeros((CAnbasis-1)/2+1,1);
% [ceptsub,chatsub,sigma2hatsub,LLFsub]=LFARmle_cept(CAsubcoef,step);


%%%%%%%%%---the following critical values come from cross-validation-----
CvalueLFAR=[1000000;12;8;6;8;7;9;12;8];  %13-18
CvalueLFAR=[1000000;12;8;6;8;7;9;12;8];
% CvalueLFAR=ones(9,1)*12;
% CvalueLFAR(3)=8;
% CvalueLFAR(4)=6;
% CvalueLFAR(5)=8;
% CvalueLFAR(6)=7;
% CvalueLFAR(7)=9;
% CvalueLFAR(8)=15;
% CvalueLFAR(9)=3;
% CvalueLFAR=[1000000;16;6.4;4.5;5.1;3.7;4.3;3.2;2.5];
% CvalueLFAR_1=[1000000;15;6.4;4.4;5;3.8;4.3;3.3;2.6];
%
plot(2:9,CvalueLFAR(2:9),'linewidth',2)
axis([1,9,2,16])
title('Electricity prices: critical values for 2nd to 9th candidate intervals','fontsize',12)
xlabel('Index of candidate intervals','fontsize',12)
ylabel('Critical values','fontsize',12)
nforc=343-301+1;%43
totalK=(CAnbasis-1)/2;
CAinter=zeros(nforc,1);
CAchat=zeros(totalK+1,nforc);%12x43
CAsigma2hat=zeros(totalK+1,nforc);%12x43
CAcepthat=zeros(CAnbasis,nforc);
step=1;
%matlabpool open local 2
%pmode start local 4
%parfor i=1:npro
for i=1:nforc
    [CAinter(i),CAcepthat(:,i),CAchat(:,i),CAsigma2hat(:,i)]=LFARestsimu_cept(CAfourierfdcoef(:,i:i+299),CvalueLFAR,LFARinterval,1);
end
 
% CAinter_try=zeros(nforc,1);
% CAchat_try=zeros(totalK+1,nforc);%12x43
% CAsigma2hat_try=zeros(totalK+1,nforc);%12x43
% CAcepthat_try=zeros(CAnbasis,nforc);
% CvalueLFAR_2=CvalueLFAR;
% for i=1:nforc
%     [CAinter_try(i),CAcepthat_try(:,i),CAchat_try(:,i),CAsigma2hat_try(:,i)]=LFARestsimu_cept(CAfourierfdcoef(:,i:i+299),CvalueLFAR_2,LFARinterval,1);
% end
% xlswrite('coef.xls',CAfourierfdcoef,'Sheet1','A1')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot of K, there are 43 estimations of K for each time point
%CAchat(12x43) contains the fourier coef of K
coefK=zeros(23,43);
coefK(1,:)=CAchat(1,:);
for i=1:11
    coefK(2*i+1,:)=CAchat(i+1,:);
    coefK(2*i,:)=zeros(1,43);
end
kernelK=fd(coefK,CAfourierb);
plot(kernelK)
title('Estimated kernel function K','fontsize',12)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CAcoef_forc=zeros(CAnbasis,nforc);
for i=1:nforc
    CAcoef_forc(1,i)=CAcepthat(1,i)+CAchat(1,i)*CAfourierfdcoef(1,i+299);
    for j=1:totalK
        CAcoef_forc(2*j,i)=CAcepthat(2*j,i)+1/sqrt(2)*CAchat(j+1,i)*CAfourierfdcoef(2*j,i+299);
        CAcoef_forc(2*j+1,i)=CAcepthat(2*j+1,i)+1/sqrt(2)*CAchat(j+1,i)*CAfourierfdcoef(2*j+1,i+299);
    end
end
CAcoef_true=CAfourierfdcoef(:,(300+step):343);
%phaseindex=1:nforc;
% plotmovieLFAR(CAcoef_forc,CAcoef_true,CAfourierb,phaseindex,step)
CAfd_forc=fd(CAcoef_forc,CAfourierb);
%CAfd_true=fd(CAcoef_true,CAfourierb);
CAfd_forcval=eval_fd(electimescaled,CAfd_forc);%24x43
%CAfd_trueval=eval_fd(electimescaled,CAfd_true);%24x43
err_forc=CAfd_forcval-CAlogall(:,301:343);%24x43
RMSEforc=sqrt(mean(err_forc.^2,2)) %24x1
target = [0.170 0.223 0.260 0.236 0.230 0.236 0.179 0.163 0.151 0.224 0.287 0.297 0.320 0.335 0.326 0.338 0.334 0.312 0.313 0.302 0.299 0.279 0.133 0.118];
target2 = [0.171 0.225 0.265 0.246 0.239 0.271 0.207 0.193 0.171 0.229 0.291 0.304 0.322 0.349 0.351 0.36 0.352 0.328 0.317 0.303 0.3 0.295 0.141 0.122];
[sum(RMSEforc<=target')  sum(RMSEforc<=target2')]
plot(CAinter)

fourierb=create_fourier_basis([1,24],CAnbasis);
eval_penalty(fourierb,int2Lfd(0))


%------forecasting by using FAR(fix window 300)----------------------
CAchat_FAR=zeros(totalK+1,nforc);%12x43
CAsigma2hat_FAR=zeros(totalK+1,nforc);%12x43
CAcepthat_FAR=zeros(CAnbasis,nforc);

%matlabpool open local 2
%pmode start local 4
%parfor i=1:npro
for i=1:nforc
    [CAcepthat_FAR(:,i),CAchat_FAR(:,i),CAsigma2hat_FAR(:,i)]=LFARmle_cept(CAfourierfdcoef(:,i:i+299),step);
end

CAcoef_forcFAR=zeros(CAnbasis,nforc);
for i=1:nforc
    CAcoef_forcFAR(1,i)=CAcepthat_FAR(1,i)+CAchat_FAR(1,i)*CAfourierfdcoef(1,i+299);
    for j=1:totalK
        CAcoef_forcFAR(2*j,i)=CAcepthat_FAR(2*j,i)+1/sqrt(2)*CAchat_FAR(j+1,i)*CAfourierfdcoef(2*j,i+299);
        CAcoef_forcFAR(2*j+1,i)=CAcepthat_FAR(2*j+1,i)+1/sqrt(2)*CAchat_FAR(j+1,i)*CAfourierfdcoef(2*j+1,i+299);
    end
end

CAfd_forcFAR=fd(CAcoef_forcFAR,CAfourierb);
CAfd_forcvalFAR=eval_fd(electimescaled,CAfd_forcFAR);%24x43
err_forcFAR=CAfd_forcvalFAR-CAlogall(:,301:343);%24x43
RMSEforc_FAR=sqrt(mean(err_forcFAR.^2,2)) %24x1

%------forecasting by using FAR (fix window 150)----------------------
CAchat_FAR150=zeros(totalK+1,nforc);%12x43
CAsigma2hat_FAR150=zeros(totalK+1,nforc);%12x43
CAcepthat_FAR150=zeros(CAnbasis,nforc);

%matlabpool open local 2
%pmode start local 4
%parfor i=1:npro
for i=1:nforc
    [CAcepthat_FAR150(:,i),CAchat_FAR150(:,i),CAsigma2hat_FAR150(:,i)]=LFARmle_cept(CAfourierfdcoef(:,i+299-150+1:i+299),step);
end

CAcoef_forcFAR150=zeros(CAnbasis,nforc);
for i=1:nforc
    CAcoef_forcFAR150(1,i)=CAcepthat_FAR150(1,i)+CAchat_FAR150(1,i)*CAfourierfdcoef(1,i+299);
    for j=1:totalK
        CAcoef_forcFAR150(2*j,i)=CAcepthat_FAR150(2*j,i)+1/sqrt(2)*CAchat_FAR150(j+1,i)*CAfourierfdcoef(2*j,i+299);
        CAcoef_forcFAR150(2*j+1,i)=CAcepthat_FAR150(2*j+1,i)+1/sqrt(2)*CAchat_FAR150(j+1,i)*CAfourierfdcoef(2*j+1,i+299);
    end
end

CAfd_forcFAR150=fd(CAcoef_forcFAR150,CAfourierb);
CAfd_forcvalFAR150=eval_fd(electimescaled,CAfd_forcFAR150);%24x43
err_forcFAR150=CAfd_forcvalFAR150-CAlogall(:,301:343);%24x43
RMSEforc_FAR150=sqrt(mean(err_forcFAR150.^2,2)) %24x1

%------forecasting by ARX (Weron (2008))--------------------------------
CAforec_load=reshape(CAhourly(:,5),24,1037);%The forecasted load
CAforec_actload=CAforec_load(:,461:803);%The actual forecasted load for out-of-sample forecast
CAforec_logload=log(CAforec_actload);%24x343 for the log of forecasted load
%%%%%begf=274;%nrep=63;
begf=301;nforc=343-begf+1;

mp=min(CAlogall);
mCAlog=mp';%The min log price for each day 343x1

DumMat=[1 0 0;0 0 0;0 0 0;0 0 0;0 0 0;0 1 0;0 0 1];
DumMat49=zeros(343,3);
for i=1:49
    DumMat49(7*(i-1)+1:7*i,:)=DumMat;
end

forecARXlog=zeros(24,nforc);
%i=1;
for i=1:nforc
    final=begf+i-2;
    start=1+7;
    nrep=final-start+1;
    Xm=zeros(nrep,8);
    finalmod=mod(final,7);
    Xm(:,6:8)=DumMat49(start:final,:);
    
       %j=2; 
    for j=1:24
        yv=CAlogall(j,start:final)';
        Xm(:,1)=CAlogall(j,start-1:final-1)';
        Xm(:,2)=CAlogall(j,start-2:final-2)';
        Xm(:,3)=CAlogall(j,start-7:final-7)';
        Xm(:,4)=mCAlog(start-1:final-1);
        Xm(:,5)=CAforec_logload(j,start:final)';
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%8x1
        
        forecARXlog(j,i)=[CAlogall(j,final) CAlogall(j,final-1) CAlogall(j,final-6) mCAlog(final) CAforec_logload(j,final+1) DumMat49(final+1,:)]*CAlogpricebeta;
    end
end

err_forcARX=forecARXlog-CAlogall(:,301:343);%24x43
RMSEforcARX=sqrt(mean(err_forcARX.^2,2)) %24x1

%----------------------------------------------------------------------
%------forecasting by AR(1)--------------------------------------------
begf=301;nforc=343-begf+1;

forecARlog=zeros(24,nforc);
%i=1;
for i=1:nforc
    final=begf+i-2;
    start=1+1;
    nrep=final-start+1;
    Xm=zeros(nrep,2);
    
    for j=1:24
        yv=CAlogall(j,start:final)';
        Xm(:,1)=ones(nrep,1);
        Xm(:,2)=CAlogall(j,start-1:final-1)';
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%2x1
        
        forecARlog(j,i)=[1 CAlogall(j,final)]*CAlogpricebeta;
    end
end

err_forcAR=forecARlog-CAlogall(:,301:343);%24x43
RMSEforcAR=sqrt(mean(err_forcAR.^2,2)) %24x1
%-----------------------------------------------------------------------
%------forecasting by seasonalAR (without X variables in ARX)------------
begf=301;nforc=343-begf+1;

forecARsealog=zeros(24,nforc);
%i=1;
for i=1:nforc
    final=begf+i-2;
    start=1+7;
    nrep=final-start+1;
    Xm=zeros(nrep,3);
   
    for j=1:24
        yv=CAlogall(j,start:final)';
        Xm(:,1)=CAlogall(j,start-1:final-1)';
        Xm(:,2)=CAlogall(j,start-2:final-2)';
        Xm(:,3)=CAlogall(j,start-7:final-7)';
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%8x1
        
        forecARsealog(j,i)=[CAlogall(j,final) CAlogall(j,final-1) CAlogall(j,final-6)]*CAlogpricebeta;
    end
end

err_forcARsea=forecARsealog-CAlogall(:,301:343);%24x43
RMSEforcARsea=sqrt(mean(err_forcARsea.^2,2)) %24x1
%------------------------------------------------------------------------
%------forecasting by VAR(1)---------------------------------------------
begf=301;nforc=343-begf+1;
forecVARlog=zeros(24,nforc);

for i=1:nforc
    final=begf+i-2;
    start=1+1;
    nrep=final-start+1;
    Xm=zeros(nrep,25);
    
    for j=1:24
        yv=CAlogall(j,start:final)';
        Xm(:,1)=ones(nrep,1);
        Xm(:,2:25)=CAlogall(:,start-1:final-1)';
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%2x1
        
        forecVARlog(j,i)=[1 CAlogall(:,final)']*CAlogpricebeta;
    end
end

err_forcVAR=forecVARlog-CAlogall(:,301:343);%24x43
RMSEforcVAR=sqrt(mean(err_forcVAR.^2,2)) %24x1



for ind=301:343
figure(ind)
plot(electimescaled,CAlogall(:,ind),'bo','linewidth',1.5)
hold
fdraw=fd(CAcoef_true(:,ind-300),CAfourierb);
plotfdtype(fdraw,'b-',2)
fdfore=fd(CAcoef_forc(:,ind-300),CAfourierb);
plotfdtype(fdfore,'r--',2)
fdFAR300=fd(CAcoef_forcFAR(:,ind-300),CAfourierb)
plotfdtype(fdFAR300,'g--',2)
plot(electimescaled,forecARXlog(:,ind-300),'k--')
plot(electimescaled,forecVARlog(:,ind-300),'r--')
legend('raw','Bspline','AFAR forecast','FAR300','ARX','VAR','location','southeast')
title(['1-step ahead forecast for curve ',num2str(ind)],'fontsize',12)
axis([0,1,2,6]);
xl={0,6,12,18,24};
set(gca,'XTick',0:0.25:1)
set(gca,'XTicklabel',xl)
xlabel('Time(Hours)','fontsize',12)
ylabel('Log prices','fontsize',12)
hold
date=[num2str(ind),'.jpg'];
%date=[num2str(i+299),'.jpg'];
saveas(gcf,date)
close
end


CAfdval=eval_fd(electimescaled,CAfourierfd);

ind=303;
figure(ind)
plot(electimescaled,CAlogall(:,ind),'bo','linewidth',1.5)
hold
fdraw=fd(CAcoef_true(:,ind-300),CAfourierb);
plotfdtype(fdraw,'b-',1.5)
%plot(electimescaled,CAfdval(:,ind),'b-o','linewidth',1.5)
fdfore=fd(CAcoef_forc(:,ind-300),CAfourierb);
plotfdtype(fdfore,'r--',1.5)
%plot(electimescaled,CAfd_forcval(:,ind-300),'r--','linewidth',1.5)
fdFAR300=fd(CAcoef_forcFAR(:,ind-300),CAfourierb)
plotfdtype(fdFAR300,'g--',1.5)
%plot(electimescaled,CAfd_forcvalFAR(:,ind-300),'g-.','linewidth',1.5)
plot(electimescaled,forecARXlog(:,ind-300),'k--','linewidth',1.5)
plot(electimescaled,forecVARlog(:,ind-300),'y--','linewidth',1.5)
legend('','raw','AFAR','FAR300','ARX','VAR','location','southeast')
title('1-step ahead forecast: Date 2 May 2000','fontsize',12)
%axis([0,1,2,6]);
xl={0,6,12,18,24};
set(gca,'XTick',0:0.25:1)
set(gca,'XTicklabel',xl)
xlabel('Time(hours)','fontsize',12)
ylabel('Log prices','fontsize',12)
hold





