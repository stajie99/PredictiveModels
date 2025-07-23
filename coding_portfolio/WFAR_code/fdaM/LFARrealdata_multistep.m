% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%------------Electricity data-------------------------------
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
step=1;%we change step to 1,3,7,14 and 30 for multi-step ahead forecasts

%import the data 'CAhourly' and add the package 'fdaM' to the path
load('CAhourly.csv')
addpath(genpath('fdaM'))  
averprice=CAhourly(:,3);
%avepricematrix=reshape(CAhourly(:,3),24,1037);
avepricematrix=reshape(averprice,24,1037);
CAlogall=log(avepricematrix(:,461:803));%24x343;   natural logrithm e


electime=1:24;
electimescaled=zeros(24,1);
for i=1:24
    electimescaled(i)=electime(i)/24;
end
%Create the fd objects by fourier transformation
CAnbasis=23;
CAfourierb=create_fourier_basis([0,1],CAnbasis,1);
%eval_penalty(CAfourierb,int2Lfd(0))

CAfourierfd=smooth_basis(electimescaled,CAlogall,CAfourierb);

% plot(CAfourierfd)
% title('CAlog_fd new')

% figure(1)
% %fourierbasis=create_fourier_basis([1,24],nbasis);
% %logfdall=smooth_basis(electime,CAlogall,fourierbasis)
% plot(CAfourierfd)
% axis([0,1,1.5,6.5])
% hold
% plot(electimescaled,CAlogall,'x')
% title('Smoothed curves 5 Jul 1999--11 Jun 2000','fontsize',12)
% xlabel('Time (Hours)','fontsize',12)
% ylabel('Log Price','fontsize',12)
% hold
% 
% 
% 
% figure(2)
% fd10=fd(CAfourierfdcoef(:,10),CAfourierb);
% plotfdtype(fd10,'r--',2)
% hold
% plot(electimescaled,CAlogall(:,10),'bo','linewidth',2)
% hold
% 
% CAfourierb2=create_fourier_basis([0,24],CAnbasis);
% eval_penalty(CAfourierb2,int2Lfd(0))
% CAfd2=smooth_basis(electime,CAlogall,CAfourierb2);
% CAcoef2=getcoef(CAfd2);
% figure(4)
% fd12_2=fd(CAcoef2(:,12),CAfourierb2);
% plotfdtype(fd12_2,'r--',2)
% hold
% plot(electime,CAlogall(:,12),'bo','linewidth',2)
% hold
% axis([0,24,2.5,3.7])
% xlabel('Time(Hours)','fontsize',12)
% ylabel('Log price','fontsize',12)
% title('Electricity log prices for 16 Jul 1999','fontsize',12)
% legend('Smooth curve by Fourier','Raw','location','southeast')
% 
% figure(3)
% timeY=zeros(24,343);
% datesX=zeros(24,343);
% for i=1:343
%     timeY(:,i)=electime;%24x343
% end
% x=1:343;
% for i=1:24
%     datesX(i,:)=x;%24x343
% end
% plot3(datesX,timeY,CAlogall,'bo')
% axis([1,343,0,24,1,7])
% set(gca,'YTick',0:2:24)
% set(gca,'XTick',1:60:343)
% %eleclabel={'Jul99','Aug99','Sep99','Oct99','Nov99','Dec99','Jan00','Feb00','Mar00','Apr00','May00','Jun00'}
% eleclabel={'Jul99','Sep99','Nov99','Jan00','Mar00','May00'}
% set(gca,'XTickLabel',eleclabel);
% xlabel('Dates','Fontsize',12)
% ylabel('Time(Hours)','Fontsize',12)
% zlabel('Log price','Fontsize',12)
% grid on
% title('Raw electricity log prices 5 Jul 1999--11 Jun 2000','Fontsize',12)
% 
% figure(4)
% timeY=zeros(24,343);
% datesX=zeros(24,343);
% for i=1:343
%     timeY(:,i)=electime;%24x343
% end
% x=1:343;
% for i=1:24
%     datesX(i,:)=x;%24x343
% end
% plot3(datesX,timeY,CAlogall)
% axis([1,343,0,24,1,7])
% set(gca,'YTick',0:2:24)
% set(gca,'XTick',1:60:343)
% %eleclabel={'Jul99','Aug99','Sep99','Oct99','Nov99','Dec99','Jan00','Feb00','Mar00','Apr00','May00','Jun00'}
% eleclabel={'Jul99','Sep99','Nov99','Jan00','Mar00','May00'}
% set(gca,'XTickLabel',eleclabel);
% xlabel('Dates','Fontsize',12)
% ylabel('Time(Hours)','Fontsize',12)
% zlabel('Log price','Fontsize',12)
% grid on
% title('Smoothed electricity log price curves 5 Jul 1999--11 Jun 2000','Fontsize',12)
% 
% %----autocorr and crosscorr---------------------------------
% nhr=size(CAlogall,1);
% CAACFnum=zeros(21,nhr);
% for i=1:nhr
%     [CAACFnum(:,i),lags,bounds]=autocorr(CAlogall(i,:));
%     figure(i)
%     autocorr(CAlogall(i,:))
%     title([num2str(i),' CA sample ACF'])
%     saveas(gcf,[num2str(i),'CA sacf.jpg'])
%     close
% end
% for i=1:nhr-1
%     figure(i)
%     crosscorr(CAlogall(i,:),CAlogall(i+1,:))
%     %axis([-20,20,-0.2,1.2])
%     title([num2str(i),'and',num2str(i+1),' CA cross-correlation'])
%     saveas(gcf,[num2str(i),'and',num2str(i+1),' CA xcf.jpg'])
%     close
% end
% [XCF1and2,lags,bounds]=crosscorr(yieldterm1(1,:),yieldterm1(2,:))
% 
% for i=1:nhr
%     subplot(6,4,i),autocorr(CAlogall(i,:))
%     title(['Sample ACF at ',num2str(i),':00'],'fontsize',9)
%     xlabel('');
% end
% 
% for i=1:nhr-1
%     subplot(6,4,i),crosscorr(CAlogall(i,:),CAlogall(i+1,:))
%     title(['CCF for ',num2str(i),':00 vs ',num2str(i+1),':00'],'fontsize',9)
%     ylabel('');
%     xlabel('');
% end
% 
% autocorr(CAlogall(9,:))
% title('Sample ACF at 9:00 for the whole electricity data set','fontsize',12)
% axis([0,20,-0.4,1.1])
% 
% crosscorr(CAlogall(8,:),CAlogall(9,:))
% title('Cross-correlation 8:00 vs 9:00 for the whole electricity data set','fontsize',12)
% axis([-20,20,-0.4,1.1])
% 
% autocorr(CAsub1(9,:))
% title('Sample ACF at 9:00 for the first 50 electricity log price curves','fontsize',12)
% axis([0,20,-0.4,1.1])
% 
% crosscorr(CAsub1(8,:),CAsub1(9,:))
% title('Cross-correlation 8:00 vs 9:00 for the first 50 electricity log price curves','fontsize',12)
% axis([-20,20,-0.4,1.1])
% %non-stationarity
% CAsub1=CAlogall(:,1:50);
% CAsub2=CAlogall(:,294:343);
% CAACFnumsub1=zeros(21,nhr);
% for i=1:nhr
%     [CAACFnumsub1(:,i),lags,bounds]=autocorr(CAsub1(i,:));
%     figure(i)
%     autocorr(CAsub1(i,:))
%     title([num2str(i),' CA subsample1 ACF'])
%     saveas(gcf,[num2str(i),'CA s1acf.jpg'])
%     close
% end
% CAACFnumsub2=zeros(21,nhr);
% for i=1:nhr
%     [CAACFnumsub2(:,i),lags,bounds]=autocorr(CAsub2(i,:));
%     figure(i)
%     autocorr(CAsub2(i,:))
%     title([num2str(i),' CA subsample2 ACF'])
%     saveas(gcf,[num2str(i),' CA s2acf.jpg'])
%     close
% end
% 
% 
% for i=1:nhr-1
%     figure(i)
%     crosscorr(CAsub1(i,:),CAsub1(i+1,:))
%     %axis([-20,20,-0.8,1.2])
%     title([num2str(i),'and',num2str(i+1),' CA cross-correlation1'])
%     saveas(gcf,[num2str(i),'and',num2str(i+1),'CA x1cf.jpg'])
%     close
% end
% for i=1:nhr-1
%     figure(i)
%     crosscorr(CAsub2(i,:),CAsub2(i+1,:))
%     %axis([-20,20,-0.8,1.2])
%     title([num2str(i),'and',num2str(i+1),' CA cross-correlation2'])
%     saveas(gcf,[num2str(i),'and',num2str(i+1),'CA x2cf.jpg'])
%     close
% end
% figure(1)
% CAfdsub1=smooth_basis(electimescaled,CAsub1,CAfourierb);
% CAfdsub2=smooth_basis(electimescaled,CAsub2,CAfourierb);
% plotfdtype(mean(CAfdsub1),'b-',2)
% hold
% plotfdtype(mean(CAfdsub2),'r-',2)
% legend('mean CAsub1','mean CAsub2')
% title('mean functions for two subsamples','fontsize',12)
% 
% %covariance surface
% CAcov=cov(CAlogall');
% size(CAcov)
% [X,Y]=meshgrid(electime,electime);
% surf(X,Y,CAcov);
% axis([1,24,1,24,0.04,0.2])
% xlabel('Time(hours)','fontsize',12)
% ylabel('Time(hours)','fontsize',12)
% zlabel('Covariance','fontsize',12)
% title('CA Covariance surface','fontsize',12)
% 
% figure(1)
% CAcovsub1=cov(CAsub1');
% size(CAcovsub1)
% [X,Y]=meshgrid(electime,electime);
% surf(X,Y,CAcovsub1);
% axis([1,24,1,24,0,0.4])
% xlabel('Time(hours)','fontsize',12)
% ylabel('Time(hours)','fontsize',12)
% zlabel('Covariance','fontsize',12)
% title('CAsub1 Covariance surface','fontsize',12)
% 
% figure(2)
% CAcovsub2=cov(CAsub2');
% size(CAcovsub2)
% [X,Y]=meshgrid(electime,electime);
% surf(X,Y,CAcovsub2);
% axis([1,24,1,24,0,0.4])
% xlabel('Time(hours)','fontsize',12)
% ylabel('Time(hours)','fontsize',12)
% zlabel('Covariance','fontsize',12)
% title('CAsub2 Covariance surface','fontsize',12)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%---insample est and forecasts-----------------------------------
%The recursive forecasts are from 301 to 343 and in-sample est from 300 to
%343
%we calculate the CV first
CAfourierfdcoef=getcoef(CAfourierfd);%23x343
ncurve=size(CAfourierfdcoef,2);%343

CAsub=CAlogall(:,1:300);
CAsubcoef=CAfourierfdcoef(:,1:300);
% fdsub=fd(CAsubcoef,CAfourierb);
% plot(fdsub)
% axis([0,1,1.5,5.5])

if step==14
    LFARinterval=[30,45,60,90,120,180,240,300];
elseif step==30
    LFARinterval=[45,60,90,120,180,240,300];
else
    LFARinterval=[14,30,45,60,90,120,180,240,300];
end

ninter=length(LFARinterval);

ceptsub=zeros(CAnbasis,1); % I_[0,1]£¬cosine and sine are three basis.
chatsub=zeros((CAnbasis-1)/2+1,1);
sigma2hatsub=zeros((CAnbasis-1)/2+1,1);
[ceptsub,chatsub,sigma2hatsub,LLFsub]=LFARmle_cept(CAsubcoef,step);
% %---check sieve constraint----------------------
% Kseries=1:2;
% Kseries.^2*chatsub(2:3).^2
% %------------------------------------------------
ngene=10;%We generate 1000 processes with 300 curves in each process first.
nrep=300;%There are 300 curves for each process.
totalK=(CAnbasis-1)/2;
coefgene=ones(CAnbasis,nrep+step,ngene);%23x301x1000
coefgene(1,1:step,:)=coefgene(1,1:step,:)*(ceptsub(1)/(1-chatsub(1)));
for i=2:CAnbasis
    index=floor(i/2);
    coefgene(i,1:step,:)=coefgene(i,1:step,:).*(ceptsub(i)/(1-chatsub(index+1)/sqrt(2)));
end

%Fill up the first row of the coef matrix first
for i=1:ngene %1 of the 1000 processes; for one of the 3x300 matrices 
    error_0=randn(nrep,1);
    for j=step+1:nrep+step 
        coefgene(1,j,i)=ceptsub(1)+chatsub(1)*coefgene(1,j-step,i)+sqrt(sigma2hatsub(1))*error_0(j-step);
    end
%Fill up the rest
    errorK=randn(nrep,2*totalK);
    for k=1:totalK
        for j=step+1:nrep+step
            coefgene(2*k,j,i)=ceptsub(2*k)+1/sqrt(2)*chatsub(k+1)*coefgene(2*k,j-step,i)+sqrt(sigma2hatsub(k+1))*errorK(j-step,2*k-1);
            coefgene(2*k+1,j,i)=ceptsub(2*k+1)+1/sqrt(2)*chatsub(k+1)*coefgene(2*k+1,j-step,i)+sqrt(sigma2hatsub(k+1))*errorK(j-step,2*k);
        end
    end
end
coefgene=coefgene(:,step+1:nrep+step,:);%23x300x1000

nprocess=size(coefgene,3);%1000 %5000
LRval=zeros(nprocess,1);
c_hatMgene=zeros(totalK+1,nprocess);
cept_hatMgene=zeros(CAnbasis,nprocess);
sigma2_hatMgene=zeros(totalK+1,nprocess);
LLFvalgene=zeros(nprocess,1);

for i=1:nprocess
    i
    [cept_hatMgene(:,i) c_hatMgene(:,i),sigma2_hatMgene(:,i),LLFvalgene(i)]=LFARmle_cept(coefgene(:,:,i),step);
    LRval(i)=sqrt(abs(LLFvalgene(i)-LFARllf_cept(ceptsub,chatsub,sigma2hatsub,coefgene(:,:,i),step,totalK)));
end
% %------check the sieve constraints--------
% Kseries=1:11;
% Kseries*c_hatMgene(2:12,100).^2%>11
% Kseries.^2*c_hatMgene(2:12,100).^2%>11
% sum(abs(c_hatMgene(2:12,100)))
% %%-----------------------------------------------------------
%LLFzeros=find(LLFvalgene==0);
ksiLFAR=sum(LRval)/nprocess; %4.8454 %4.8423 %4.8438 %4.8248(14) %4.8412(30) %4.8417(60) %4.8604(90) %4.8511(120) %4.8605(180) %4.8395(240)

global LLFM c_hatM sigma2_hatM cept_hatM %all data which will be used later are stored here!
c_hatM=zeros(totalK+1,ninter,nprocess);
sigma2_hatM=zeros(totalK+1,ninter,nprocess);
cept_hatM=zeros(CAnbasis,ninter,nprocess);
LLFM=zeros(ninter,nprocess);
ncurve=size(coefgene,2);%300
LLFM_true=zeros(ninter,nprocess);
for i=1:nprocess
    i
    cepttemp=zeros(CAnbasis,ninter);ctemp=zeros(totalK+1,ninter);sigma2temp=zeros(totalK+1,ninter);LLFtemp=zeros(ninter,1);LLFtemp_true=zeros(ninter,1);
    for j=1:ninter
        LFARinterval(j)
        [cepttemp(:,j),ctemp(:,j),sigma2temp(:,j),LLFtemp(j)]=LFARmle_cept(coefgene(:,ncurve-LFARinterval(j)+1:ncurve,i),step);
        LLFtemp_true(j)=LFARllf_cept(ceptsub,chatsub,sigma2hatsub,coefgene(:,ncurve-LFARinterval(j)+1:ncurve,i),step,totalK);
    end
    c_hatM(:,:,i)=ctemp;
    sigma2_hatM(:,:,i)=sigma2temp;
    cept_hatM(:,:,i)=cepttemp;
    LLFM(:,i)=LLFtemp;
    LLFM_true(:,i)=LLFtemp_true;
end
%length(find((LLFM-LLFM_true)<=0)) %should be 0
% %%%-------check the sieve constraints----------------------------
% Kseries*c_hatM(2:12,9,1).^2%most time>11
% Kseries.^2*c_hatM(2:12,9,1).^2%>11
% sum(abs(c_hatM(2:12,3,1)))
% %%----------------------------------------------------------------
%%----------------------------------------------------------------
%----calculate the ciritical values-----------------------
%------------------------------------------------------------------

format LongG
[CvalueLFAR,LRcompare]=LFARCvalue_cept(coefgene,ksiLFAR,nprocess,LFARinterval,step);
%CvalueLFAR=[1000000;11.2;5.4;3.9;4.6;3.5;4.1;3.2;2.5];%use mle from curve286-300 as underlying 
%CvalueLFAR=[1000000;15.3;6.2;4.3;4.8;3.6;4.2;3.2;2.5];%use mle from curve270-300 as underlying
%CvalueLFAR=[10000;16.8;6.4;4.6;5;3.7;4.3;3.3;2.6];
%CvalueLFAR=[10000;15.4;6.5;4.4;5.1;3.8;4.2;3.3;2.6];%use mle from curve240-300 as underlying
%CvalueLFAR=[10000;18.6;6.8;4.8;5.3;3.9;4.3;3.3;2.6];%use mle from curve210-300 as underlying
%CvalueLFAR=[10000;16.2;6.8;4.7;5.1;3.9;4.3;3.3;2.6];%use mle from curve180-300 as underlying
%CvalueLFAR=[10000;14.1;6.3;4.5;5;3.7;4.2;3.2;2.6];%use mle from curve120-300 as underlying
%CvalueLFAR=[10000;18.6;6.4;4.5;5.2;3.7;4.3;3.2;2.6];%use mle from curve60-300 as underlying

% plot(2:9,CvalueLFAR_1(2:9),'linewidth',2)
% axis([1,9,2,16])
% title('Electricity prices: critical values for 2nd to 9th candidate intervals','fontsize',12)
% xlabel('Index of candidate intervals','fontsize',12)
% ylabel('Critical values','fontsize',12)

nforc=343-(300+step)+1; % no. of curves to forecast
CAinter=zeros(nforc,1);
CAchat=zeros(totalK+1,nforc);%12x43
CAsigma2hat=zeros(totalK+1,nforc);%12x43
CAcepthat=zeros(CAnbasis,nforc);

%matlabpool open local 2
%pmode start local 4
%parfor i=1:npro

for i=1:nforc
    [CAinter(i),CAcepthat(:,i),CAchat(:,i),CAsigma2hat(:,i)]=LFARestsimu_cept(CAfourierfdcoef(:,i:i+299),CvalueLFAR,LFARinterval,step);
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
% %plot of K, there are 43 estimations of K for each time point 
% %CAchat(12x43) contains the fourier coef of K
% coefK=zeros(23,43);
% coefK(1,:)=CAchat(1,:);
% for i=1:11
%     coefK(2*i+1,:)=CAchat(i+1,:);
%     coefK(2*i,:)=zeros(1,43);
% end
% kernelK=fd(coefK,CAfourierb);
% plot(kernelK)
% 
% title('Estimated kernel function K','fontsize',12)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CAcoef_forc=zeros(CAnbasis,nforc);
for i=1:nforc
    CAcoef_forc(1,i)=CAcepthat(1,i)+CAchat(1,i)*CAfourierfdcoef(1,i+299);
    for j=1:totalK
        CAcoef_forc(2*j,i)=CAcepthat(2*j,i)+1/sqrt(2)*CAchat(j+1,i)*CAfourierfdcoef(2*j,i+299);
        CAcoef_forc(2*j+1,i)=CAcepthat(2*j+1,i)+1/sqrt(2)*CAchat(j+1,i)*CAfourierfdcoef(2*j+1,i+299);
    end
end

% CAcoef_true=CAfourierfdcoef(:,(300+step):343);
% phaseindex=1:nforc;
% plotmovieLFAR(CAcoef_forc,CAcoef_true,CAfourierb,phaseindex,step)

CAfd_forc=fd(CAcoef_forc,CAfourierb);
%CAfd_true=fd(CAcoef_true,CAfourierb);
CAfd_forcval=eval_fd(electimescaled,CAfd_forc);%24x43
%CAfd_trueval=eval_fd(electimescaled,CAfd_true);%24x43
err_forc=CAfd_forcval-CAlogall(:,300+step:343);%24x43
format shortG
RMSEforc=sqrt(mean(err_forc.^2,2)); %24x1



% CAcoef_forc_try=zeros(CAnbasis,nforc);
% for i=1:nforc
%     CAcoef_forc_try(1,i)=CAcepthat_try(1,i)+CAchat_try(1,i)*CAfourierfdcoef(1,i+299);
%     for j=1:totalK
%         CAcoef_forc_try(2*j,i)=CAcepthat_try(2*j,i)+1/sqrt(2)*CAchat_try(j+1,i)*CAfourierfdcoef(2*j,i+299);
%         CAcoef_forc_try(2*j+1,i)=CAcepthat_try(2*j+1,i)+1/sqrt(2)*CAchat_try(j+1,i)*CAfourierfdcoef(2*j+1,i+299);
%     end
% end
% 
% %CAcoef_true=CAfourierfdcoef(:,(300+step):343);
% %phaseindex=1:nforc;
% %plotmovieLFAR(CAcoef_forc,CAcoef_true,CAfourierb,phaseindex,step)
% 
% CAfd_forc_try=fd(CAcoef_forc_try,CAfourierb);
% %CAfd_true=fd(CAcoef_true,CAfourierb);
% CAfd_forcval_try=eval_fd(electimescaled,CAfd_forc_try);%24x43
% %CAfd_trueval=eval_fd(electimescaled,CAfd_true);%24x43
% err_forc_try=CAfd_forcval_try-CAlogall(:,301:343);%24x43
% RMSEforc_try=sqrt(mean(err_forc_try.^2,2)) %24x1

% %------forecasting by using FAR(all past data)----------------------
% CAchat_FARall=zeros(totalK+1,nforc);%12x43
% CAsigma2hat_FARall=zeros(totalK+1,nforc);%12x43
% CAcepthat_FARall=zeros(CAnbasis,nforc);
% 
% %matlabpool open local 2
% %pmode start local 4
% %parfor i=1:npro
% for i=1:nforc
%     [CAcepthat_FARall(:,i),CAchat_FARall(:,i),CAsigma2hat_FARall(:,i)]=LFARmle_cept(CAfourierfdcoef(:,1:i+299),1);
% end
% 
% CAcoef_forcFARall=zeros(CAnbasis,nforc);
% for i=1:nforc
%     CAcoef_forcFARall(1,i)=CAcepthat_FARall(1,i)+CAchat_FARall(1,i)*CAfourierfdcoef(1,i+299);
%     for j=1:totalK
%         CAcoef_forcFARall(2*j,i)=CAcepthat_FARall(2*j,i)+1/sqrt(2)*CAchat_FARall(j+1,i)*CAfourierfdcoef(2*j,i+299);
%         CAcoef_forcFARall(2*j+1,i)=CAcepthat_FARall(2*j+1,i)+1/sqrt(2)*CAchat_FARall(j+1,i)*CAfourierfdcoef(2*j+1,i+299);
%     end
% end
% 
% CAfd_forcFARall=fd(CAcoef_forcFARall,CAfourierb);
% CAfd_forcvalFARall=eval_fd(electimescaled,CAfd_forcFARall);%24x43
% err_forcFARall=CAfd_forcvalFARall-CAlogall(:,301:343);%24x43
% RMSEforc_FARall=sqrt(mean(err_forcFARall.^2,2)) %24x1
% [RMSEforc_FARall,RMSEforc_FAR]


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
err_forcFAR=CAfd_forcvalFAR-CAlogall(:,300+step:343);%24x43
RMSEforc_FAR=sqrt(mean(err_forcFAR.^2,2)); %24x1
%[RMSEforc_try,RMSEforc_FAR]

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
err_forcFAR150=CAfd_forcvalFAR150-CAlogall(:,300+step:343);%24x43
RMSEforc_FAR150=sqrt(mean(err_forcFAR150.^2,2)); %24x1
% index=1:24;
% colums=[index' RMSEforc_try,RMSEforc_FAR,RMSEforc_FAR150]

%------forecasting by ARX (Weron (2008))--------------------------------
CAforec_load=reshape(CAhourly(:,5),24,1037);%The forecasted load
CAforec_actload=CAforec_load(:,461:803);%The actual forecasted load for out-of-sample forecast
CAforec_logload=log(CAforec_actload);%24x343 for the log of forecasted load
%%%%%begf=274;%nrep=63;
begf=300+step;nforc=343-begf+1;

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
    final=begf+i-(step+1);
    start=step+7;
    nrep=final-start+1;
    Xm=zeros(nrep,8);
    finalmod=mod(final,7);
    Xm(:,6:8)=DumMat49(start:final,:);
    
       %j=2; 
    for j=1:24
        yv=CAlogall(j,start:final)';
        Xm(:,1)=CAlogall(j,start-step:final-step)';
        Xm(:,2)=CAlogall(j,start-(step+1):final-(step+1))';
        Xm(:,3)=CAlogall(j,start-(step+6):final-(step+6))';
        Xm(:,4)=mCAlog(start-step:final-step);
        Xm(:,5)=CAforec_logload(j,start:final)';
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%8x1
        
        forecARXlog(j,i)=[CAlogall(j,final) CAlogall(j,final-1) CAlogall(j,final-6) mCAlog(final) CAforec_logload(j,final+step) DumMat49(final+step,:)]*CAlogpricebeta;
    end
end

err_forcARX=forecARXlog-CAlogall(:,300+step:343);%24x43
RMSEforcARX=sqrt(mean(err_forcARX.^2,2)); %24x1

%----------------------------------------------------------------------
%------forecasting by AR(1)--------------------------------------------
begf=300+step;nforc=343-begf+1;

forecARlog=zeros(24,nforc);
%i=1;
for i=1:nforc
    final=begf+i-(step+1);
    start=step+1;
    nrep=final-start+1;
    Xm=zeros(nrep,2);
    
    for j=1:24
        yv=CAlogall(j,start:final)';
        Xm(:,1)=ones(nrep,1);
        Xm(:,2)=CAlogall(j,start-step:final-step)';
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%2x1
        
        forecARlog(j,i)=[1 CAlogall(j,final)]*CAlogpricebeta;
    end
end

err_forcAR=forecARlog-CAlogall(:,300+step:343);%24x43
RMSEforcAR=sqrt(mean(err_forcAR.^2,2)); %24x1
%-----------------------------------------------------------------------
%------forecasting by seasonalAR (without X variables in ARX)------------
begf=300+step;nforc=343-begf+1;

forecARsealog=zeros(24,nforc);
%i=1;
for i=1:nforc
    final=begf+i-(step+1);
    start=step+7;
    nrep=final-start+1;
    Xm=zeros(nrep,3);
   
    for j=1:24
        yv=CAlogall(j,start:final)';
        Xm(:,1)=CAlogall(j,start-step:final-step)';
        Xm(:,2)=CAlogall(j,start-(step+1):final-(step+1))';
        Xm(:,3)=CAlogall(j,start-(step+6):final-(step+6))';
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%8x1
        
        forecARsealog(j,i)=[CAlogall(j,final) CAlogall(j,final-1) CAlogall(j,final-6)]*CAlogpricebeta;
    end
end

err_forcARsea=forecARsealog-CAlogall(:,300+step:343);%24x43
RMSEforcARsea=sqrt(mean(err_forcARsea.^2,2)); %24x1
%------------------------------------------------------------------------
%------forecasting by VAR(1)---------------------------------------------
begf=300+step;nforc=343-begf+1;
forecVARlog=zeros(24,nforc);

for i=1:nforc
    final=begf+i-(step+1);
    start=step+1;
    nrep=final-start+1;
    Xm=zeros(nrep,25);
    
    for j=1:24
        yv=CAlogall(j,start:final)';
        Xm(:,1)=ones(nrep,1);
        Xm(:,2:25)=CAlogall(:,start-step:final-step)';
        
        result=ols(yv,Xm);
        CAlogpricebeta=result.beta;%2x1
        
        forecVARlog(j,i)=[1 CAlogall(:,final)']*CAlogpricebeta;
    end
end

err_forcVAR=forecVARlog-CAlogall(:,300+step:343);%24x43
RMSEforcVAR=sqrt(mean(err_forcVAR.^2,2)); %24x1

RMSEresults=[RMSEforc,RMSEforc_FAR,RMSEforc_FAR150,RMSEforcVAR,RMSEforcARX,RMSEforcAR,RMSEforcARsea];

rowLabels = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'};
columnLabels = {'AFAR','FAR(300)','FAR(150)','VAR(1)','ARX','AR(1)','ARseasonal'};
matrix2latex(RMSEresults, ['RMSE_AFARreal_step_' num2str(step) '.tex'], 'rowLabels', rowLabels, 'columnLabels', columnLabels, 'alignment', 'c', 'format', '%-6.3f');

str=['AFAR_realdata_step_' num2str(step) '.mat'];
save(str)

[minv,ind]=min(RMSEresults,[],2);
index=1:24;
[index' ind minv]

%%%%%%%%%%-----forecast plot------------

CAcoef_true=CAfourierfdcoef(:,(300+step):343);

%%%%forecast curves for 2 May 2000
ind=303;
figure(ind)
plot(electimescaled,CAlogall(:,ind),'bo','linewidth',1.5)
hold
fdraw=fd(CAcoef_true(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdraw,'b-',1.5)
%plot(electimescaled,CAfdval(:,ind),'b-o','linewidth',1.5)
fdfore=fd(CAcoef_forc(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdfore,'r-',1.5)
%plot(electimescaled,CAfd_forcval(:,ind-300),'r--','linewidth',1.5)
fdFAR300=fd(CAcoef_forcFAR(:,ind-(300+step)+1),CAfourierb)
plotfdtype(fdFAR300,'r-.',1.5)
%plot(electimescaled,CAfd_forcvalFAR(:,ind-300),'g-.','linewidth',1.5)
plot([0;electimescaled],[forecARXlog(24,ind-(300+step)+1);forecARXlog(:,ind-(300+step)+1)],'b--','linewidth',1.5)
plot([0;electimescaled],[forecVARlog(24,ind-(300+step)+1);forecVARlog(:,ind-(300+step)+1)],'g-.','linewidth',1.5)
legend('raw','smoothed','AFAR','FAR300','ARX','VAR','location','southeast')
title([num2str(step) '-step ahead forecast: Date 2 May 2000'],'fontsize',12)
axis([0,1,2,4.5]);
xl={0,6,12,18,24};
set(gca,'XTick',0:0.25:1)
set(gca,'XTicklabel',xl)
xlabel('Time(hours)','fontsize',12)
ylabel('Log prices','fontsize',12)
hold

%%%%forecast curves for 15 May 2000
ind=316;
figure(ind)
plot(electimescaled,CAlogall(:,ind),'bo','linewidth',3,'Markersize',8)
hold
fdraw=fd(CAcoef_true(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdraw,'b-',3)
%plot(electimescaled,CAfdval(:,ind),'b-o','linewidth',1.5)
fdfore=fd(CAcoef_forc(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdfore,'r-',3)
%plot(electimescaled,CAfd_forcval(:,ind-300),'r--','linewidth',1.5)
fdFAR300=fd(CAcoef_forcFAR(:,ind-(300+step)+1),CAfourierb)
plotfdtype(fdFAR300,'r-.',3)
%plot(electimescaled,CAfd_forcvalFAR(:,ind-300),'g-.','linewidth',1.5)
plot([0;electimescaled],[forecARXlog(24,ind-(300+step)+1);forecARXlog(:,ind-(300+step)+1)],'b--','linewidth',3)
plot([0;electimescaled],[forecVARlog(24,ind-(300+step)+1);forecVARlog(:,ind-(300+step)+1)],'g-.','linewidth',3)
legend('raw','smoothed','AFAR','FAR300','ARX','VAR','location','southeast')
title([num2str(step) '-step ahead forecast: Date 15 May 2000'],'fontsize',12)
axis([0,1,1.8,4.6]);
xl={0,6,12,18,24};
set(gca,'XTick',0:0.25:1)
set(gca,'XTicklabel',xl)
xlabel('Time(hours)','fontsize',12)
ylabel('Log prices','fontsize',12)
hold

%%%%forecast curves for 29 May 2000
ind=330;
figure(ind)
plot(electimescaled,CAlogall(:,ind),'bo','linewidth',3,'MarkerSize',8)
hold
fdraw=fd(CAcoef_true(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdraw,'b-',3)
%plot(electimescaled,CAfdval(:,ind),'b-o','linewidth',1.5)
fdfore=fd(CAcoef_forc(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdfore,'r-',3)
%plot(electimescaled,CAfd_forcval(:,ind-300),'r--','linewidth',1.5)
fdFAR300=fd(CAcoef_forcFAR(:,ind-(300+step)+1),CAfourierb)
plotfdtype(fdFAR300,'r-.',3)
%plot(electimescaled,CAfd_forcvalFAR(:,ind-300),'g-.','linewidth',1.5)
plot([0;electimescaled],[forecARXlog(24,ind-(300+step)+1);forecARXlog(:,ind-(300+step)+1)],'b--','linewidth',3)
plot([0;electimescaled],[forecVARlog(24,ind-(300+step)+1);forecVARlog(:,ind-(300+step)+1)],'g-.','linewidth',3)
legend('raw','smoothed','AFAR','FAR300','ARX','VAR','location','southeast')
title([num2str(step) '-step ahead forecast: Date 29 May 2000'],'fontsize',12)
axis([0,1,1.8,4.6]);
xl={0,6,12,18,24};
set(gca,'XTick',0:0.25:1)
set(gca,'XTicklabel',xl)
xlabel('Time(hours)','fontsize',12)
ylabel('Log prices','fontsize',12)
hold

%%%%forecast curves for 3 June 2000
ind=335;
figure(ind)
plot(electimescaled,CAlogall(:,ind),'bo','linewidth',3,'MarkerSize',8)
hold
fdraw=fd(CAcoef_true(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdraw,'b-',3)
%plot(electimescaled,CAfdval(:,ind),'b-o','linewidth',1.5)
fdfore=fd(CAcoef_forc(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdfore,'r-',3)
%plot(electimescaled,CAfd_forcval(:,ind-300),'r--','linewidth',1.5)
fdFAR300=fd(CAcoef_forcFAR(:,ind-(300+step)+1),CAfourierb)
plotfdtype(fdFAR300,'r-.',3)
%plot(electimescaled,CAfd_forcvalFAR(:,ind-300),'g-.','linewidth',1.5)
plot([0;electimescaled],[forecARXlog(24,ind-(300+step)+1);forecARXlog(:,ind-(300+step)+1)],'b--','linewidth',3)
plot([0;electimescaled],[forecVARlog(24,ind-(300+step)+1);forecVARlog(:,ind-(300+step)+1)],'g-.','linewidth',3)
legend('raw','smoothed','AFAR','FAR300','ARX','VAR','location','southeast')
title([num2str(step) '-step ahead forecast: Date 3 June 2000'],'fontsize',12)
axis([0,1,1.8,4.6]);
xl={0,6,12,18,24};
set(gca,'XTick',0:0.25:1)
set(gca,'XTicklabel',xl)
xlabel('Time(hours)','fontsize',12)
ylabel('Log prices','fontsize',12)
hold

%%%%forecast curves for 8 June 2000
ind=340;
figure(ind)
plot(electimescaled,CAlogall(:,ind),'bo','linewidth',3,'Markersize',8)
hold
fdraw=fd(CAcoef_true(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdraw,'b-',3)
%plot(electimescaled,CAfdval(:,ind),'b-o','linewidth',1.5)
fdfore=fd(CAcoef_forc(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdfore,'r-',3)
%plot(electimescaled,CAfd_forcval(:,ind-300),'r--','linewidth',1.5)
fdFAR300=fd(CAcoef_forcFAR(:,ind-(300+step)+1),CAfourierb)
plotfdtype(fdFAR300,'r-.',3)
%plot(electimescaled,CAfd_forcvalFAR(:,ind-300),'g-.','linewidth',1.5)
plot([0;electimescaled],[forecARXlog(24,ind-(300+step)+1);forecARXlog(:,ind-(300+step)+1)],'b--','linewidth',3)
plot([0;electimescaled],[forecVARlog(24,ind-(300+step)+1);forecVARlog(:,ind-(300+step)+1)],'g-.','linewidth',3)
legend('raw','smoothed','AFAR','FAR300','ARX','VAR','location','southeast')
title([num2str(step) '-step ahead forecast: Date 8 June 2000'],'fontsize',12)
axis([0,1,1.8,4.6]);
xl={0,6,12,18,24};
set(gca,'XTick',0:0.25:1)
set(gca,'XTicklabel',xl)
xlabel('Time(hours)','fontsize',12)
ylabel('Log prices','fontsize',12)
hold

%%%%forecast curves for 11 June 2000
ind=343;
figure(ind)
plot(electimescaled,CAlogall(:,ind),'bo','linewidth',3,'Markersize',8)
hold
fdraw=fd(CAcoef_true(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdraw,'b-',3)
%plot(electimescaled,CAfdval(:,ind),'b-o','linewidth',1.5)
fdfore=fd(CAcoef_forc(:,ind-(300+step)+1),CAfourierb);
plotfdtype(fdfore,'r-',3)
%plot(electimescaled,CAfd_forcval(:,ind-300),'r--','linewidth',1.5)
fdFAR300=fd(CAcoef_forcFAR(:,ind-(300+step)+1),CAfourierb)
plotfdtype(fdFAR300,'r-.',3)
%plot(electimescaled,CAfd_forcvalFAR(:,ind-300),'g-.','linewidth',1.5)
plot([0;electimescaled],[forecARXlog(24,ind-(300+step)+1);forecARXlog(:,ind-(300+step)+1)],'b--','linewidth',3)
plot([0;electimescaled],[forecVARlog(24,ind-(300+step)+1);forecVARlog(:,ind-(300+step)+1)],'g-.','linewidth',3)
legend('raw','smoothed','AFAR','FAR300','ARX','VAR','location','southeast')
title([num2str(step) '-step ahead forecast: Date 11 June 2000'],'fontsize',12)
axis([0,1,1.8,4.6]);
xl={0,6,12,18,24};
set(gca,'XTick',0:0.25:1)
set(gca,'XTicklabel',xl)
xlabel('Time(hours)','fontsize',12)
ylabel('Log prices','fontsize',12)
hold
     
 
% %-------Diebold-Mariano test-------------------------------
% %AFAR and VAR(1), errors are in err_forc and err_forcVAR
% err2AFAR=err_forc.^2;%24x43
% err2VAR=err_forcVAR.^2;
% intercept_term=ones(43,1);
% beta1=zeros(2,24);
% DM1=zeros(24,1);
% for j=1:24
%     result=ols(err2AFAR(j,:)',[intercept_term err2VAR(j,:)']);
%     beta1(:,j)=result.beta;%2x1
%     DM1(j)=dmtest(err_forc(j,:)',err_forcVAR(j,:)',1);
% end
% %All absolute value of DM statistics < 1.96. not reject null of equal
% %predictive accuracy
    
%AFAR and ARX, errors in err_forc and err_forcARX 
%err2AFAR=err_forc.^2;%24x43
% err2ARX=err_forcARX.^2;
% beta2=zeros(2,24);
% DM2=zeros(24,1);
% for j=1:24
%     result=ols(err2AFAR(j,:)',[intercept_term err2ARX(j,:)']);
%     beta2(:,j)=result.beta;%2x1
%     DM2(j)=dmtest(err_forc(j,:)',err_forcARX(j,:)',1);
% end
% %only DM2>1.96 for 7am and 8am. By R function dm.test, we find ARX is
% %better for 7am and 8am
% 
% dmtest(RMSEforc,RMSEforcVAR) %||>1.96
% dmtest(RMSEforc,RMSEforcARX) %||>1.96
% 
% xlswrite('error_AFAR.xls',err_forc,'Sheet1','A1');
% xlswrite('error_ARX.xls',err_forcARX,'Sheet1','A1');
% xlswrite('error_VAR.xls',err_forcVAR,'Sheet1','A1');






















