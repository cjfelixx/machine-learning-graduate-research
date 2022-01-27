clc;clear all;close all;
load('simulateddata_1.mat');
%import simulated data using model 1.
%B,W are the coefficients in model 1
[p,N]=size(W);
B0=B*W;
%%plot the dense Y
figure;
for i=1:N
plot(trainY{i}','Color',[0.5,0.5,0.5],'LineWidth',1);
hold on;
end
xlabel('Time','LineWidth',2,'FontSize',14);
ylabel('Response','LineWidth',2,'FontSize',14);


%%%-----------------------Experement on Dense Sample------------------%%%%
%%------------Modeling learning-----------------%%

%%CDM
%%define the parameters
options=[];
options.maxIter = 1000;
options.Converge=0;
options.optimizeB=0;
alphmin=0;
alphmax=1;

%initial value

%find the optimal number of clusters
K_final = 2;
[B_CDM, W_CDM, nIter_CDM, objhistory_CDM] = CDM(trainY', trainX', K_final, [],options,[],[]);
% rmse_train = sqrt(dYIGM{i}'*dYIGM{i}/length(dYIGM{i}));
%goodness of fit
BCDM=B_CDM*W_CDM;
% dBCDM=B0-BCDM;
% sdB(1)=sum(sum(dBCDM))/N;
% vdB(1)=sqrt(sum(diag(dBCDM'*dBCDM))/(N*11));

% RMSE_train=mean(rmse_train);
% figure;
% boxplot(rmse_train,'labels',{'IGM','CDM','MEM','NCDM'});
% for i=1:N
%     setot(i,1)=(trainY{i}-mean(trainY{i}))*(trainY{i}-mean(trainY{i}))';
%     R2(i,1)=1-dYCDM(i,:)*dYCDM(i,:)'/setot(i,1);
% end
% R=mean(R2);

%%------------------------Prediciton------------------%%
nt=5;
for i=1:N
    yCDM(i,:)=BCDM(:,i)'*trainX{i};
end
for i=1:N
    rCDM(i,:)=BCDM(:,i)'*trainX{i}-trainY{i};
end
clear rmse_test

rmse_test(:,2)=sqrt(diag(rCDM*rCDM')/nt);
figure;
boxplot(rmse_test,'labels',{'CDM'});

clear RMSE_test
for i=1:5
    RMSE_test(i,1)=sqrt(mean(rCDM(:,i).^2))'; 
end
RMSE_test=RMSE_test/N;
for i=1:5
    nMSE_test(i,1)=sum(rCDM(:,i).^2)/var(Y(:,20+i));
end
nMSE_test=sum(nMSE_test)/(5*N);
wR=zeros(1,4);
for i=1:5
    wR(1)=corr(Y(:,20+i),yCDM(:,i))*N+wR(1);
end
wR=wR./(5*N);


%%%%---------------Experiment on Sparse Sample-----------------%%
%%-------------------sparse sampling----------------%%
for i=1:N
    trainXs{i}=trainX{i}(:,1:3:size(trainX{i},2));
    trainYs{i}=trainY{i}(:,1:3:size(trainY{i},2));
end
%%--------------------model learning----------------%%

%CDM
options.alpha = 0;
[AICs, K_finals, B0_CDMs, B_CDMs, W_CDMs, nIter_CDMs, objhistory_CDMs] = selectK( trainYs', trainXs',options, 1, 10);
[B_CDMs, W_CDMs, nIter_CDMs, objhistory_CDMs] = CDM(trainYs', trainXs', K, [],options,[],[]);
%goodness of fit
BCDMs=B_CDMs*W_CDMs;
dBCDMs=B0-BCDMs;
sdBs(1)=sum(sum(dBCDMs))/N;
vdBs(1)=sqrt(sum(diag(dBCDMs*dBCDMs'))/(N*11));

nt=5;
for i=1:N
    yCDMs(i,:)=BCDMs(:,i)'*testX{i};
end
for i=1:N
    rCDMs(i,:)=BCDMs(:,i)'*testX{i}-testY{i};
end
rmse_tests(:,2)=sqrt(diag(rCDMs*rCDMs')/nt);
figure;
boxplot(rmse_tests,'labels',{'CDM'});


%%%----------------------Prediction-------------------%%%
clear RMSE_tests
for i=1:5
    RMSE_tests(i, 1)=sqrt(mean(rCDMs(:,i).^2))';
end
RMSE_test=RMSE_test/N;
for i=1:5
    nMSE_tests(i,1)=sum(rCDMs(:,i).^2)/var(Y(:,20+i));
end
nMSE_tests=sum(nMSE_tests)/(5*N);
wRs=zeros(1,4);
for i=1:5
    wRs(1)=corr(Y(:,20+i),yCDMs(:,i))*N+wRs(1);
end
wRs=wRs./(5*N);