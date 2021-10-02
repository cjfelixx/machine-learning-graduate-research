%===============================================================================
% execB.m
% contact: cedric.richard@unice.fr
%
% This function reproduces the results described in Experiment B of:
%
% C. Richard, J. C. M. Bermudez, and P. Honeine, "Online prediction of
% time series data with kernels,"
% IEEE Transactions on Signal Processing, vol. 57, no. 3, pp. 1058-1067, 2009.
%
%===============================================================================


% Initializations
clear
Ndata=3000; % number of input data
Nexp=20;   % number of experiments (200 experiments are considered in the paper)

% kernel parameters
ker='laplace';
p1=0.35;

% Set performance arrays to zero
mse_dodd=zeros(1,Ndata);
mse_krls=zeros(1,Ndata);
mse_knlms=zeros(1,Ndata);
mse_kapa2=zeros(1,Ndata);
mse_kapa3=zeros(1,Ndata);
mse_norma=zeros(1,Ndata);


% Loop on experiments
for i=1:Nexp,
    i
    
    % Generate the data
    [v,d,dref]=richardbench(Ndata);
    
    % Dodd et al. algorithm
    eta=0.03;
    seuil=5e-3;
    [e,nb]=dodd(v,d,eta,seuil,ker,p1);
    mse_dodd=mse_dodd+(dref-d+e).^2/Nexp;
   
    % KRLS algorithm
    tresh=0.7;
    [e_krls,nb]=krls(v,d,tresh,'laplace',p1);
    mse_krls=mse_krls+(dref-d+e_krls).^2/Nexp;
    
    % KNLMS algorithm (here KAPA with p=1)
    seuil=0.3;
    p=1;
    eta=0.01;
    epsilon=9e-4;
    % [e_knlms,nb]=kapa(v,d,p,eta,epsilon,seuil,'laplace',p1);
    % mse_knlms=mse_knlms+(dref-d+e_knlms).^2/Nexp;
    
    % KAPA algorithm (with p=2)
    seuil=0.3;
    p=2;
    eta=0.009;
    epsilon=7e-2;
    [e_kapa2,nb]=kapa(v,d,p,eta,epsilon,seuil,'laplace',p1);
    mse_kapa2=mse_kapa2+(dref-d+e_kapa2).^2/Nexp;
    
    % KAPA algorithm (with p=3)
    seuil=0.3;
    p=3;
    eta=0.01;
    epsilon=7e-2;
    [e_kapa3,nb]=kapa(v,d,p,eta,epsilon,seuil,'laplace',p1);
    mse_kapa3=mse_kapa3+(dref-d+e_kapa3).^2/Nexp;
    
    % NORMA algorithm
    order=35;
    lambda=0.09;
    eta=0.09;
    e_norma=norma(v,d,order,lambda,eta,'laplace',p1);
    mse_norma=mse_norma+(dref-d+e_norma).^2/Nexp;
end


% Mse smoothing by moving average for vizualization
mse_dodd_smooth=zeros(size(mse_dodd));
mse_krls_smooth=zeros(size(mse_krls));
mse_knlms_smooth=zeros(size(mse_knlms));
mse_kapa2_smooth=zeros(size(mse_kapa2));
mse_kapa3_smooth=zeros(size(mse_kapa3));
mse_norma_smooth=zeros(size(mse_norma));

for k=1:length(mse_dodd)-19,
     mse_dodd_smooth(k)=mean(mse_dodd(k:min([k+19,length(mse_dodd)])));
     mse_krls_smooth(k)=mean(mse_krls(k:min([k+19,length(mse_krls)])));
     mse_knlms_smooth(k)=mean(mse_knlms(k:min([k+19,length(mse_knlms)])));
     mse_kapa2_smooth(k)=mean(mse_kapa2(k:min([k+19,length(mse_kapa2)])));
     mse_kapa3_smooth(k)=mean(mse_kapa3(k:min([k+19,length(mse_kapa3)])));
     mse_norma_smooth(k)=mean(mse_norma(k:min([k+19,length(mse_norma)])));     
end

% Figure: comparison of the mse
figure(1)
clf
semilogy(mse_dodd_smooth,'r')
hold on
semilogy(mse_krls_smooth,'g')
semilogy(mse_knlms_smooth,'b')
semilogy(mse_kapa2_smooth,'k')
semilogy(mse_kapa3_smooth,'m')
semilogy(mse_norma_smooth,'c')
legend('dodd','krls','knlms','kapa 2','kapa 3','norma')
title('Learning curves')
xlabel('iteration')
ylabel('mean-square-error')


