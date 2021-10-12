%===============================================================================
% execA.m
% contact: cedric.richard@unice.fr
%
% This function reproduces the results described in Experiment A of:
%
% C. Richard, J. C. M. Bermudez, and P. Honeine, "Online prediction of
% time series data with kernels,"
% IEEE Transactions on Signal Processing, vol. 57, no. 3, pp. 1058-1067, 2009.
%
%===============================================================================


% Initializations
clear
Ndata=3000; % number of input data
Nexp=1;    % number of experiments (200 experiments are considered in the paper)

% kernel parameters
ker = 'gauss';
p1=1/sqrt(2*3.73);

% Set performance arrays to zero
mse_dodd=zeros(1,Ndata);
mse_krls=zeros(1,Ndata);
mse_kapa=zeros(1,Ndata);
mse_norma=zeros(1,Ndata);


% Loop on experiments
for i=1:Nexp,
    i
    
    % Generate the data
    [v,d,dref]=doddbench(Ndata);
    
    % Dodd et al. algorithm
%     eta=0.1;
%     seuil=1*1e-3;
%     [e,nb]=dodd(v,d,eta,seuil,ker,p1);
%     mse_dodd=mse_dodd+(dref-d+e).^2/Nexp;
%    
%     % KRLS algorithm
%     tresh=0.6;
%     [e_krls,nb]=krls(v,d,tresh,ker,p1);
%     mse_krls=mse_krls+(dref-d+e_krls).^2/Nexp;
    
    % KAPA algorithm (here KNLMS since p=1)
    seuil=0.5;
    p=1;
    eta=0.09;
    epsilon=0.03;
    [e_kapa,nb]=kapa(v,d,p,eta,epsilon,seuil,ker,p1);
    mse_kapa=mse_kapa+(dref-d+e_kapa).^2/Nexp;
    e_kapa
    % NORMA algorithm
%     order=38;
%     lambda=0.98;
%     eta=1;
%     e_norma=norma(v,d,order,lambda,eta,ker,p1);
%     mse_norma=mse_norma+(dref-d+e_norma).^2/Nexp;
end


% Mse smoothing by moving average for vizualization
mse_dodd_smooth=zeros(size(mse_dodd));
mse_krls_smooth=zeros(size(mse_krls));
mse_kapa_smooth=zeros(size(mse_kapa));
mse_norma_smooth=zeros(size(mse_norma));

for k=1:length(mse_dodd)-19,
     mse_dodd_smooth(k)=mean(mse_dodd(k:min([k+19,length(mse_dodd)])));
     mse_krls_smooth(k)=mean(mse_krls(k:min([k+19,length(mse_krls)])));
     mse_kapa_smooth(k)=mean(mse_kapa(k:min([k+19,length(mse_kapa)])));
     mse_norma_smooth(k)=mean(mse_norma(k:min([k+19,length(mse_norma)])));     
end

% Figure: comparison of the mse
figure(1)
clf
semilogy(mse_dodd_smooth,'r')
hold on
semilogy(mse_krls_smooth,'g')
semilogy(mse_kapa,'b')
semilogy(mse_norma_smooth,'c')
legend('dodd','krls','knlms','norma')
title('Learning curves')
xlabel('iteration')
ylabel('mean-square-error')