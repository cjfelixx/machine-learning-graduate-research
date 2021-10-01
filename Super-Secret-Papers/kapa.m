%===============================================================================
% The Kernel Affine Projection Algorithm
% contact: cedric.richard@unice.fr
%
% C. Richard, J. C. M. Bermudez, and P. Honeine, "Online prediction of
% time series data with kernels,"
% IEEE Transactions on Signal Processing, vol. 57, no. 3, pp. 1058-1067,
% 2009.
%
%
%
% function [err,ndict]=kapa(v,d,order,mu,epsilon,tresh,ker,p1);
%
% inputs of the function
% v     : matrix of the inputs of the filter, each row corresponding
%         to the input values at a given time instant.
% d     : row vector of desired outputs.
%
% order : order of KAPA, that is, size of the short-time window
% mu    : step-size of KAPA
% eps   : regularization parameter (eps-KAPA)
%
% tresh : novelty threshold in [0,1]. If the novelty coefficient (coherence)
%         of an input v(i) is larger than tresh, it is inserted into the
%         dictionary. A consequence is the increase of the order of the filter.
%         
% ker   : kernel, e.g., 'poly', 'rbf', ...
% p1    : parameter of the kernel, e.g., std of the rbf kernel
%
% output of the function
% err   : a priori output estimation error
% ndict : size of the dictionary
%===============================================================================


function [err,ndict]=kapa(v,d,order,mu,epsilon,tresh,ker,p1);

%====================================================================
% Initialization
%
% dict   : dictionary
% modict : modulus of the elements v(i) of the dictionary
%====================================================================
dic=v(1,:);
tdict=1;
k=kernel(ker,dic,dic,p1);
modict=sqrt(k);

% Initialization of the weigth vector
alpha=d(1)/k;

% Filtering process
i=1;
while i<size(v,1)
    i=i+1;
    
    % Mapping of input v(i) into the feature space
    b=v(i,:);
    kv=kernel(ker,dic,b,p1);
    k=kernel(ker,b,b,p1);
    C=kv./(sqrt(k)*modict);
    
    % If v(i) is sufficiently novel, that is, the coherence of 
    % the dictionary remains smaller than the threshold tresh, v(i)
    % is inserted into the dictionary and the size of the weight
    % vector alpha is increased by 1.
    if (max(abs(C))<tresh)  
        dic=[dic;b];
        modict=[modict;sqrt(k)];
        tdict=[tdict;i];
        alpha=[alpha;0];
    end
    
    % Iteration of APA in the feature space
    H=kernel(ker,dic,v(i:-1:max([i-order+1 1]),:),p1)';
    D=d(i:-1:max([i-order+1 1]))';
    E=D-H*alpha;
    err(i)=E(1);
    alpha=alpha+mu*H'*inv(H*H'+epsilon*eye(length(D)))*E;
end

tdict;
ndict=length(tdict);