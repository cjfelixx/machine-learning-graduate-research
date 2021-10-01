%===============================================================================
% The NORMA algorithm
% contact: cedric.richard@unice.fr
%
% J. Kivinen, A. J. Smola, and R. C. Williamson, "Online learning with kernels,?
% IEEE Transactions on Signal Processing, vol. 52, no. 8, pp. 2165?2176,
% 2004.
%
% function e=norma(v,d,order,lambda,eta,ker,p1);
%
% inputs of the function
% v     : matrix of the inputs of the filter, each row corresponding
%         to the input values at a given time instant.
% d     : row vector of desired outputs.
% order : order of the filter
% lambda: regularization parameter
% eta   : truncation parameter
% ker   : kernel, e.g., 'poly', 'gauss', ...
% p1    : parameter of the kernel, e.g., std of the rbf kernel
%
% output of the function
% e     : a priori output estimation error
%===============================================================================


function err = norma(v,d,order,lambda,eta,ker,p1)

% Initializations
i=1;
it=1;
alpha=0;

% Loop for filtering
while i<size(v,1)
    i=i+1;
    
    s=kernel(ker,v(it,:),v(i,:),p1);
    dest(i)=alpha'*s;
    err(i)=d(i)-dest(i);

    alpha=[eta*err(i);(1-lambda*eta*i^(-0.5))*alpha(1:min([order-1 length(alpha)]))];
    
    it=(i:-1:max([1 i-order+1]));
        
end
