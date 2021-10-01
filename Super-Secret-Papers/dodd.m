
%===============================================================================
% The kernel-based online algorithm for time series prediction by T.J. Dodd et al.
% contact: cedric.richard@unice.fr
%
% Dodd, T.J., Kadirkamanathan, V. and Harrison, R.F., "Function
% estimation in Hilbert space using sequential projections,"
% Proc. of the IFAC Conf. on Intelligent Control Systems and Signal Processing, 113-118, 2003.
%
% function [e,o]=dodd(v,d,eta,tresh,ker,p1)
%
% inputs of the function
% v     : matrix of the inputs of the filter, each row corresponding
%         to the input values at a given time instant.
% d     : row vector of desired outputs.
% eta   : step-size
% tresh : novelty threshold in [0, 1]. If the novelty coefficient (error
%         norm)vis larger than tresh, v(n) is included in the set of representers. A
%         consequence is the increase of the order of the filter.
% ker   : kernel, e.g., 'poly', 'rbf', ...
% p1    : parameter of the kernel, e.g., std of the rbf kernel
%
% output of the function
% e     : a priori output estimation error
% o     : order of the filter at the end of the filtering process
%
% Last Update : 29/09/2012 (C. Richard)
%==========================================================================

function [e,p]=dodd(x,y,eta,refrate,ker,kerpar)

% Initializations
N = length(y);
v = zeros(N,1);
it = 1;
alpha = 0;
p = 1;
K = gram(ker,x(it,:),x(it,:),kerpar);
v(it) = sqrt(diag(K));
K = K./(v(it)*v(it)');

Q = inv(K);

% Loop for filtering
for t=1:N
    
    v(t) = sqrt(gram(ker,x(t,:),x(t,:),kerpar));
    s = gram(ker,x(it,:),x(t,:),kerpar);
    yest(t) = alpha'*(s./v(it));
    e(t) = y(t) - yest(t);
    alpha(p+1,1) = eta*e(t);
       
    rho = s./(v(it)*v(t));
    c = [K rho]*alpha;
    beta = K\c; 
       
    if eta >= 2/v(t)
        warning('eta does not satisfy eta < 2/sqrt(kappa(x(t+1),x(t+1))')
    end    
    
    % Sparsification (incremental step)
    if alpha'*[K rho;rho' 1]*alpha - alpha'*[K rho]'*beta > refrate
        
        % If the error norm is larger than refrate, then increase the model
        % order
        A = Q*rho;
        B = A';
        k = 1/(1 - rho'*A);
        Q = [Q+A*B*k -k*A; -k*B k];
        K = [K rho; rho' 1];
        it = [it t];
        p = p + 1;      
    else
        % Else do not increase the order of the model by projecting the
        % extended model onto the space spanned by the kernel basis
        % functions used by the previous (reduced-order) model.
        alpha = alpha(1:p) + alpha(p+1)*Q*rho;
    end 
end
