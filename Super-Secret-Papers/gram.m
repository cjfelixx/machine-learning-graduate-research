%================================================================
% Gram : Gram matrix calculation for kernel-based methods
% contact: cedric.richard@unice.fr
%
% function K = gram(ker,U,V,varargin,['+']);
%
% inputs of the function
% ker       : kernel type; see below
% U,V       : kernel arguments (row-wise data)
% varargin  : parameters of the kernel, e.g., the degree of polynomial kernel
% '+'       : normalization k(xi,xj)/sqrt(k(xi,xi)*k(xj,xj)) 'optional'
%
% output of the function
% K   : Gram matrix
%
% Remark
% This function uses efficient matrix calculation to compute Gram matrices.
% It must be preferred to kernel.m that is dedicated to individual kernel
% calculations.
%
% ker:  'linear'    -
%       'poly'      - k(xi,xj) = (gamma*xi'xj + 1)^d
%       'gauss'     - k(xi,xj) = exp(-|xi - xj|^2/(2*sigma^2)))
%       'laplace'   - k(xi,xj) = exp(-|xi - xj|/(2*sigma^2))      
%
% varargin:
%       'linear':   kerpar=gamma [default=1]
%       'poly':     kerpar(1)=d; kerpar(2)=gamma [default=1]
%       'gauss':    kerpar=sigma
%       'laplace':  kerpar=sigma
%
%==========================================================================

function [K]=gram(ker,X1,X2,kerpar,opt)

if (nargin ~= 4) & (nargin ~= 5)
   help svkernel
   return
end

[n1,p1]=size(X1);
[n2,p2]=size(X2);

ker = lower(ker);

if strcmp(ker,'linear')
    % linear kernel
    if (length(kerpar) == 1)
        gamma = kerpar(1);
    else
        gamma = 1;
    end
    K = gamma*X1*X2';
    if nargin == 5
        if strcmp(opt,'+')
            k1 = zeros(n1,1);
            k2 = zeros(1,n2);        
            for i=1:n1
                k1(i) = gamma*X1(i,:)*X1(i,:)';
            end
            for i=1:n2
                k2(i) = gamma*X2(i,:)*X2(i,:)';
            end
            K1 = sqrt(k1*ones(1,n2));
            K2 = sqrt(ones(n1,1)*k2);
            K = K./(K1.*K2);
        end
    end
elseif strcmp(ker,'poly')
    % polynomial kernel
    if (length(kerpar) == 1)
        kerpar(2) = 1; % gamma  
    end
    K = (kerpar(2)*X1*X2' + 1).^kerpar(1);
    if nargin == 5
        if strcmp(opt,'+')
            k1 = zeros(n1,1);
            k2 = zeros(1,n2);        
            for i=1:n1
                k1(i) = (kerpar(2)*X1(i,:)*X1(i,:)' + 1).^kerpar(1);
            end
            for i=1:n2
                k2(i) = (kerpar(2)*X2(i,:)*X2(i,:)' + 1).^kerpar(1);
            end
            K1 = sqrt(k1*ones(1,n2));
            K2 = sqrt(ones(n1,1)*k2);
            K = K./(K1.*K2);
        end
    end
elseif (strcmp(ker,'gauss') | strcmp(ker,'laplace'))
	% Gaussian kernel and Laplacian kernel 
	if (p1 == 1)
        MX1 = (X1'.*X1')';
        MX2 = (X2'.*X2');
    else            
        MX1 = sum(X1.*X1, 2);
        MX2 = sum(X2.*X2, 2)';
    end;   
    K = MX1*ones(1,n2) + ones(n1,1)*MX2 - 2*X1*X2';   
    roundoff_error = find(K < 0);
    K(roundoff_error) = 0;
    if strcmp(ker,'gauss')
        % Gaussian kernel      
        K = exp(-K/(2*kerpar(1)^2));
    else
		% Laplacian kernel      
        K = exp(-sqrt(K)/(2*kerpar(1)^2));      
    end
else
	error('Unknown kernel function')   
end