
%================================================================
% KERNEL: kernel for kernel-based methods
% contact: cedric.richard@unice.fr
%
% function k=kernel(ker,u,v,varargin);
%
% inputs of the function
% ker      : kernel type; see below
% u,v      : kernel arguments (row-wise data)
% varargin : parameters of the kernel, e.g., the degree of polynomial kernel
%
% output of the function
% k   : kernel value
%
% Remark
% The arguments u and v can be matrices. 
% Then we have: k(i,j)=ker(u(i,:),v(j,:))
%
%  Values for ker: 'linear'     -
%                  'poly'       - p1 is degree of polynomial
%                  'gaussian'   - p1 is width of rbfs (.../(2*sigma^2))
%                  'sigmoid'    - p1 is scale, p2 is offset
%                  'laplace'    - p1 is width of erbfs (.../(2*sigma^2))
%===============================================================================

function k = kernel(ker,u,v,varargin)

p1=[];
p2=[];

% check correct number of arguments
if (nargin < 1) 
     help kernel.m
 elseif (nargin==4),
     p1=varargin{1};
 elseif (nargin==5),
     p1=varargin{1};
     p2=varargin{2};
 end

% compute the kernel values
switch lower(ker)
    case 'linear'
        k = real(u*v');
    case 'mono'
        if isempty(p1), p1=4; end
        k = (u*v').^p1;
        % normalization to have ker(u,u)=1 for all u (optional)
        % k = k./((sum(u.^2,2))*(sum(v.^2,2)').^(p1/2)); 
    case 'poly'
        if isempty(p1), p1=4; end
        k = (u*v'+1).^p1;
        % normalization to have ker(u,u)=1 for all u (optional)
        % k = k./((sum(u.^2,2)+1)*(sum(v.^2,2)'+1).^(p1/2)); 
    case 'gauss'
        if isempty(p1), p1=2.25; end
        k = exp(-(dist(u,v').^2)./(2*p1^2));
    case 'laplace'
        if isempty(p1), p1=1; end
        k = exp(-(dist(u,v'))./(2*p1^2));
    case 'sigmoid'
        if isempty(p1), p1=1; end
        if isempty(p2), p2=1; end
        k = tanh(p1*u*v'/size(u,2)+p2);
 end
