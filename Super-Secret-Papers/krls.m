%===============================================================================
% The KRLS algorithm of Engel, Mannor and Meir.
% contact: cedric.richard@unice.fr
%
% Y. Engel, S. Mannor, and R. Meir, "Kernel recursive least squares,"
% IEEE Transactions on Signal Processing, vol. 52, no. 8, pp. 2275-2285, 2004.
%
% This method performs kernel-based adaptive filtering of order-recursive RLS
% type. See the above-mentionned reference for more details.
%
% function [e,o]=krls(v,d,tresh,ker,p1);
%
% inputs of the function
% v     : matrix of the inputs of the filter, each row corresponding
%         to the input values at a given time instant.
% d     : row vector of desired outputs.
% tresh : novelty threshold in [0, 1]. If the novelty coefficient of an input v(n)
%         is larger than tresh, v(n) is included in the set of representers. A
%         consequence is the increase of the order of the filter.
% ker   : kernel, e.g., 'poly', 'rbf', ...
% p1    : parameter of the kernel, e.g., std of the rbf kernel
%
% output of the function
% e     : a priori output estimation error
% o     : order of the filter at the end of the filtering process
%===============================================================================


function [e,nb]=krls(v,d,tresh,ker,p1);

%====================================================================
% Initialization
%
% Rep    : set of representers
% Nbrep  : number of representers
% Indrep : time indexes of representers
%====================================================================
Rep=v(1,:);
Nbrep=1;
Indrep=1;
k=kernel(ker,Rep,Rep,p1);
Kinv=1/k;
%alpha=d(1)/k;
alpha=0;

modRep=sqrt(k);

P=1;
index=1;

% Loop for filtering
for i=2:size(v,1)
    b=v(i,:);
    kv=kernel(ker,Rep(1:Nbrep,:),b,p1);
    k=kernel(ker,b,b,p1);
        
    % Computation of the novelty parameter, denoted below by delta, 
    % of the current input v(i)
    a=Kinv*kv;
    delta=k-kv'*a;
    e(i)=d(i)-kv'*alpha;
    
    % Case 1: v(i) is sufficiently novel to be included in the set of
    % representers. The order of the filter increases and its tap weights
    % are updated.
    
    if(delta>tresh)
        Rep=[Rep;b];
        Indrep=[Indrep;i];

        Kinv=[delta*Kinv+a*a', -a;-a',1]/delta;
        z=zeros(Nbrep,1);
        P=[P,z;z',1];
        alpha=[alpha-a*e(i)/delta;e(i)/delta];
        modRep=[modRep;sqrt(k)];
        Nbrep=Nbrep+1;
    
    % Case 2: v(i) is unsufficiently novel to be included in the set of
    % represented. The order of the filter is not modified. Only the tap
    % weights are updated.
    else
        pa=P*a;
        q=pa/(1+a'*pa);
        P=P-q*pa';
        alpha=alpha+Kinv*q*e(i);
    end
end
order=Nbrep;
Indrep;
sv=Rep;
nb=length(Indrep);