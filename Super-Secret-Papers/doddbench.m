%===============================================================================
% doddbench.m
% contact: cedric.richard@unice.fr
%
% Benchmark signal introduced in:
%
% Dodd, T.J., Kadirkamanathan, V. and Harrison, R.F., "Function
% estimation in Hilbert space using sequential projections,"
% Proc. of the IFAC Conf. on Intelligent Control Systems and Signal
% Processing, 113-118, 2003.
%
% Benchmark signal used in:
% C. Richard, J. C. M. Bermudez, and P. Honeine, "Online prediction of
% time series data with kernels,"
% IEEE Transactions on Signal Processing, vol. 57, no. 3, pp. 1058-1067, 2009.
%
% function [v,d,dref]=benchA(Ndata)
%
% input of the function
% Ndata : signal length
%
% outputs of the function
% v     : input sequence (2-dimensional sequence [v(1,:);v(2,:)])
% d     : noisy desired output (1-dimensional sequence)
% dref  : noise-free desired output
%
%===============================================================================



function [v,d,dref]=doddbench(Ndata)

clear v d dref
dref(1:2)=[0.1 0.1];
for t=3:Ndata+2,
    dref(t)=(0.8-0.5*exp(-dref(t-1)^2))*dref(t-1)-(0.3+0.9*exp(-dref(t-1)^2))*dref(t-2)+0.1*sin(pi*dref(t-1));
end
d=dref+0.1*randn(1,Ndata+2);
v=[d(1:Ndata);d(2:Ndata+1)]';
d(1:2)=[];
dref(1:2)=[];
