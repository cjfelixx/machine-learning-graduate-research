%===============================================================================
% richardbench.m
% contact: cedric.richard@unice.fr
%
% Benchmark signal used in:
% C. Richard, J. C. M. Bermudez, and P. Honeine, "Online prediction of
% time series data with kernels,"
% IEEE Transactions on Signal Processing, vol. 57, no. 3, pp. 1058-1067, 2009.
%
% function [v,d,dref]=richardbench(Ndata)
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



function [u,d,dref]=richardbench(Ndata)

clear u v d dref
v(1)=0.5;
u=0.25*randn(Ndata+1,1);
for t=2:Ndata+1,
    v(t)=1.1*exp(-abs(v(t-1)))+u(t);
    dref(t)=v(t)^2;
end
d=dref+randn(1,Ndata+1);
d(1)=[];
dref(1)=[];
u(1)=[];
 