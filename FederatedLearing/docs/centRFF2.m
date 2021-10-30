tic

total = 5; %number of repetitions of  simulation
iter = 10000;
amse = zeros(iter,1);
for epoch = 1:total
dn =zeros(1,3000); %nonlinear time series
dn(1) = .1;
dn(2) = .1;
for ii = 3:3000
dn(ii) = (0.8-0.5*exp(-dn(ii-1)^2))*dn(ii-1) - (0.3+0.9*exp(-dn(ii-1)^2))*dn(ii-2) + .1*sin(dn(ii-1)*pi)+0.1*randn(1,1);
end
X = [dn(1:2998);dn(2:2999)]; %input data
D = transpose(dn(3:3000)); %output data

mu = 0.2; %step size
samples = 100; %number of RFF
sigma = sqrt(1/(3.73*2)); %Gaussian widths
w = 1/sigma * randn(2,samples);
theta = rand(1,samples);
Z = sqrt(2/samples)*cos(transpose(X)*w + pi*ones(2998,1)*theta); %inputs in RFF space

%Centralized RFF kernel learning
h= zeros(1,samples);
mse=zeros(iter,1);
error= zeros(iter,1);

for jj= 1:iter
v = ceil(rand(1,1)*2998); %randomly pick data
x = X(:,v);
%z = sqrt(2/samples)*cos(transpose(x)*w + pi*theta);
error(jj) = D(v) - Z(v,:)*transpose(h);
h = h + mu *Z(v,:)*error(jj); %LMS update

mse(jj) = norm(D(2499:2998) - Z(2499:2998,:)*transpose(h))^2/500; %MSE on last 500 data points
end
amse = amse + mse(1:iter,1)/total;
end
toc

semilogy(mse)
