%Simulation of synchronous Federated Learning
tic
total = 100; %number of repetitions of simulation
iter = 10000;
amse = zeros(iter,1);
for epoch = 1:total
dn =zeros(1,3000); %nonlinear time series
dn(1) = .1;
dn(2) = .1;
for ii = 3:3000
dn(ii) = (0.8-0.5*exp(-dn(ii-1)^2))*dn(ii-1) - (0.3+0.9*exp(-dn(ii-1)^2))*dn(ii-2) + .1*sin(dn(ii-1)*pi)+0.1*randn(1,1); %random Gaussian noise with std = 0.1
end
X = [dn(1:2998);dn(2:2999)]; %input data
D = transpose(dn(3:3000)); %output data

%Here we perform synchronous updating
el= 10; %number of edge processors
dcount = zeros(1,el);
agg = 500; %number iterations when cloud contacts edge processors
mu = .8;
samples = 100; %number of random fourier features
sigma = sqrt(1/(3.73*2)); %Gaussian widths
w = 1/sigma * randn(2,samples); %random fourier weight vector
theta = rand(1,samples); %random phase vector
Z = sqrt(2/samples)*cos(transpose(X)*w + pi*ones(2998,1)*theta); %inputs in RFF space

hc = zeros(1,samples); % learning parameter for cloud
hfl = zeros (el,samples); % learning parameter for edge
mse= zeros(iter,1);
msed = zeros(iter,el);
mse(1:agg-1)= var(D);

for jj= 1:iter
v = ceil(rand(1,1)*2998); %randomly pick data
edge = ceil(rand(1,1)*el); %randomly pick  edge processor  where data goes to
dcount(edge)=dcount(edge)+1; %update number of updates for edge processor
x = X(:,v);
error = D(v) -  Z(v,:)*transpose(hfl(edge,:)); %compute error
hfl(edge,:) = hfl(edge,:) + mu *Z(v,:)*error; % update edge learner using LMS

if mod(jj,agg) ==0 %test to see if central processor should communicate with all edge processors

hc = mean(hfl); %central processor is mean of all edge processors

mse(jj) = norm(D(2499:2998) - Z(2499:2998,:)*transpose(hc))^2/500; %MSE on last 500 data points
hfl = ones(el,1)*hc; %reset all edge processor weights
elseif jj>1, mse(jj) = mse(jj-1);
end
end
amse = amse + mse(1:iter,1)/total;
end
toc






