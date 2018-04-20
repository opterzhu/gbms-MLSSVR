function [ alpha,b ] = NewTrainMLSSVR( X,Y,p,lambda,gamma )
% New training algorithm for MLS-SVR according to reference[1].
% 
% References  : [1] "An efficient gradient-based model selection algorithm
%                   for multi-output least-squares support vector regression machines",
%                   Pattern Recognition Letters, 2018, doi="10.1016/j.patrec.2018.01.023"
%
% author: Zhu Xinqi (zhuxq3594@gmail.com)
[L,m] = size(Y);
K = Kerfun('rbf',X, X, p, 0);
Omega = (m/lambda)*K + 1/gamma*eye(L);
T = ones(L+1);
T(end,end) = 0;
T(1:end-1,1:end-1) = Omega;
B = T;
T(1:end-1,1:end-1) = Omega+m*K;
G = T;
Binv = getinv(B);
Ginv = getinv(G);
[alpha,b] = newtrain(Binv,Ginv,Y);
end

function C = getinv(B)
H = B(1:end-1,1:end-1);
l             = size(H,1);
[R,~]         = chol(H);
one           = ones(l,1);
eta           = R\(R'\one);
oneoversumeta = 1/sum(eta);
Ri = R\eye(size(R));
Mi = Ri*Ri';
SMi = -oneoversumeta;
C = zeros(size(H)+[1 1]);
C(1:end-1,1:end-1) = Mi + eta*SMi*eta';
C(end,1:end-1) = -SMi*eta';
C(1:end-1,end) = -SMi*eta;
C(end,end) = SMi;
end

function [alpha,b] = newtrain(Binv,Ginv,Y)
m = size(Y,2);
rho = [Y; zeros(1,m)];
s = Binv*rho - (1/m)*(Binv-Ginv)*repmat(sum(rho,2),[1,m]);
b = s(end,:);
alpha = s(1:end-1,:);
end