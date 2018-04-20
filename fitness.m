function [ Q,grad ] = fitness( x,trnX,trnY )
% Calculate the PRESS and its partial dirivatibes with respect to the
% hyper-parameters
% 
% References  : [1] "An efficient gradient-based model selection algorithm
%                   for multi-output least-squares support vector regression machines",
%                   Pattern Recognition Letters, 2018, doi="10.1016/j.patrec.2018.01.023"
%               [2] Cawley, G.C., Talbot, N.L.Preventing over-fitting during model 
%                   selection via bayesian regularisation of the hyper-parameters. Journal of Machine
%                   Learning Research, 2007, 8, 841¨C861
%
% author: Zhu Xinqi (zhuxq3594@gmail.com)

if length(x)==3
    p = 2^x(1);
elseif length(x)>3
    p = 2.^x(1:size(trnX,2));
end
lambda = 2^x(end-1);
gama = 2^x(end);

% Construct the matrix B, D and G
% Compute the kernal matrix K and its derivative dKdp
[l, m] = size(trnY); 
[K,dKdp] = Kerfun('rbf', trnX, trnX, p, 0); 
B = ones(l+1,l+1);
B(end,end) = 0;
M1 = K*(m/lambda) + (1/gama)*eye(l);
B(1:end-1,1:end-1) = M1;
D = zeros(l+1);
D(1:end-1,1:end-1) = K;
G = B + m*D;
M2 = G(1:end-1,1:end-1);
% dKdp = -K.*Xij;

[Bikk,dBidp,dBidl,dBidg,Bi] = caldiagonal(M1,m/lambda,-m/lambda^2,-1/gama^2,dKdp,K);
dBidpkk = zeros(l,size(dBidp,1));
for i=1:size(dBidp,1)
    tmp = squeeze(dBidp(i,:,:));
    dBidpkk(:,i) = diag(tmp);
end
% dBidpkk = diag(dBidp);
dBidlkk = diag(dBidl);
dBidgkk = diag(dBidg);
[Gikk,dGidp,dGidl,dGidg,Gi] = caldiagonal(M2,m/lambda+m,-m/lambda^2,-1/gama^2,dKdp,K);
dGidpkk = zeros(l,size(dGidp,1));
for i=1:size(dGidp,1)
    tmp = squeeze(dGidp(i,:,:));
    dGidpkk(:,i) = diag(tmp);
end
% dGidpkk = diag(dGidp);
dGidlkk = diag(dGidl);
dGidgkk = diag(dGidg);

meanY = repmat(mean(trnY,2),[1,m]);
dalfdp = zeros(size(dBidp,1),l,m);
for i=1:size(dBidp,1)
    dBidpi = squeeze(dBidp(i,:,:));
    dGidpi = squeeze(dGidp(i,:,:));
    dalfdp(i,:,:) = dBidpi*(trnY - meanY) + dGidpi*meanY;
end
dalfdl = dBidl*(trnY - meanY) + dGidl*meanY;
dalfdg = dBidg*(trnY - meanY) + dGidg*meanY;
Bikk = 1./Bikk;
Gikk = 1./Gikk;
BGikk = repmat(Bikk+(Gikk-Bikk)/m,[1,m]);

% Compute the solution vector alpha using B and G.
Y = [trnY; zeros(1,m)];
s = Bi*Y - (1/m)*((Bi-Gi)*repmat(sum(Y,2),[1,m]));
alpha = s(1:end-1,:);

% Compute the virtual LOO error r.
r = alpha.*repmat(Bikk,[1,m]) + repmat(Gikk-Bikk,[1,m]).*repmat(sum(alpha,2),[1,m])/m;

% Compute the derivative vector of r with respect to p, lambda and gama.
meanalf = mean(alpha,2);
tmp1   = (alpha - repmat(meanalf,[1,m])).*repmat(Bikk.^2,[1,m]);
tmp2   = repmat(meanalf.*Gikk.^2,[1,m]);
tmp3   = repmat(Gikk-Bikk,[1,m])/m;

sumdalfdl = sum(dalfdl,2);
sumhdalfdl = repmat(sumdalfdl,[1,m]) - dalfdl;
sumdalfdg = sum(dalfdg,2);
sumhdalfdg = repmat(sumdalfdg,[1,m]) - dalfdg;
drdp = zeros(size(dBidpkk,2),l,m);
for i=1:size(drdp,1)
    dalfdpi = squeeze(dalfdp(i,:,:));
    sumdalfdpi = sum(dalfdpi,2);
    sumhdalfdpi = repmat(sumdalfdpi,[1,m]) - dalfdpi;
    dBidpikk = squeeze(dBidpkk(:,i));
    dGidpikk = squeeze(dGidpkk(:,i));
    drdp(i,:,:) = BGikk.*dalfdpi - tmp1.*repmat(dBidpikk,[1,m]) - tmp2.*repmat(dGidpikk,[1,m]) + tmp3.*sumhdalfdpi;
end
drdl = BGikk.*dalfdl - tmp1.*repmat(dBidlkk,[1,m]) - tmp2.*repmat(dGidlkk,[1,m]) + tmp3.*sumhdalfdl;
drdg = BGikk.*dalfdg - tmp1.*repmat(dBidgkk,[1,m]) - tmp2.*repmat(dGidgkk,[1,m]) + tmp3.*sumhdalfdg;

% Compute the LOO criterion
Q = sum(sum(sum((r).^2),2))/2;
dQdr = r;
% Compute the derivative of LOO criterion Q with respect to p, lambda and
% gama
dQdp = zeros(size(drdp,1),1);
for i=1:size(drdp,1)
    drdpi = squeeze(drdp(i,:,:));
    dQdp(i) = sum(sum(dQdr.*drdpi))*p(i)*log(2);
end
dQdl = sum(sum(dQdr.*drdl))*lambda*log(2);
dQdg = sum(sum(dQdr.*drdg))*gama*log(2);

grad = [dQdp; dQdl; dQdg];
end

function [s,dCidp,dCidl,dCidg,C] = caldiagonal(H,para1,para2,para3,dK,K)
l             = size(H,1);
[R,p]         = chol(H);
if p>0
    [R,~]     = chol(H+eye(l)*1e-8);
end
one           = ones(l,1);
eta           = R\(R'\one);
oneoversumeta = 1/sum(eta);
Ri            = R\eye(size(R));
s             = sum(Ri.^2,2) - oneoversumeta*eta.^2;   %diagnal element of inverse C
A             = Ri*Ri' - (oneoversumeta*eta)*eta';
dCidp = zeros(size(dK,1),l,l);
for i=1:size(dK,1)
    deekay = squeeze(dK(i,:,:));
    dCidp(i,:,:) = -((para1*A*deekay)*A);
end
% dCidp         = -((para1*A*deekay)*A);
dCidl         = -((para2*A*K)*A);
dCidg         = -(para3*A*A);

Mi = Ri*Ri';
SMi = -oneoversumeta;
C = zeros(size(H)+[1 1]);
C(1:end-1,1:end-1) = Mi + eta*SMi*eta';
C(end,1:end-1) = -SMi*eta';
C(1:end-1,end) = -SMi*eta;
C(end,end) = SMi;
end