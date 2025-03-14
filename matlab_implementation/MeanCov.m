function [mean,cov] = MeanCov(GP,xtest)

% Kxx = zeros(1,size(xtrain,1));
% Kxz = zeros(1,size(xtest,1));
% Kzz = zeros(1,size(xtest,1));

% for ii = 1:size(xtrain,1)
%     Kxx(:,ii) = ((sf)*exp(-(0.5*(xtrain/l-xtrain(ii,:)/l).^2)));
% end

% for ii = 1:size(xtrain,1)
%     Kxz(:,ii) = ((sf)*exp(-(0.5*(xtest/l-xtrain(ii,:)/l).^2)));
% end
%
% for ii = 1:size(xtest,1)
%     Kzz(:,ii) = ((sf)*exp(-(0.5*(xtest/l-xtest(ii,:)/l).^2)));
% end
%
%     Kxx = (sf)*exp(-(0.5*sq_dist(xtrain/l,xtrain/l))); %% upper left corner of K
%     Kzz = (sf)*exp(-(0.5*sq_dist(xtest/l,xtest/l))); %% lower right corner of K
%     Kxz = (sf)*exp(-(0.5*sq_dist(xtrain/l,xtest/l))); %% upper right corner of K


Kxz = Kernel(GP.xtrain,xtest,GP.sf,GP.l);
Kzz = Kernel(xtest,xtest,GP.sf,GP.l);
Kzx = Kxz'; %% lower left corner of K

% GP.Kxx = nearestSPD(GP.Kxx);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R = chol(GP.Kxx+diag((GP.sn).^2));

%     if size(hypmean,1)==2
%     mmA1 = hypmean(1)*xz+hypmean(2);
%     else
% %     mmA1 = hypmean(1)*xz.^2+hypmean(2)*xz+hypmean(3);
%     mmA1 = hypmean(1)*xz.^5+hypmean(2)*xz.^4+hypmean(3)*xz.^3+hypmean(4)*xz.^2+hypmean(5)*xz+hypmean(6);
%     end
ad=R'\(GP.ytrain);
bd=R'\(Kxz);
%xz = xtest;
%     if size(hypmean,1)==2
%     mmA2 = hypmean(1)*xz+hypmean(2);
%     else
% %     mmA2 = hypmean(1)*xz.^2+hypmean(2)*xz+hypmean(3);
%     mmA2 = hypmean(1)*xz.^5+hypmean(2)*xz.^4+hypmean(3)*xz.^3+hypmean(4)*xz.^2+hypmean(5)*xz+hypmean(6);
%     end
mean = bd'*ad;
cov = Kzz-bd'*bd;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mean = Kxz'*(GP.Kxx+(GP.sn).^2.*eye(size(GP.xtrain,1)))^(-1)*GP.ytrain;
% cov = Kzz-Kxz'*(GP.Kxx+(GP.sn).^2.*eye(size(GP.xtrain,1)))^(-1)*Kxz;
end