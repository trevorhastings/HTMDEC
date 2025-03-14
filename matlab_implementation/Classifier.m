function [means,var] = Classifier(GP,xtest)


n = size(GP.xtrain,1); % number of training points
C = size(GP.ytrain,2); % number of classes
K = GP.Kxx;
N_test = size(xtest,1);

for i = 1 : C
    K(:,:,i)=K(:,:,i)+(diag(GP.sn)).^2; %%% For stability of K
end

tol = 1e-5; % to check the convergence

f_vec = zeros(C*n,1);
% f_mat = GP.ytrain;
f_mat = zeros(n,C);
k = 0; % turn this to 1 once convergence met
p_mat = [];
p_vec = [];
II = [];
E = [];
z = [];
samples=10000;
objective_old = 0;

R=[];
f_vec_original=[];

for i = 1 : C
    R = [ R ; eye(n)];
end

for i = 1 : C
    f_vec_original = [f_vec_original ; GP.ytrain(:,i)];
end


while k == 0
    p_mat=[];
    p_vec=[];
    for i = 1 : n
        p_mat(i,:) = exp(f_mat(i,:))./sum(exp(f_mat(i,:)));
    end
    
    for i = 1 : C
        p_vec = [p_vec ; p_mat(:,i)];
    end
    
    II = [];
    for i = 1 : C
        II = [ II ; diag(p_mat(:,i))];
    end
    
    for i = 1 : C
        L=chol(eye(n)+ (diag(p_mat(:,i)))^0.5 * K(:,:,i) * (diag(p_mat(:,i)))^0.5);
        L=L'; %% L needs to be lower triangular but Matlab gives upper triangular using chol
        E(:,:,i) = (diag(p_mat(:,i)))^0.5 * (L' \ (L\((diag(p_mat(:,i)))^0.5)));
        z(i) = sum(log(diag(L)));
    end
    
    M = chol(sum(E,3));
    M = M';
    b = (diag(p_vec)-II*II')*f_vec + f_vec_original - p_vec;
    EE = zeros(C*n,C*n);
    shift = 1;
    for i = 1 : C
        EE(shift:shift+n-1,shift:shift+n-1)=E(:,:,i);
        shift = shift+n;
    end
       
    KK = zeros(C*n,C*n);
    shift = 1;
    for i = 1 : C
        KK(shift:shift+n-1,shift:shift+n-1)=K(:,:,i);
        shift = shift+n;
    end
    
%      EE = (diag(p_vec))^0.5 * inv(( eye(C*n) + (diag(p_vec))^0.5 * KK * (diag(p_vec))^0.5)) * (diag(p_vec))^0.5; % This might be unstable
    
    c = EE*KK*b;
    a = b-c+EE*R*(M'\(M\(R'*c))); %%% This needs to be checked (Now is corrected)
    f = KK*a;
    f_vec = f;
    
    f_mat = [];
    shift = 1;
    for i = 1 : C
        f_mat = [f_mat f_vec(shift:shift+n-1,1)];
        shift = shift+n;
    end
    
    objective = -0.5*a'*f + f_vec_original'*f - sum(log(sum(exp(f_mat),2))); %%% Note that this is corrected and GPML book expression is wrong
    
    if (abs(objective-objective_old))<tol
        k=1;
    end
    
    objective_old = objective;
    
end

q = objective - sum(z); %% This is lml


%%%%% xtest is in the game from now on

p_mat = [];
p_vec = [];

for i = 1 : n
    p_mat(i,:) = exp(f_mat(i,:))./sum(exp(f_mat(i,:)));
end

for i = 1 : C
    p_vec = [p_vec ; p_mat(:,i)];
end

II = [];
for i = 1 : C
    II = [ II ; diag(p_mat(:,i))];
end

E=[];
for i = 1 : C
    L=chol(eye(n)+ (diag(p_mat(:,i)))^0.5 * K(:,:,i) * (diag(p_mat(:,i)))^0.5);
    L=L'; %% L needs to be lower triangular but Matlab gives upper triangular using chol
    E(:,:,i) = (diag(p_mat(:,i)))^0.5 * (L' \ (L\(diag(p_mat(:,i)))^0.5));
end

M = chol(sum(E,3));
M=M';

means=[];
var=[];

for zz = 1 : N_test

mu_star = [];

K_star = [];

%%% in line below we can use a different kernel for each class
%%% for now we assume it is 1 kernel for all classes
for i = 1 : C
    K_star(:,:,i)=Kernel(GP.xtrain,xtest(zz,:),GP.sf,GP.l);
end

sigma = zeros(C,C);
for i = 1 : C
    mu_star(i) = (GP.ytrain(:,i) - p_mat(:,i))' * K_star(:,:,i);
    b = E(:,:,i) * K_star(:,:,i);
    c = E(:,:,i)*(M'\(M\(b))); %%% Note that this is corrected and GPML book expression is wrong
        
    for j = i : C
        sigma(i,j)=c'*K_star(:,:,j);
    end
    
    sigma(i,i) = sigma(i,i) + Kernel(xtest(zz,:),xtest(zz,:),GP.sf,GP.l) - b'*K_star(:,:,i);
end

ssigma=sigma';
sigma = sigma+ssigma;
for i = 1:C
sigma(i,i)=sigma(i,i)/2;
end

p_star=zeros(1,C);

f_star = mvnrnd(mu_star,sigma,samples);

for i = 1 : samples
    p_star = p_star + exp(f_star(i,:))./sum(exp(f_star(i,:)));
end


var_star = exp(f_star);
var_star_sum = sum(var_star,2);

for i = 1 : C
    var_star(:,i)=var_star(:,i)./var_star_sum;
end

var_star1 = (std(var_star)).^2;
% sigma
% std(f_star(:,1))
% std(f_star(:,2))
p_star = p_star./samples;

means(zz,:) = p_star;
var(zz,:) = var_star1;

% means(zz,:) = mu_star;
% var(:,:,zz) = sigma ;

end