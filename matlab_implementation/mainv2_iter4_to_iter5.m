clc; clear all; clc; close all

toolboxesRequired = [
    "Parallel Computing Toolbox",...
    "Statistics and Machine Learning Toolbox"
    ];
toInstalled = string({ver().Name});
toMissing = toolboxesRequired(~ismember(toolboxesRequired, toInstalled));
toError = sprintf('Missing toolboxes: %s', strjoin(toMissing, ', '));
toConfirm = sprintf('Required toolboxes ✔\n');
if ~isempty(toMissing); error(toError); else; fprintf(toConfirm); end %#ok<SPERR>

filesRequired = [
    "HV_Calc.m",...
    "Kernel.m",...
    "MeanCov.m",...
    "Pareto_finder.m",...
    "recursive_HV.m"
    ];
fiInDirectory = {dir().name};
fiMissing = filesRequired(~ismember(filesRequired, fiInDirectory));
fiError = sprintf('Missing files: %s', strjoin(toMissing, ', '));
fiConfirm = sprintf('Required files ✔\n');
if ~isempty(fiMissing); error(fiError); else; fprintf(fiConfirm); end %#ok<SPERR>

% outs: means of data
% sns: standard deviations; if you only have one, add like 0.1% of value to make
% sure COV matrix has nonzero diagonals (kernal function will break)
load Final_data_LSFE_iter4
load space
load in
load out1
load out2
load out3
load sn1
load sn2
load sn3

% Parallelization line -> determine # of cores
parpool(24, 'IdleTimeout', Inf); % 15x time (1.5*2*3*2*6 if 48 cores)
rng('shuffle');

%% This is for Low SFE
% profile on -history
%% set parameters
% system ['Co', 'Cr', 'Fe', 'Ni', 'V', 'Al']
% objectives:
% 1. Maximize tensile/yield strength ratio
% 2. Maximize Hardness at a high strain rate (at 0.05/s)
% 3. Maximize strain rate sensitivity %%% multiplied by 100
% itr = 1; % number of iterations
% N_dim = 6; % number of input dimensions
% N_test = 1500;
% N_alt=384; %% check for this later

N_samp = 1; % number of samples generated at each alternative to compute the average EHVI

% N_GP = 1000; %% check this later 
% goal = [ 1 1 1];
% ref = [0 0 20 ];
% batch_size=8;

ref = [ 0  0  0 ];
candidates=[];
improvements=[];

cand = space;
ind=find(ismember(space,in,'rows'));
cand(ind,:)=[];
bad_samps=[0.55	0.1	0.1	0.1	0.15 0];
ind=find(ismember(space,bad_samps,'rows'));
cand(ind,:)=[];
x_test=cand;

%%% scaling SRS
sn3=sn3.*100;
out3=out3.*100;

%%% samples to exclude AAB02
ex=[10];
in(ex,:)=[];
out1(ex)=[];
out2(ex)=[];
out3(ex)=[];
sn1(ex)=[];
sn2(ex)=[];
sn3(ex)=[];

%%% desire to find ~20%-25% above the best
% GP uncertainty for each of the objectives
% Tell the GP "what is the possible maximum out there"
% For the next iteration, this number probably exists, so increase the
% uncertainty ;; this is covered by 95% CI of the GP
% stdev for 12 should be 6 ; sf would be 36 (squared)
% If you know material upper bound, we can set this and keep it fixed
% Every time, we're assuming something larger -> dynamically updating
% Have been multiplying SRS by 100 to maintain a scale
% A similar scale helps the GP, improvement of tiny scale will have low
% significance on HV, it may ignore

sf1 = 1.65^2;
sf2 = 2.3^2;
sf3 = 0.54^2;

% BBO based on how many number of GPs you want to generate; generate length
% scales -> use same for all iterations
% We chose 1,000, hyperparameter_1 has them
% It's "not enough", but our feasible space is limited (1500 designs)

% hyperparameter_1 = lhsdesign(N_GP,N_dim).*0.9+0.05;
% load hyperparameter_1.mat

% II = zeros(N_GP,1);
% hp1_count_1 = [ hyperparameter_1 , II];
% N_Optimization=0;

% load constraints_data
% out=out-200; %%% GPs are shifted if needed (check sf values)
% Current data iteration
iter=4;

% Recalculating EHVI for table for each iteration (not necessary)
% [y_pareto_truth,ind] = Pareto_finder([out1(1:8) out2(1:8) out3(1:8)],goal);
% hv_truth(1) = HV_Calc(goal,ref,y_pareto_truth);
% [y_pareto_truth,ind] = Pareto_finder([out1(1:15) out2(1:15) out3(1:15)],goal);
% hv_truth(2) = HV_Calc(goal,ref,y_pareto_truth);
% [y_pareto_truth,ind] = Pareto_finder([out1(1:23) out2(1:23) out3(1:23)],goal);
% hv_truth(3) = HV_Calc(goal,ref,y_pareto_truth);
% [y_pareto_truth,ind] = Pareto_finder([out1(1:31) out2(1:31) out3(1:31)],goal);
% hv_truth(4) = HV_Calc(goal,ref,y_pareto_truth);
% x_pareto_truth = in(ind,:);

[y_pareto_truth,ind] = Pareto_finder([out1 out2 out3],goal);
x_pareto_truth = in(ind,:);
hv_truth = [hv_truth HV_Calc(goal,ref,y_pareto_truth)]

% Can use any number of cores for parallel for loop
% Each GP gets different set of hyperparameters
% Kernel may take time, but not for ~30 datapoints
% A few thousand can cause Kernel to take time & memory (size of matrix
% squared)
% Best samples are saved in candidates along with associated improvement
% Should have 1,000 suggestions, same as GPs

parfor kk = 1 : N_GP
    kk
    % Prints out the loops, can see iterations in log file
    % note that HPRC has no sequencing
    % Can still see seconds to update for logfile -> estimate computational
    % time for code.
    
    xtrain1 = in;
    ytrain1 = out1;
    % sf1 = sf1;
    l1 = hyperparameter_1(kk,:);
    % sn = sn;
    Kxx1 = Kernel(xtrain1,xtrain1,sf1,l1);
    gp1 = cell(1,6);
    gp1{1} = xtrain1;
    gp1{2} = ytrain1;
    gp1{3} = sf1;
    gp1{4} = l1;
    gp1{5} = sn1; %%% enter this manually for all observations
    gp1{6} = Kxx1;
    GP1 = cell2struct(gp1,{'xtrain','ytrain','sf','l','sn','Kxx'},2);
    
    xtrain2 = in;
    ytrain2 = out2;
    % sf2 = sf2;
    l2 = hyperparameter_1(kk,:);
    Kxx2 = Kernel(xtrain2,xtrain2,sf2,l2);
    gp2 = cell(1,6);
    gp2{1} = xtrain2;
    gp2{2} = ytrain2;
    gp2{3} = sf2;
    gp2{4} = l2;
    gp2{5} = sn2; %%% enter this manually for all observations
    gp2{6} = Kxx2;
    GP2 = cell2struct(gp2,{'xtrain','ytrain','sf','l','sn','Kxx'},2);
    
    xtrain3 = in;
    ytrain3 = out3;
    % sf3 = sf3;
    l3 = hyperparameter_1(kk,:);
    Kxx3 = Kernel(xtrain3,xtrain3,sf3,l3);
    gp3 = cell(1,6);
    gp3{1} = xtrain3;
    gp3{2} = ytrain3;
    gp3{3} = sf3;
    gp3{4} = l3;
    gp3{5} = sn3; %%% enter this manually for all observations
    gp3{6} = Kxx3;
    GP3 = cell2struct(gp3,{'xtrain','ytrain','sf','l','sn','Kxx'},2);
    
    % Querying the GPs for all the feasible samples
    % Predicts possible outcomes & uncertainty
    % Takes some time, not for small input datasets
    [y1_1,sig1_1] = MeanCov(GP1,x_test);
    [y1_2,sig1_2] = MeanCov(GP2,x_test);
    [y1_3,sig1_3] = MeanCov(GP3,x_test);
    
    % EHVI using the above
    ehvi = EHVI([y1_1 y1_2 y1_3],[(abs(diag(sig1_1))).^0.5 (abs(diag(sig1_2))).^0.5 (abs(diag(sig1_3))).^0.5],goal,ref,y_pareto_truth);
    [m1,x_star1] = max(ehvi);
    % x_star1 is index of sample with maximum improvement to HV
    % Don't need improvement value
    % Only thing that is important is candidate that has been selected

    % All samples you want to test (x_test)
    % Continuous space, have to determine points
    % Use latin hypercube sampling, choose best samples
    % Our discrete space, use all of them
    % Will select best next experiment form x_test

    candidates(kk,:)=x_test(x_star1,:);
    improvements(kk,1)=m1;

    % Multiple iteration runs
    % "Every 2 iterations, save the data"
    % Nothing lost if something happens or job is terminated
    % if rem(kk,2)==0
    %    save data_iter4_v1_1
    % else
    %    save data_iter4_v1_2
    % end
end

save data_iter5

% EHVI improvements
% How many unique designs have been suggested
% All post

%%% no model selection since there is only one model available

% Using K-Medoids to get samples to test from Candidates
% Using unique datapoints, tracking expected improvement for each candidate
% Tracking how many times each candidate is selected
% "Which sample has large expected improvement or selected many times"
% Length scales with overfitting will give you fake, large expected
% improvements -> don't just take the largest 8 (You do not know)
% Duplicates more trustable, but K-Medoid problem is ideal, less subjective

% %% Solving K-Medoids problem here
% 
% %%%%%%%%%%%%%%%%%%% Taking all the candidates together%%%%%%%%
% 
% Quick analysis & then solve K-Medoid problem

% load data_iter5
% 
% [c1,c2,c3]=unique(candidates,'rows');
% 
% repeats=[]; %%% tells the number of repeated designs in c1
% best_repeated=[];
% best_repeated_imp=[];
% high_repeated=[];
% best_repeated_ind=[];
% 
% for i = 1 : size(c1,1)
%     ind=find(ismember(candidates,c1(i,:),'rows'));
%     repeats(i,1)=size(ind,1);
%     imp=improvements(ind);
%     [~,bb]=max(imp);
%     best_repeated(i,:)=candidates(ind(bb),:);
%     best_repeated_imp(i,1)=improvements(ind(bb),:);
%     best_repeated_ind(i,:)=ind(bb);
% end
% 
% [repeats_sorted,bb]=sort(repeats,'descend');
% high_repeated=c1(bb,:);
% 
% [bests_sorted,bb]=sort(best_repeated_imp,'descend');
% best_repeated_sorted=c1(bb,:);
% 
% [~,batch_points,~,~,batch_index] = kmedoids(c1,batch_size);
% 
% % for bbb = 1 : batch_size
% %     hp1_count_1(batch_index(bbb),N_dim+1)=hp1_count_1(batch_index(bbb),N_dim+1)+1;
% % end
% 
% 
% Optimization_query_candidate = batch_points
% csvwrite('Iter5_candidate_alloys_Low_SFE.csv',Optimization_query_candidate)
% save Final_data_LSFE_iter5

% p = profile('info')
% save myprofiledata p

