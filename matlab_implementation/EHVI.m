function [ehvi] = EHVI(means,sigmas,goal,ref,pareto)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Created by Danial Khatamsaz: dKhatamsaz@gmail.com , khatamsaz@tamu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% means : GP mean estimation of objectives of the test points (fused means in
% multifidelity cases). Each column for 1 objective values

% sigmas : uncertainty of GP mean estimations (std). Each column for 1 objective

% goal : a row vector to define which objectives to be minimized or
% maximized. zero for minimizing and 1 for maximizing. Example: [ 0 0 1 0 ... ]

% ref : hypervolume reference for calculations

% pareto : Current true pareto front obtained so far

%%%%%%%%%%% Note that in all variables, the order of columns should be the
%%%%%%%%%%% same. For example, the 1st column of all matrices above is
%%%%%%%%%%% related to the objective 1. Basically, each row = 1 design
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_obj = size(means,2); %% number of objectives


%% Turn the problem into minimizing for all objectives:
%%% this is essential as the method works for minimizing
for i = 1 : size(goal,2)
    if goal(i)==1
        means(:,i)=-1*means(:,i);
        pareto(:,i)=-1*pareto(:,i);
    end
end

%% Sorting the non_dominated points considering the first objective
%%%%% It does not matter which objective to sort but lets do it with the
%%%%% 1st objective
[~,d]=sort(pareto(:,1));
pareto = pareto(d,:);


ind = zeros(size(means,1),1);
ehvi = 0;

%% EHVI calculation for test points

for i = 1 : size(means,1)
    
    if ind(i)==1
        ehvi(i,1)=0;
    else
        
        hvi = 0;
        box = 1;
        %%% EHVI over the box from infinity to the ref point
        for j = 1 : N_obj
            s = (ref(j)-means(i,j))/sigmas(i,j);
            box = box*((ref(j)-means(i,j))*normcdf(s)+sigmas(i,j)*normpdf(s));
        end
        
        %%% calculate how much adding a test point can improve the hypervolume
        hvi = recursive(means(i,:),sigmas(i,:),ref,pareto);
        
        ehvi(i,1)=box-hvi;
        
    end
end

end