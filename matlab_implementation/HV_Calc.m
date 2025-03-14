function [hv] = HV_Calc(goal,ref,pareto)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Created by Danial Khatamsaz: dKhatamsaz@gmail.com , khatamsaz@tamu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% this function calculates the hypervolume

% goal : a row vector to define which objectives to be minimized or
% maximized. zero for minimizing and 1 for maximizing. Example: [ 0 0 1 0 ... ]

% ref : hypervolume reference for calculations

% pareto : Current true pareto front obtained so far

%%%%%%%%%%% Note that in all variables, the order of columns should be the
%%%%%%%%%%% same. For example, the 1st column of all matrices above is
%%%%%%%%%%% related to the objective 1. Basically, each row = 1 design
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_obj = size(pareto,2);


%% Turn the problem into minimizing for all objectives:
for i = 1 : size(goal,2)
    if goal(i)==1
        pareto(:,i)=-1*pareto(:,i);
    end
end

%% Sorting the non_dominated points considering the first objective
%%%%% values
[~,d]=sort(pareto(:,1));
pareto = pareto(d,:);


%% HV calculation

hv = recursive_HV(ref,pareto);

end