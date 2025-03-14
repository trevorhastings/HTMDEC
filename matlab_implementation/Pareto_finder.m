function [pareto,ind] = Pareto_finder(V,goal)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Created by Danial Khatamsaz: dKhatamsaz@gmail.com , khatamsaz@tamu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% V is a matrix, each row is the objectives of one design points

% goal : a row vector to define which objectives to be minimized or
% maximized. zero for minimizing and 1 for maximizing. Example: [ 0 0 1 0 ... ]


%% Turn the problem into minimizing for all objectives:
for i = 1 : size(goal,2)
    if goal(i)==1
        V(:,i)=-1*V(:,i);
    end
end

%%

pareto=[];
ind=[];

for i = 1 : size(V,1)
    
    p = V(i,:);
    s = V;
    s(i,:)=[];
    trig = 0;
    
    for j = 1 : size(s,1)
        
        temp = p-s(j,:);
        if min(temp)>=0
            trig=1; %%% this means vector p is dominated
        end
    end
    
    if trig==0
        pareto = [pareto ; p];
        ind = [ind ; i];
    end
    
end

%% Changing back the signs if were changed before.

for i = 1 : size(goal,2)
    if goal(i)==1
        pareto(:,i)=-1*pareto(:,i);
    end
end

end
