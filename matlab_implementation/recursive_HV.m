function hypervolume = recursive_HV(ref,pareto)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Created by Danial Khatamsaz: dKhatamsaz@gmail.com , khatamsaz@tamu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_obj=size(pareto,2);

if size(pareto,1)==1
    
    hv_temp = 1;
    
    for j = 1 : N_obj
        len = ref(j)-pareto(j);
        hv_temp = hv_temp * len;
    end
    
    hypervolume = hv_temp;
    
else
    hypervolume=0;
    hv_temp = 1;   
    pareto_prime=[];
    
    for j = 1 : N_obj
        len = ref(j)-pareto(1,j);
        hv_temp = hv_temp * len;
    end
    
    for z = 2:size(pareto,1)
        temp = max([pareto(1,:);pareto(z,:)]);
        pareto_prime = [pareto_prime ; temp];
    end
    
    [pareto_prime,~] = Pareto_finder(pareto_prime,zeros(1,N_obj));
    
    hv_temp2 = recursive_HV(ref,pareto_prime);
    
    pareto(1,:)=[];
    
    improve2 = recursive_HV(ref,pareto);
    
    hypervolume = improve2 + hv_temp - hv_temp2;
    
end



