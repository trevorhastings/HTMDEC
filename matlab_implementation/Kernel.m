
function K = Kernel(dataset1,dataset2,sf,l)

K = zeros(size(dataset1,1),1);

L = repmat(l,size(dataset1,1),1);
for ii = 1:size(dataset2,1)
    K(:,ii) = ((sf)*exp(-(0.5*sum((dataset1./L-repmat(dataset2(ii,:)./l,size(dataset1,1),1)).^2,2))));
end
end

%for jj=1:size(dataset1,1)
%for ii = 1:size(dataset2,1)
%    K(jj,ii) = ((sf)*exp(-(0.5*sum((dataset1(jj,:)./l-dataset2(ii,:)./l).^2))));
%end
%end
%end