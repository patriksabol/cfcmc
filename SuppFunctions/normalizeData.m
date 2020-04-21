function data = normalizeData(set)
dataNorm = normalize(set(:,1:end-1),1);
if(any(set(:,end)<1))
    set(:,end) = set(:,end) + 1;
end
data = [dataNorm,set(:,end)];
end