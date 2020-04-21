% copmute rate for correct, unknown and incorrect samples based on value of
% threshold
% if any sample has membership below value of threshold, it is classified
% as unknown
function [correct, unknown, incorrect] = accuracy(expOut, compOut, memb, threshold)
% number of samples
samplesCount = length(expOut);
cor = expOut == compOut;
unk = memb < threshold;

temp = cor - unk;
temp(temp<0) = 0; 

correct = sum(temp)/samplesCount;
unknown = sum(unk)/samplesCount;
incorrect = 1 - correct - unknown;

end
