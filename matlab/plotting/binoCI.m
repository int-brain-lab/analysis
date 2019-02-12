

function outp = binoCI(x)

[binomP, binoCI] = binofit(sum(x),numel(x));
outp = [[binomP, binoCI]];
end