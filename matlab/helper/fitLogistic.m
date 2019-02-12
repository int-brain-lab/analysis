function [bias, slope, lapseLow, lapseHigh] = fitLogistic(x,y)

% if there are NaNs, remove
nanIdx = isnan(x) | isnan(y);
x(nanIdx) = [];
y(nanIdx) = [];

% based on the range of X, decide starting point and range
% for slope and bias
b = glmfit(x,y, 'binomial', 'link', 'logit');

% make gamma and lambda symmetrical
[pBest,~,exitflag,~] = fminsearchbnd(@(p) logistic_LL(p, ...
    x, y), [b(1) b(2) 0.1 0.1], [min(x) 0 0 0], [max(x) b(2)*10 1 1]);
try
    assert(exitflag == 1); % check that this worked
    
    bias        = pBest(1);
    slope       = pBest(2);
    lapseLow    = pBest(3);
    lapseHigh   = pBest(4);
catch
    bias = NaN;
    slope = NaN;
    lapseLow = NaN;
    lapseHigh = NaN;
end

end

function err = logistic_LL(p, intensity, responses)
% see http://courses.washington.edu/matlab1/Lesson_5.html#1

% compute the vector of responses for each level of intensity
w   = logistic(p, intensity);

% negative loglikelihood, to be minimised
err = -sum(responses .*log(w) + (1-responses).*log(1-w));

end

function y = logistic(p, x)
% Parameters: p(1) bias
%             p(2) slope
%             p(3) lapse rate-low (guess rate)
%             p(4) lapse rate-high (lapse rate)
%             x   intensity values.

% include a lapse rate, see Wichmann and Hill parameterisation
y =  p(3)+(1-p(3)-p(4)) * (1./(1+exp(- ( p(1) + p(2).*x ))));


end