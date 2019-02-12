
function y = bootstrappedCI(x, fun, bound)

try
    fun = str2func(fun);
    ci = bootci(2000,fun,x);
    switch bound
        case 'low'
            y = fun(x) - ci(1);
        case 'high'
            y = ci(2) - fun(x);
    end
catch
    % if for some reason we can't bootstrap (e.g. there is only 1
    % datapoint), no errorbars
    y = 0;
end

end