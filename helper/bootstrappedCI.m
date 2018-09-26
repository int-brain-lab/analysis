
function y = bootstrappedCI(x, fun, bound)

fun = str2func(fun);
ci = bootci(2000,fun,x);
switch bound
    case 'low'
        y = fun(x) - ci(1);
    case 'high'
        y = ci(2) - fun(x);
end

end