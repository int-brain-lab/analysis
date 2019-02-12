

function y = dpr(stim, resp, what)

[d, crit] = dprime(stim, resp);
switch what
    case 'dprime'
        y = d;
    case 'criterion'
        y = crit;
end
end
