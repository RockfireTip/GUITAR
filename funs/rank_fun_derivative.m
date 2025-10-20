function y = rank_fun_derivative(x,delta,rank_fun)
switch rank_fun
    case 1
        y = (delta .* sech(x).^2 .* (1 + delta)) ./ (delta + tanh(x)).^2;
    otherwise
        error('Unsupported rank function');
end
