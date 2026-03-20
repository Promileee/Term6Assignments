function optimize_structure

    x0 = [0.1, 0.01]; 
   
    lb = [1e-6, 1e-6]; 
    ub = [];

    options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');
    
    [x_opt, fval, exitflag, output] = fmincon(@objective, x0, [], [], [], [], lb, ub, @constraints, options);
    
    fprintf('\n--- 优化结果 ---\n');
    fprintf('最优半径 R: %.4f m\n', x_opt(1));
    fprintf('最优厚度 t: %.4f m\n', x_opt(2));
    fprintf('最小目标函数值 (系数归一化后): %.4f\n', fval);
end

function f = objective(x)
    R = x(1);
    t = x(2);

    f = 2 * pi * R * t; 
end

function [c, ceq] = constraints(x)
    R = x(1);
    t = x(2);
    P = 10e6;
    sigma_y = 248e6;
    E = 207e9;
    L = 5;

    c(1) = P / (2 * pi * R * t) - sigma_y;

    c(2) = P - (pi^3 * R^3 * t * E) / (4 * L^2);
    
    ceq = [];
end