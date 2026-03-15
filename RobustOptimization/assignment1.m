function optimize_structure
    % 1. 定义初始猜测值 [R, t]
    x0 = [0.1, 0.01]; 
    
    % 2. 定义变量的下界 (R > 0, t > 0)
    lb = [1e-6, 1e-6]; 
    ub = []; % 无上界
    
    % 3. 使用 fmincon 进行优化
    options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');
    
    [x_opt, fval, exitflag, output] = fmincon(@objective, x0, [], [], [], [], lb, ub, @constraints, options);
    
    % 4. 输出结果
    fprintf('\n--- 优化结果 ---\n');
    fprintf('最优半径 R: %.4f m\n', x_opt(1));
    fprintf('最优厚度 t: %.4f m\n', x_opt(2));
    fprintf('最小目标函数值 (系数归一化后): %.4f\n', fval);
end

% --- 目标函数 ---
function f = objective(x)
    R = x(1);
    t = x(2);
    % 忽略常数 rho 和 l，设为 1
    f = 2 * pi * R * t; 
end

% --- 约束条件 ---
function [c, ceq] = constraints(x)
    R = x(1);
    t = x(2);
    P = 10e6;
    sigma_y = 248e6;
    E = 207e9;
    L = 5;
    
    % g1: 强度约束
    c(1) = P / (2 * pi * R * t) - sigma_y;
    
    % g2: 稳定性/欧拉失稳约束 (注意原式 P - ... <= 0)
    c(2) = P - (pi^3 * R^3 * t * E) / (4 * L^2);
    
    ceq = []; % 无等式约束
end