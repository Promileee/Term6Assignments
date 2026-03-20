function StructuralOptimization()
    x0 = [0.2, 0.05];
    lb = [1e-3, 1e-4]; 
    ub = [1, 1];
    
    options = optimoptions('fmincon', 'Display', 'final', 'Algorithm', 'sqp');
    [x_opt, fval] = fmincon(@obj_fun, x0, [], [], [], [], lb, ub, @const_fun, options);
    
    fprintf('求解器找到的最优解:\n R = %.4f m\n t = %.4f m\n', x_opt(1), x_opt(2));

    R_axis = linspace(0.05, 0.5, 500);
    t_axis = linspace(0.001, 0.15, 500);
    [R_grid, T_grid] = meshgrid(R_axis, t_axis);

    P = 10e6; sigma_y = 248e6; E = 207e9; L = 5;

    t_g1 = P ./ (2 * pi * R_axis * sigma_y);
    t_g2 = (4 * P * L^2) ./ (pi^3 * R_axis.^3 * E);

    Obj_grid = 2 * pi * R_grid .* T_grid;

    figure('Color', 'w'); hold on; grid on;

    [C, h] = contour(R_grid, T_grid, Obj_grid, 30, 'LineWidth', 0.5, 'LineStyle', ':');
    clabel(C, h, 'FontSize', 7);

    p1 = plot(R_axis, t_g1, 'r', 'LineWidth', 2.5, 'DisplayName', 'g_1: Strength Boundary');
    p2 = plot(R_axis, t_g2, 'b', 'LineWidth', 2.5, 'DisplayName', 'g_2: Buckling Boundary');

    t_lower_bound = max(t_g1, t_g2);
    fill([R_axis, fliplr(R_axis)], [t_lower_bound, 0.2*ones(size(R_axis))], ...
        [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Feasible Region');

    plot(x_opt(1), x_opt(2), 'kp', 'MarkerFaceColor', 'y', 'MarkerSize', 12, 'DisplayName', 'Optimal Point (fmincon)');

    title('Structural Optimization: Solver Result & Design Space');
    xlabel('Radius R (m)'); ylabel('Thickness t (m)');
    legend('Location', 'northeast');
    axis([0.05 0.5 0 0.12]);

    R_inter = sqrt( (8 * L^2 * sigma_y) / (pi^2 * E) );
    plot(R_axis(R_axis >= R_inter), t_g1(R_axis >= R_inter), 'g--', 'LineWidth', 2, 'DisplayName', 'Optimal Range');

    hold off;
end

function f = obj_fun(x)
    f = 2 * pi * x(1) * x(2);
end

function [c, ceq] = const_fun(x)
    R = x(1); t = x(2);
    P = 10e6; sigma_y = 248e6; E = 207e9; L = 5;
    
    c(1) = P / (2 * pi * R * t) - sigma_y; 
    c(2) = P - (pi^3 * R^3 * t * E) / (4 * L^2);
    ceq = [];
end