% 清理工作区和命令行
clear; clc; close all;

% 1. 定义课件中的采样点 (x0)
x0 = [0; 0.2; 0.4; 0.6; 0.8; 1.0];

% 2. 根据 True function 计算观测值 (y0)
% 公式: y = sin(6x - 2) * x^0.5
y0 = sin(6*x0 - 2) .* (x0.^0.5);

% 打印 y0 以验证与图片表格中的数据是否一致
disp('观测值 y0:');
disp(y0);

% 3. 准备绘图环境
figure('Position', [100, 100, 800, 600]);
hold on; grid on;
xlabel('x');
ylabel('y');
% 注意：使用 LaTeX 解析器时，数学公式需要用 $ 符号包围
title('Polynomial Regression of $y = \sin(6x-2)x^{0.5}$', 'Interpreter', 'latex');

% 生成用于绘制平滑曲线的高密度测试点
x_test = linspace(0, 1, 200)';
y_true = sin(6*x_test - 2) .* (x_test.^0.5);

% 绘制真实的函数曲线和离散的采样点
plot(x_test, y_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Function');
scatter(x0, y0, 80, 'ko', 'filled', 'DisplayName', 'Observations');

% 设置6种不同的颜色用于区分拟合曲线
colors = lines(6);

% 4. 循环进行 1阶 到 5阶 的多项式拟合
for order = 1:5
    
    % 使用 polyfit 进行多项式拟合
    % 返回的 p 是多项式系数，按降幂排列: p_n*x^n + ... + p_1*x + p_0
    [p, S] = polyfit(x0, y0, order);
    
    % 使用 polyval 计算拟合曲线在测试点上的 y 值
    y_fit = polyval(p, x_test);
    
    % 绘制拟合曲线
    plot(x_test, y_fit, 'Color', colors(order, :), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('%d-order PR', order));
     
    % 将 MATLAB 的降幂系数转换为课件中的升幂 beta (beta_0, beta_1...)
    beta_hat = flip(p)';
    
    % 在控制台输出每次拟合的参数，方便与课件对照
    fprintf('--- %d阶多项式模型参数 (升幂 \\beta) ---\n', order);
    disp(beta_hat);
end

% 添加图例并放在图表外部
legend('Location', 'northeastoutside');
hold off;