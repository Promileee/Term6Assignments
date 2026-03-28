clear; clc; close all;

cmap = jet(256);

[X1_h, X2_h] = meshgrid(linspace(-6, 6, 201));

f1 = (X1_h.^2 + X2_h - 11).^2 + (X1_h + X2_h.^2 - 7).^2;

levels1 = 5:15:140;

legend1_str = {
    'Contour ID', ...
    ['A    5.000'], ...
    ['B   20.000'], ...
    ['C   35.000'], ...
    ['D   50.000'], ...
    ['E   65.000'], ...
    ['F   80.000'], ...
    ['G   95.000'], ...
    ['H  110.000'], ...
    ['I  125.000'], ...
    ['J  140.000']
};

[X1_nl, X2_nl] = meshgrid(linspace(-2.4, 2.4, 201));

f2 = 2*X1_nl.^2 + 4*X1_nl.*X2_nl.^3 - 10*X1_nl.*X2_nl + X2_nl.^2;

levels2 = -40:10:50;

legend2_str = {
    'Contour ID', ...
    ['A  -40.000'], ...
    ['B  -30.000'], ...
    ['C  -20.000'], ...
    ['D  -10.000'], ...
    ['E    0.000'], ...
    ['F   10.000'], ...
    ['G   20.000'], ...
    ['H   30.000'], ...
    ['I   40.000'], ...
    ['J   50.000']
};

hFig = figure('Name', 'Multimodal and Nonlinear Functions: Contours and Surfaces', ...
              'Units', 'normalized', 'OuterPosition', [0 0 1 1], 'Color', 'w'); % 创建全屏白色背景窗口
colormap(cmap); % 应用颜色映射

% --- Subplot 1: Himmelblau's 等高线图 (左上) ---
subplot(2, 2, 1);
[C1, h1] = contour(X1_h, X2_h, f1, levels1, 'LineColor', 'k'); % 绘制等高线
title('Figure 3.1: Himmelblau''s Function (Contours)');
xlabel('x_1'); ylabel('x_2');
grid on; axis square; % 开启网格，强制比例一致
set(gca, 'XTick', -6:2:6, 'YTick', -6:2:6); % 设置刻度
clabel(C1, h1, 'FontSize', 8, 'LabelSpacing', 200, 'Color', 'k', 'FontName', 'Helvetica'); % 标注等高线上的数值

% --- 添加 Himmelblau's Legend (作为文本框) ---
% annotation('textbox', [0.35, 0.72, 0.1, 0.2], 'String', legend1_str, ...
%            'FitBoxToText', 'on', 'EdgeColor', 'none', 'BackgroundColor', 'w', ...
%            'FontName', 'Consolas', 'FontSize', 9); % 使用 Consolas 字体获得整齐的对齐效果

% --- Subplot 2: Himmelblau's 曲面图 (右上) ---
subplot(2, 2, 2);
surf(X1_h, X2_h, f1); % 绘制曲面图
shading interp; % 平滑颜色过渡，使其看起来更专业
title('Figure 3.1: Himmelblau''s Function (Surface)');
xlabel('x_1'); ylabel('x_2'); zlabel('f(x)');
grid on;
set(gca, 'XTick', -6:2:6, 'YTick', -6:2:6);
view(-37.5, 30); % 设置初始视角

%% 6. 绘制非线性函数 (Fig 3.3) 的等高线和曲面

% --- Subplot 3: 非线性函数等高线图 (左下) ---
subplot(2, 2, 3);
[C2, h2] = contour(X1_nl, X2_nl, f2, levels2, 'LineColor', 'k'); % 绘制等高线
title('Figure 3.3: Nonlinear Function (Contours)');
xlabel('x_1'); ylabel('x_2');
grid on; axis square;
set(gca, 'XTick', -2.4:1.2:2.4, 'YTick', -2.4:1.2:2.4);
clabel(C2, h2, 'FontSize', 8, 'LabelSpacing', 200, 'Color', 'k', 'FontName', 'Helvetica'); % 标注等高线数值

% --- 添加非线性函数 Legend (作为文本框) ---
% annotation('textbox', [0.85, 0.22, 0.1, 0.2], 'String', legend2_str, ...
%            'FitBoxToText', 'on', 'EdgeColor', 'none', 'BackgroundColor', 'w', ...
%            'FontName', 'Consolas', 'FontSize', 9);

% --- Subplot 4: 非线性函数曲面图 (右下) ---
subplot(2, 2, 4);
surf(X1_nl, X2_nl, f2); % 绘制曲面图
shading interp;
title('Figure 3.3: Nonlinear Function (Surface)');
xlabel('x_1'); ylabel('x_2'); zlabel('f(x)');
grid on;
set(gca, 'XTick', -2.4:1.2:2.4, 'YTick', -2.4:1.2:2.4);
view(-120, 20); % 调整初始视角，更好地观察复杂的峰和谷

%% 7. 添加主标题
sgtitle('Optimization Test Functions from Reklaitis et al.', 'FontSize', 16, 'FontWeight', 'bold');

%% 脚本结束