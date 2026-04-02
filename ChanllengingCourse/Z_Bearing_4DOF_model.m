clc; clear; close all;
%% 参考文献：ZHANG Z, ZI Y, ZHANG M, 等. Deep Unsupervised Subdomain Adaptation Network for Intelligent Fault Diagnosis: From Simulated Domain to Physical Domain[J/OL]. IEEE Transactions on Instrumentation and Measurement, 2025, 74: 1-16. DOI:10.1109/TIM.2025.3604989.
%% 第一部分：参数定义与系统初始化（来自于文献以及文献所提及的文献[41])
% --- 1. 轴承几何参数 (以 SKF-6203-RS 为例) ---
param.Db = 6.746e-3;    % 滚动体直径 (m)
param.Dm = 28.5e-3;     % 节圆直径 (m)
param.Nb = 8;           % 滚动体个数
param.alpha = 0;        % 接触角 (rad)
param.Cr = 2e-6;        % 径向游隙 (m)

% --- 2. 系统动力学参数 ---
% 坐标系定义说明：
% Y轴：沿重力方向（向下为正）。承载区中心位于 theta = 0 (0度)。
% X轴：水平方向。
param.min = 50;        % 内圈及轴的等效质量 (kg)
param.mout = 5;       % 外圈及轴承座的等效质量 (kg)
param.kin = 7.42e7;        % 内圈支撑刚度 (N/m)
param.kout = 1.51e7;       % 外圈支撑刚度 (N/m)
param.cin = 1376.8;        % 内圈阻尼 (N.s/m)
param.cout = 2210.7;       % 外圈阻尼 (N.s/m)
param.Kc = 8.753e9;         % 赫兹接触刚度系数 (N/m^1.5)

% --- 3. 运行工况参数 ---
param.g = 9.8;                  % 重量加速度的设定
param.rpm = 1797;               % 转速 (r/min)
param.ws = param.rpm * 2*pi/60; % 轴旋转角频率 (rad/s)
param.fs = 12000;               % 采样频率 (Hz)
param.T = 1;                  % 仿真时长 (s)

% --- 4. 故障参数设置 ---
% 故障类型: 0=正常, 1=外圈故障(OF), 2=内圈故障(IF), 3=滚动体故障(BF)
param.fault_type = 0; %故障类型        
param.defect_width = 3.6e-4;      % 故障宽度 (m)
param.defect_depth = 1e-3;    % 故障深度 (m)

%% 第二部分：主仿真循环
fprintf('正在仿真工况: 故障类型 %d (0:正常, 1:外圈, 2:内圈, 3:滚动体)...\n', param.fault_type);

tspan = 0:1/param.fs:param.T;   % 时间向量
% 初始状态: [x_in, vx_in, y_in, vy_in, x_out, vx_out, y_out, vy_out]
y0 = zeros(8, 1); 

% 设置积分器精度
options = odeset('RelTol', 1e-5, 'AbsTol', 1e-6);

% 调用 ODE45 求解
% 注意：若遇到极度刚性的冲击问题，可考虑改用 ode15s
[t, Y] = ode45(@(t, y) bearing_dynamics_4dof(t, y, param), tspan, y0, options);

% ---重新计算加速度 ---
% 将ODE输出的位移和速度代回动力学方程，从而得到振动信号（即加速度，我们采用外圈y方向作为绘图加速度）
acc_out_y = zeros(length(t), 1);
for i = 1:length(t)
    dydt = bearing_dynamics_4dof(t(i), Y(i,:)', param);
    acc_out_y(i) = dydt(8); % 第8个状态变量是外圈Y方向加速度
end
disp('仿真完成。正在绘图...');

%% 第三部分：结果可视化 (时域与包络谱)
figure('Color', 'w', 'Position', [100, 100, 1000, 700]);

% --- 1. 时域波形绘制 ---
subplot(2,1,1);
cut_idx = round(length(t)*0.1); 
valid_t = t(cut_idx:end);
valid_sig = acc_out_y(cut_idx:end);

plot(valid_t, valid_sig, 'b');
fault_names = {'正常 (Normal)', '外圈故障 (Outer Race)', '内圈故障 (Inner Race)', '滚动体故障 (Ball)'};
title(['时域波形 - 当前状态: ', fault_names{param.fault_type + 1}], 'FontSize', 12);
xlabel('时间 (s)'); ylabel('加速度 (m/s^2)');
grid on; 
xlim([valid_t(1), valid_t(1)+0.9]); 

% --- 2. 包络谱分析 (Hilbert变换) ---
subplot(2,1,2);
sig_env = abs(hilbert(valid_sig));  
sig_env = sig_env - mean(sig_env);  

L = length(sig_env);
NFFT = 2^nextpow2(L);
Y_fft = fft(sig_env, NFFT)/L;
f = param.fs/2 * linspace(0, 1, NFFT/2+1);
Amp = 2*abs(Y_fft(1:NFFT/2+1));

plot(f, Amp, 'r', 'LineWidth', 1);
title('包络谱 (Envelope Spectrum)', 'FontSize', 12);
xlabel('频率 (Hz)'); ylabel('幅值');
xlim([0, 500]); grid on; 
hold on;

% --- 3. 标记理论特征频率 (包含边带) ---
gamma = param.Db / param.Dm;
fr = param.ws / (2*pi); % 转频

BPFO = (param.Nb/2) * fr * (1 - gamma*cos(param.alpha));
BPFI = (param.Nb/2) * fr * (1 + gamma*cos(param.alpha));
BSF  = fr * (param.Dm / (2*param.Db)) * (1 - (gamma * cos(param.alpha))^2);

is_fault_plotted = false; 

if param.fault_type == 1 % --- 外圈故障 ---
    xline(BPFO, '--k', 'Label', 'BPFO', 'LabelVerticalAlignment', 'top','LineWidth', 1.5);
    xline(2*BPFO, ':k', 'Label', '2BPFO','LineWidth', 1.5);
    is_fault_plotted = true;
    
elseif param.fault_type == 2 % --- 内圈故障  ---
    % 1. 标记主特征频率
    xline(BPFI, '--k', 'Label', 'BPFI', 'LabelVerticalAlignment', 'top', 'LineWidth', 1.5);
    xline(2*BPFI, ':k', 'Label', '2BPFI','LineWidth', 1.5);
    
    % 2. 标记转频 (fr)
    xline(fr, '-.b', 'Label', 'fr', 'LabelVerticalAlignment', 'top','LineWidth', 1.5);
    
    % 3. 标记调制边带 (BPFI ± fr)
    xline(BPFI - fr, '-.g', 'Label', '-fr', 'LabelHorizontalAlignment', 'right','LineWidth', 1.5);
    xline(BPFI + fr, '-.g', 'Label', '+fr', 'LabelHorizontalAlignment', 'left','LineWidth', 1.5);
    
    %标记 2BPFI ± fr
    xline(2*BPFI - fr, '-.g', 'Label', '-fr', 'LabelHorizontalAlignment', 'right','LineWidth', 1.5);
    xline(2*BPFI + fr, '-.g', 'Label', '+fr', 'LabelHorizontalAlignment', 'left','LineWidth', 1.5);
    
    is_fault_plotted = true;
    
elseif param.fault_type == 3 % --- 滚动体故障 ---
    xline(2*BSF, '--k', 'Label', '2xBSF', 'LabelVerticalAlignment', 'top','LineWidth', 1.5);
    xline(4*BSF, ':k', 'Label', '4xBSF','LineWidth', 1.5);
    is_fault_plotted = true;
end

% 图例
if is_fault_plotted
    if param.fault_type == 2
        legend('仿真信号包络', '理论特征频率 (BPFI)', '2BPFI','转频 (fr)', '调制边带 (±fr)');
    else
        legend('仿真信号包络', '理论特征频率');
    end
else
    legend('仿真信号包络'); 
end
%% 第四部分：动力学模型函数 
function dydt = bearing_dynamics_4dof(t, y, p)
    % 状态变量
    xin = y(1); vxin = y(2);
    yin = y(3); vyin = y(4);
    xout = y(5); vxout = y(6);
    yout = y(7); vyout = y(8);
    
    % --- A. 运动学计算 ---
    gamma = p.Db / p.Dm;
    wc = (p.ws / 2) * (1 - gamma * cos(p.alpha)); % 保持架角速度
    
    FHX = 0; % 水平方向总接触力
    FHY = 0; % 垂直方向总接触力
    
    % --- B. 循环计算滚动体接触力 ---
    for j = 1:p.Nb
        % 当前滚动体位置 (相位差 2pi/N)
        theta_curr = mod(wc * t + 2*pi*(j-1)/p.Nb, 2*pi);
        
        % 1. 计算故障变形量 Bj
        Bj = calculate_defect_Bj(t, theta_curr, p, gamma, wc);
        
        % 2. 计算总接触变形 delta
        % 坐标约定: sin对应X分量, cos对应Y分量 (这意味着theta=0对应Y轴正向，注意：Y轴正向竖直向下)
        % 承载区在底部(theta=0), 此时cos(pi)=1
        delta = (xin - xout)*sin(theta_curr) + (yin - yout)*cos(theta_curr) - p.Cr - Bj;
        
        % 3. 判断接触 (Heaviside)
        if delta > 0
            force = p.Kc * (delta^1.5);
            FHX = FHX + force * sin(theta_curr);
            FHY = FHY + force * cos(theta_curr);
        end
    end
    
    % --- C. 运动微分方程 ---
    % 1. 内圈 (承受径向载荷 Fr, 沿Y轴负向)
    ax_in = (-FHX - p.cin*vxin - p.kin*xin) / p.min;
    % 考虑重力 (+mg) 与 FHY 的方向关系
    ay_in = (-FHY - p.cin*vyin - p.kin*yin + p.min*p.g) / p.min; 
    
    % 2. 外圈
    ax_out = (FHX - p.cout*vxout - p.kout*xout) / p.mout;
    % 考虑重力 (+mg) 与 FHY 的方向关系
    ay_out = (FHY - p.cout*vyout - p.kout*yout + p.mout*p.g) / p.mout;
    
    dydt = [vxin; ax_in; vyin; ay_in; vxout; ax_out; vyout; ay_out];
end

%% 第五部分：故障变形计算辅助函数
function Bj = calculate_defect_Bj(t, theta_ball, p, gamma, wc)
    Bj = 0;
    if p.fault_type == 0, return; end
    
    % 几何半径定义
    phi_o = (p.Dm + p.Db)/2; 
    phi_i = (p.Dm - p.Db)/2; 
    phi_b = p.Db/2;          
    
    % 
    % 计算Co、Ci、Cb
    sagitta = phi_b - sqrt(max(0, phi_b^2 - (p.defect_width/2)^2));
    Co = min(sagitta, p.defect_depth);
    Ci = min(sagitta, p.defect_depth);
    Cb = min(sagitta, p.defect_depth);
    switch p.fault_type
        case 1 % --- 外圈故障 ---
            % 故障位置设为 0 (0度)，即轴承正下方
            fault_loc = 0; 
            beta_o = asin(p.defect_width / (2*phi_o)); 
            
            % 计算滚动体到故障中心的距离（考虑圆周周期性）
            dist = abs(theta_ball - fault_loc);
            dist = min(dist, 2*pi - dist);
            
            if dist < beta_o
                Bj = Co; 
            end
    
        case 2 % --- 内圈故障 ---
            beta_i = asin(p.defect_width / (2*phi_i));
            % 内圈故障点随轴转动
            rel_angle = mod(p.ws*t - theta_ball, 2*pi);
            
            % 归一化至 [-pi, pi]
            if rel_angle > pi, rel_angle = rel_angle - 2*pi; 
            end
            
            if abs(rel_angle) < beta_i 
                Bj = Ci; 
            end
            
        case 3 % --- 滚动体故障 ---
            beta_b = asin(p.defect_width / (2*phi_b));
            
            % 滚动体自转
            wb = ((1-gamma)*p.ws - wc) / gamma; 
            wb_spin = (wc + wb)/2; 
            
            % 模拟滚动体表面缺陷周期性接触滚道
            spin_angle = mod(wb_spin * t, pi);
            if spin_angle > pi/2, spin_angle = spin_angle - pi; end
            
            if abs(spin_angle) < beta_b
                Bj = Cb;
            end
    end
end