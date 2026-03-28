clc; clear; close all;

%% 第一部分：6-DOF 系统参数定义
% --- 1. 轴承几何参数 (SKF-6203-RS) ---
param.Db = 6.746e-3;    
param.Dm = 28.5e-3;     
param.Nb = 8;           
param.alpha = 0;        
param.Cr = 2e-6;        

% --- 2. 动力学参数 (内圈、外圈、保持架) ---
param.g = 9.8;
param.rpm = 1797;               
param.ws = param.rpm * 2*pi/60; 

% 内外圈参数
param.min = 50;   param.kin = 7.42e7;   param.cin = 1376.8;        
param.mout = 5;   param.kout = 1.51e7;  param.cout = 2210.7;       
param.Kc = 8.753e9; % 赫兹接触刚度

% 新增：保持架参数 (6-DOF 核心)
param.mc = 0.5;         % 保持架质量 (kg)
param.kc = 1.0e6;       % 保持架油膜/引导支撑刚度 (N/m)
param.cc = 800;         % 保持架阻尼 (N.s/m)
param.c_couple = 0.2;   % 保持架位移对滚动体挤压的耦合系数 (软约束系数)

% --- 3. 仿真与故障工况设置 ---
param.fs = 12000;               
param.T = 1.0; % 仿真时长设为 1s
param.defect_width = 3.6e-4;      
param.defect_depth = 1e-3;    

% =========================================================================
% 请在此处修改故障类型 (0~5)
% 0:正常, 1:外圈, 2:内圈, 3:滚动体, 4:保持架断裂, 5:保持架疲劳+碎屑
param.fault_type = 0; 
% =========================================================================

% 预设情况 4 (断裂) 和 5 (疲劳+碎屑) 的特定参数
% 情况 4: 保持架断裂产生的不平衡质量偏心矩 (kg.m)
param.U_cage = 5e-4; 

% 情况 5: 疲劳相位游荡与碎屑
rng(42); 
param.fatigue_amp = 0.5 * (pi/180); 
param.fatigue_phases = rand(param.Nb, 1) * 2 * pi;
param.fatigue_freqs = 5 + 10 * rand(param.Nb, 1); 
param.num_debris = 10; 
param.debris_angles = rand(param.num_debris, 1) * 2 * pi; 
param.debris_heights = (5 + 20 * rand(param.num_debris, 1)) * 1e-6; 
param.debris_width = 1.0 * (pi/180); 

%% 第二部分：主仿真循环 (12个状态变量求解)
fault_names = {'正常', '外圈故障', '内圈故障', '滚动体故障', '保持架断裂 (不平衡涡动)', '保持架疲劳与碎屑干扰'};
fprintf('正在仿真 6-DOF 模型 - 工况: %s...\n', fault_names{param.fault_type + 1});

tspan = 0:1/param.fs:param.T;   
% 初始状态 12x1: [xin, vxin, yin, vyin, xout, vxout, yout, vyout, xc, vxc, yc, vyc]
y0 = zeros(12, 1); 
options = odeset('RelTol', 1e-4, 'AbsTol', 1e-5); % 适当放宽精度防止 12-DOF 刚性卡死

tic;
[t, Y] = ode45(@(t, y) bearing_dynamics_6dof(t, y, param), tspan, y0, options);
toc;

% 提取外圈 Y 向加速度 (第8个状态变量的导数)
acc_out_y = zeros(length(t), 1);
for i = 1:length(t)
    dydt = bearing_dynamics_6dof(t(i), Y(i,:)', param);
    acc_out_y(i) = dydt(8); 
end
disp('求解完成，正在生成时频特征图...');

%% 第三部分：结果可视化与特征频率对比
figure('Color', 'w', 'Position', [100, 100, 1000, 700], 'Name', '6-DOF 轴承动力学仿真');

% --- 1. 时域波形 ---
subplot(2,1,1);
cut_idx = round(length(t)*0.1); 
valid_t = t(cut_idx:end);
valid_sig = acc_out_y(cut_idx:end);
% 增加微弱底噪模拟真实传感器
valid_sig = awgn(valid_sig, 25, 'measured'); 

plot(valid_t, valid_sig, 'b', 'LineWidth', 0.8);
title(['时域加速度波形 - ', fault_names{param.fault_type + 1}], 'FontSize', 12);
xlabel('时间 (s)'); ylabel('加速度 (m/s^2)');
grid on; xlim([valid_t(1), valid_t(1)+0.5]); 

% --- 2. 包络谱分析 ---
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
xlim([0, 400]); grid on; hold on;

% --- 3. 标记特征频率 ---
gamma = param.Db / param.Dm;
fr = param.ws / (2*pi); 
BPFO = (param.Nb/2) * fr * (1 - gamma*cos(param.alpha));
BPFI = (param.Nb/2) * fr * (1 + gamma*cos(param.alpha));
BSF  = fr * (param.Dm / (2*param.Db)) * (1 - (gamma * cos(param.alpha))^2);
FTF  = (fr/2) * (1 - gamma*cos(param.alpha)); 

% 动态图例与标线
if param.fault_type == 1
    xline(BPFO, '--k', 'BPFO', 'LineWidth', 1.5); xline(2*BPFO, ':k', '2BPFO');
elseif param.fault_type == 2
    xline(BPFI, '--k', 'BPFI', 'LineWidth', 1.5); 
    xline(BPFI-fr, '-.g', '-fr'); xline(BPFI+fr, '-.g', '+fr');
elseif param.fault_type == 3
    xline(2*BSF, '--k', '2BSF', 'LineWidth', 1.5);
elseif param.fault_type == 4 || param.fault_type == 5
    xline(FTF, '--m', 'FTF (fc)', 'LineWidth', 1.5);
    xline(2*FTF, ':m', '2fc'); xline(fr, '-.b', 'fr');
    if param.fault_type == 5, xline(BPFO, '--k', 'BPFO(受碎屑干扰)'); end
end

%% 第四部分：6-DOF 动力学核心函数
function dydt = bearing_dynamics_6dof(t, y, p)
    % 解包 12 个状态变量
    xin = y(1); vxin = y(2); yin = y(3); vyin = y(4);
    xout= y(5); vxout= y(6); yout= y(7); vyout= y(8);
    xc  = y(9); vxc  = y(10);yc  = y(11);vyc  = y(12);
    
    gamma = p.Db / p.Dm;
    wc = (p.ws / 2) * (1 - gamma * cos(p.alpha)); % 保持架角速度
    
    FHX = 0; FHY = 0; % 内外圈受力
    FCX = 0; FCY = 0; % 保持架受力
    
    % --- 故障类型 4：保持架断裂引起的剧烈不平衡力 ---
    if p.fault_type == 4
        % 离心力 F = U * w^2
        FCX = FCX + p.U_cage * wc^2 * cos(wc * t);
        FCY = FCY + p.U_cage * wc^2 * sin(wc * t);
    end
    
    for j = 1:p.Nb
        % 1. 计算理论相角
        theta_theory = wc * t + 2*pi*(j-1)/p.Nb;
        theta_curr = mod(theta_theory, 2*pi);
        
        % 2. 故障类型 5：疲劳游荡与碎屑凸起
        bump_debris = 0;
        if p.fault_type == 5
            jitter = p.fatigue_amp * sin(2*pi*p.fatigue_freqs(j)*t + p.fatigue_phases(j));
            theta_curr = mod(theta_theory + jitter, 2*pi);
            
            for k = 1:p.num_debris
                dist = abs(theta_curr - p.debris_angles(k));
                dist = min(dist, 2*pi - dist);
                if dist < p.debris_width
                    bump_debris = bump_debris + p.debris_heights(k) * 0.5 * (1 + cos(pi * dist / p.debris_width));
                end
            end
        end
        
        % 3. 故障类型 1, 2, 3：常规表面凹坑缺陷
        Bj = calculate_defect_Bj_6dof(t, theta_curr, p, gamma, wc);
        
        % 4. 核心：6-DOF 耦合接触变形 delta
        % 保持架的偏心 (xc, yc) 会通过 c_couple 挤压滚动体，改变局部间隙
        cage_effect = p.c_couple * (xc * sin(theta_curr) + yc * cos(theta_curr));
        delta = (xin - xout)*sin(theta_curr) + (yin - yout)*cos(theta_curr) ...
                + cage_effect - p.Cr - Bj + bump_debris;
        
        % 5. 计算接触力及反作用于保持架的力
        if delta > 0
            force = p.Kc * (delta^1.5);
            FHX = FHX + force * sin(theta_curr);
            FHY = FHY + force * cos(theta_curr);
            
            % 滚动体受压时，对偏心保持架产生的反向恢复力 (简化兜孔碰撞力)
            FCX = FCX - p.c_couple * force * sin(theta_curr);
            FCY = FCY - p.c_couple * force * cos(theta_curr);
        end
    end
    
    % --- 运动微分方程组 (12个方程) ---
    % 内圈 (2-DOF)
    ax_in = (-FHX - p.cin*vxin - p.kin*xin) / p.min;
    ay_in = (-FHY - p.cin*vyin - p.kin*yin + p.min*p.g) / p.min; 
    
    % 外圈 (2-DOF)
    ax_out = (FHX - p.cout*vxout - p.kout*xout) / p.mout;
    ay_out = (FHY - p.cout*vyout - p.kout*yout + p.mout*p.g) / p.mout;
    
    % 保持架 (2-DOF)
    ax_c = (FCX - p.cc*vxc - p.kc*xc) / p.mc;
    ay_c = (FCY - p.cc*vyc - p.kc*yc + p.mc*p.g) / p.mc;
    
    dydt = [vxin; ax_in; vyin; ay_in; vxout; ax_out; vyout; ay_out; vxc; ax_c; vyc; ay_c];
end

%% 第五部分：缺陷几何深度计算 (仅用于工况 1~3)
function Bj = calculate_defect_Bj_6dof(t, theta_ball, p, gamma, wc)
    Bj = 0;
    if p.fault_type == 0 || p.fault_type == 4 || p.fault_type == 5, return; end
    
    phi_o = (p.Dm + p.Db)/2; phi_i = (p.Dm - p.Db)/2; phi_b = p.Db/2;          
    sagitta = phi_b - sqrt(max(0, phi_b^2 - (p.defect_width/2)^2));
    C_depth = min(sagitta, p.defect_depth);
    
    switch p.fault_type
        case 1 % 外圈
            if min(abs(theta_ball), 2*pi-abs(theta_ball)) < asin(p.defect_width/(2*phi_o)), Bj = C_depth; end
        case 2 % 内圈
            rel_angle = mod(p.ws*t - theta_ball, 2*pi);
            if rel_angle > pi, rel_angle = rel_angle - 2*pi; end
            if abs(rel_angle) < asin(p.defect_width/(2*phi_i)), Bj = C_depth; end
        case 3 % 滚动体
            wb_spin = (wc + (((1-gamma)*p.ws - wc) / gamma))/2; 
            spin_angle = mod(wb_spin * t, pi);
            if spin_angle > pi/2, spin_angle = spin_angle - pi; end
            if abs(spin_angle) < asin(p.defect_width/(2*phi_b)), Bj = C_depth; end
    end
end