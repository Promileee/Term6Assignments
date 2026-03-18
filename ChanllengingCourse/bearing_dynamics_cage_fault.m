clc; clear; close all;
%注意：该模型生成的是一个"理想化的刚体动力学响应"，对于保持架断裂的仿真信号属于"低保真"信号（Low Fidelity）
% 不过保持架断裂的仿真信号包含了正确的频率成分（标签）。
% 可用它做包络谱分析，或者作为深度学习中一种极其简化的"强标签"数据。
% 或者采用GAN等生产式网络对低保真信号进行修补，提高其保真程度

%% 参考文献：
% 论文1：ZHANG Z, ZI Y, ZHANG M, 等. Deep Unsupervised Subdomain Adaptation Network for Intelligent Fault Diagnosis: From Simulated Domain to Physical Domain[J/OL]. IEEE Transactions on Instrumentation and Measurement, 2025, 74: 1-16. DOI:10.1109/TIM.2025.3604989.
% 论文2：FAN C, WANG P, ZHANG Y, 等. Digital Twin Assisted Degradation Assessment of Bearing Cage Performance[J/OL]. IEEE Transactions on Industrial Informatics, 2025, 21(7): 5171-5181. DOI:10.1109/TII.2025.3552655.

%% 第一部分：参数定义与系统初始化
% --- 1. 轴承几何参数 (以 SKF-6203-RS 为例) ---
param.Db = 6.746e-3;    % 滚动体直径 (m)
param.Dm = 28.5e-3;     % 节圆直径 (m)
param.Nb = 8;           % 滚动体个数
param.alpha = 0;        % 接触角 (rad)
param.Cr = 2e-6;        % 径向游隙 (m)

% --- 2. 系统动力学参数 ---
param.min = 50;        % 内圈及轴的等效质量 (kg)
param.mout = 5;        % 外圈及轴承座的等效质量 (kg)
param.kin = 7.42e7;    % 内圈支撑刚度 (N/m)
param.kout = 1.51e7;   % 外圈支撑刚度 (N/m)
param.cin = 1376.8;    % 内圈阻尼 (N.s/m)
param.cout = 2210.7;   % 外圈阻尼 (N.s/m)
param.Kc = 8.753e9;    % 赫兹接触刚度系数 (N/m^1.5)

% --- 3. 运行工况参数 ---
param.g = 9.8;                  % 重力加速度
param.rpm = 1797;               % 转速 (r/min)
param.ws = param.rpm * 2*pi/60; % 轴旋转角频率 (rad/s)
param.fs = 12000;               % 采样频率 (Hz)
param.T = 1.0;                  % 仿真时长 (s) (建议加长一点看低频)

% --- 4. 故障参数设置 ---
% 故障类型: 0=正常, 1=外圈(OF), 2=内圈(IF), 3=滚动体(BF), 4=保持架断裂(Cage)
param.fault_type = 4; 

param.defect_width = 3.6e-4;   % 用于类型 1-3
param.defect_depth = 1e-3;     % 用于类型 1-3

% 保持架故障专用参数
% Fan et al. 论文2中 theta_f 取值范围约为 0~4度。这里取 2度 模拟明显断裂
param.theta_f_deg = 2.0;       % 故障滚动体的偏转角 (度)
param.faulty_ball_idx = 1;     % 指定第几个滚动体对应的兜孔断裂 (例如第1个)

%% 第二部分：主仿真循环
fault_names = {'正常 (Normal)', '外圈故障 (Outer Race)', '内圈故障 (Inner Race)', '滚动体故障 (Ball)', '保持架断裂 (Cage Fracture)'};
fprintf('正在仿真工况: %s...\n', fault_names{param.fault_type + 1});

tspan = 0:1/param.fs:param.T;   
y0 = zeros(8, 1); 
options = odeset('RelTol', 1e-5, 'AbsTol', 1e-6);

% 调用 ODE45 求解
[t, Y] = ode45(@(t, y) bearing_dynamics_4dof(t, y, param), tspan, y0, options);

% ---重新计算加速度 ---
acc_out_y = zeros(length(t), 1);
for i = 1:length(t)
    dydt = bearing_dynamics_4dof(t(i), Y(i,:)', param);
    acc_out_y(i) = dydt(8); 
end
disp('仿真完成。正在绘图...');

%% 第三部分：结果可视化
figure('Color', 'w', 'Position', [100, 100, 1000, 700]);

% --- 1. 时域波形 ---
subplot(2,1,1);
cut_idx = round(length(t)*0.2); % 去掉前20%的瞬态响应
valid_t = t(cut_idx:end);
valid_sig = acc_out_y(cut_idx:end);
plot(valid_t, valid_sig, 'b');
title(['时域波形 - ', fault_names{param.fault_type + 1}], 'FontSize', 12);
xlabel('时间 (s)'); ylabel('加速度 (m/s^2)');
grid on; 
xlim([valid_t(1), valid_t(1)+0.5]); 

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
xlim([0, 400]); grid on; 
hold on;

% --- 3. 标记特征频率 ---
gamma = param.Db / param.Dm;
fr = param.ws / (2*pi); 
BPFO = (param.Nb/2) * fr * (1 - gamma*cos(param.alpha));
BPFI = (param.Nb/2) * fr * (1 + gamma*cos(param.alpha));
BSF  = fr * (param.Dm / (2*param.Db)) * (1 - (gamma * cos(param.alpha))^2);
FTF  = (fr/2) * (1 - gamma*cos(param.alpha)); % 计算保持架频率 (fc)

% 根据故障类型画线
if param.fault_type == 1
    xline(BPFO, '--k', 'Label', 'BPFO','LineWidth', 1.5); xline(2*BPFO, ':k','Label', '2BPFO','LineWidth', 1.5);
    
elseif param.fault_type == 2
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
elseif param.fault_type == 3
    xline(2*BSF, '--k', 'Label', '2xBSF', 'LabelVerticalAlignment', 'top','LineWidth', 1.5);
    xline(4*BSF, ':k', 'Label', '4xBSF','LineWidth', 1.5);
elseif param.fault_type == 4 % 保持架故障标记
    % 保持架故障通常表现为 FTF (fc) 及其倍频，且因为不平衡可能出现 fr
    xline(FTF, '--m', 'Label', 'FTF (fc)', 'LineWidth', 2);
    xline(2*FTF, '-.m', 'Label', '2fc', 'LineWidth', 1.5);
    xline(3*FTF, ':m', 'Label', '3fc', 'LineWidth', 1.5);
    xline(fr, ':b', 'Label', 'fr'); % 保持架断裂往往伴随不平衡
    legend('包络谱', '保持架频率 fc', '2fc', '3fc', '转频 fr');
end


%% 第四部分：动力学模型函数 
function dydt = bearing_dynamics_4dof(t, y, p)
    % 状态变量解包
    xin = y(1); vxin = y(2);
    yin = y(3); vyin = y(4);
    xout = y(5); vxout = y(6);
    yout = y(7); vyout = y(8);
    
    % --- A. 运动学计算 ---
    gamma = p.Db / p.Dm;
    wc = (p.ws / 2) * (1 - gamma * cos(p.alpha)); 
    
    FHX = 0; FHY = 0;
    
    % --- B. 循环计算滚动体接触力 ---
    for j = 1:p.Nb
        % 
        % 1. 计算理论位置
        theta_theory = wc * t + 2*pi*(j-1)/p.Nb;
        
        % 2. 如果是保持架故障，且当前是“坏掉”的那颗球，加上偏移角 theta_f
        current_theta = theta_theory; % 默认正常
        if p.fault_type == 4 && j == p.faulty_ball_idx
            % Fan et al. 2025: theta_j = theta_theory + theta_f
            theta_f_rad = p.theta_f_deg * (pi/180);
            jitter = 0.5 * (pi/180) * randn(); 
            current_theta = theta_theory + theta_f_rad + jitter;
        end
        
        % 3. 取模保证在 [0, 2pi]
        theta_curr = mod(current_theta, 2*pi);
        
        % 4. 计算故障变形量 Bj (保持架故障时 Bj 为 0)
        Bj = calculate_defect_Bj(t, theta_curr, p, gamma, wc);
        
        % 5. 计算总接触变形 delta
        % 只有当该滚动体因偏移而更加“挤压”或“松脱”时，受力会改变
        delta = (xin - xout)*sin(theta_curr) + (yin - yout)*cos(theta_curr) - p.Cr - Bj;
        
        % 6. 计算赫兹接触力
        if delta > 0
            force = p.Kc * (delta^1.5);
            FHX = FHX + force * sin(theta_curr);
            FHY = FHY + force * cos(theta_curr);
        end
    end
    
    % --- C. 运动微分方程 ---
    ax_in = (-FHX - p.cin*vxin - p.kin*xin) / p.min;
    ay_in = (-FHY - p.cin*vyin - p.kin*yin + p.min*p.g) / p.min; 
    
    ax_out = (FHX - p.cout*vxout - p.kout*xout) / p.mout;
    ay_out = (FHY - p.cout*vyout - p.kout*yout + p.mout*p.g) / p.mout;
    
    dydt = [vxin; ax_in; vyin; ay_in; vxout; ax_out; vyout; ay_out];
end

%% 第五部分：故障变形计算辅助函数
function Bj = calculate_defect_Bj(t, theta_ball, p, gamma, wc)
    Bj = 0;
    % 如果是保持架故障 (type 4)，不产生表面坑洞变形 Bj
    if p.fault_type == 0 || p.fault_type == 4 
        return; 
    end
    
    phi_o = (p.Dm + p.Db)/2; 
    phi_i = (p.Dm - p.Db)/2; 
    phi_b = p.Db/2;          
    
    sagitta = phi_b - sqrt(max(0, phi_b^2 - (p.defect_width/2)^2));
    Co = min(sagitta, p.defect_depth);
    Ci = min(sagitta, p.defect_depth);
    Cb = min(sagitta, p.defect_depth);

    switch p.fault_type
        case 1 % 外圈
            fault_loc = 0; 
            beta_o = asin(p.defect_width / (2*phi_o)); 
            dist = abs(theta_ball - fault_loc);
            dist = min(dist, 2*pi - dist);
            if dist < beta_o, Bj = Co; end
    
        case 2 % 内圈
            beta_i = asin(p.defect_width / (2*phi_i));
            rel_angle = mod(p.ws*t - theta_ball, 2*pi);
            if rel_angle > pi, rel_angle = rel_angle - 2*pi; end
            if abs(rel_angle) < beta_i, Bj = Ci; end
            
        case 3 % 滚动体
            beta_b = asin(p.defect_width / (2*phi_b));
            wb = ((1-gamma)*p.ws - wc) / gamma; 
            wb_spin = (wc + wb)/2; 
            spin_angle = mod(wb_spin * t, pi);
            if spin_angle > pi/2, spin_angle = spin_angle - pi; end
            if abs(spin_angle) < beta_b, Bj = Cb; end
    end
end