clc; clear; close all;

%% 第一部分：参数定义与系统初始化
% --- 1. 轴承几何与系统参数 (SKF-6203-RS) ---
param.Db = 6.746e-3;    
param.Dm = 28.5e-3;     
param.Nb = 8;           
param.alpha = 0;        
param.Cr = 2e-6;        
param.min = 50;        
param.mout = 5;       
param.kin = 7.42e7;        
param.kout = 1.51e7;       
param.cin = 1376.8;        
param.cout = 2210.7;       
param.Kc = 8.753e9;         
param.g = 9.8;                  
param.rpm = 1797;               
param.ws = param.rpm * 2*pi/60; 
param.fs = 12000;               
param.T = 1.0;                  

% --- 2. 新增：保持架疲劳与碎屑干扰参数 ---
param.fault_type = 5; % 5 = 疲劳与碎屑干扰复合状态

% A. 保持架疲劳参数 (兜孔松动导致的相位游荡)
param.fatigue_amp = 0.5 * (pi/180); % 滚动体相位最大游荡幅度 (0.5度)
% 为每个滚动体预设随机的相位初始偏移和不同的游荡频率，保证连续性以便ODE求解
rng(42); % 固定随机种子以便复现
param.fatigue_phases = rand(param.Nb, 1) * 2 * pi;
param.fatigue_freqs = 5 + 10 * rand(param.Nb, 1); % 5~15Hz的低频游荡

% B. 碎屑干扰参数 (模拟脱落的金属颗粒附着在外圈滚道)
param.num_debris = 15; % 滚道上存在的碎屑数量
param.debris_angles = rand(param.num_debris, 1) * 2 * pi; % 碎屑随机分布的角度
% 碎屑高度通常在 5~30 微米之间
param.debris_heights = (5 + 25 * rand(param.num_debris, 1)) * 1e-6; 
% 碎屑的影响宽度 (角度范围)，用平滑过渡函数模拟碾压过程
param.debris_width = 1.0 * (pi/180); 

%% 第二部分：主仿真循环
fprintf('正在仿真工况: 保持架疲劳与碎屑干扰复合状态...\n');
tspan = 0:1/param.fs:param.T;   
y0 = zeros(8, 1); 
options = odeset('RelTol', 1e-5, 'AbsTol', 1e-6);

% 调用 ODE45 求解
tic;
[t, Y] = ode45(@(t, y) bearing_dynamics_debris(t, y, param), tspan, y0, options);
toc;

% 重新计算外圈Y向加速度
acc_out_y = zeros(length(t), 1);
for i = 1:length(t)
    dydt = bearing_dynamics_debris(t(i), Y(i,:)', param);
    acc_out_y(i) = dydt(8); 
end
disp('仿真完成。正在绘图...');

%% 第三部分：结果可视化
figure('Color', 'w', 'Position', [100, 100, 1000, 700]);

% --- 1. 时域波形 ---
subplot(2,1,1);
cut_idx = round(length(t)*0.1); % 去掉前10%的瞬态响应
valid_t = t(cut_idx:end);
valid_sig = acc_out_y(cut_idx:end);

% 添加一点高斯白噪声模拟传感器底噪
snr_db = 20; 
valid_sig = awgn(valid_sig, snr_db, 'measured');

plot(valid_t, valid_sig, 'b');
title('时域波形', 'FontSize', 12);
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
title('包络谱', 'FontSize', 12);
xlabel('频率 (Hz)'); ylabel('幅值');
xlim([0, 500]); grid on; 
hold on;

% 标记理论特征频率以供对比
gamma = param.Db / param.Dm;
fr = param.ws / (2*pi); 
BPFO = (param.Nb/2) * fr * (1 - gamma*cos(param.alpha));
FTF  = (fr/2) * (1 - gamma*cos(param.alpha)); 
xline(BPFO, '--k', 'Label', 'BPFO', 'LabelVerticalAlignment', 'top','LineWidth', 1.5);
xline(FTF, '-.m', 'Label', 'FTF (fc)', 'LineWidth', 1.5);
legend('仿真信号包络', '外圈通过频率', '保持架频率');

%% 第四部分：动力学模型函数 (含疲劳与碎屑)
function dydt = bearing_dynamics_debris(t, y, p)
    xin = y(1); vxin = y(2);
    yin = y(3); vyin = y(4);
    xout = y(5); vxout = y(6);
    yout = y(7); vyout = y(8);
    
    gamma = p.Db / p.Dm;
    wc = (p.ws / 2) * (1 - gamma * cos(p.alpha)); 
    
    FHX = 0; FHY = 0;
    
    for j = 1:p.Nb
        % 1. 理论位置
        theta_theory = wc * t + 2*pi*(j-1)/p.Nb;
        
        % 2. 引入保持架疲劳造成的相位抖动 (连续平滑函数，防止ODE求解失败)
        jitter = p.fatigue_amp * sin(2 * pi * p.fatigue_freqs(j) * t + p.fatigue_phases(j));
        theta_curr = mod(theta_theory + jitter, 2*pi);
        
        % 3. 计算碎屑造成的附加变形 (Bump)
        bump_debris = 0;
        for k = 1:p.num_debris
            % 计算滚动体与碎屑的最短角距离
            dist = abs(theta_curr - p.debris_angles(k));
            dist = min(dist, 2*pi - dist);
            
            % 如果滚过碎屑，产生一个平滑的隆起 (Hanning Window 形状)
            if dist < p.debris_width
                bump_debris = bump_debris + p.debris_heights(k) * 0.5 * (1 + cos(pi * dist / p.debris_width));
            end
        end
        
        % 4. 计算总变形 (注意：碎屑是凸起，所以是 +bump_debris)
        delta = (xin - xout)*sin(theta_curr) + (yin - yout)*cos(theta_curr) - p.Cr + bump_debris;
        
        % 5. 接触力
        if delta > 0
            force = p.Kc * (delta^1.5);
            FHX = FHX + force * sin(theta_curr);
            FHY = FHY + force * cos(theta_curr);
        end
    end
    
    % 运动微分方程
    ax_in = (-FHX - p.cin*vxin - p.kin*xin) / p.min;
    ay_in = (-FHY - p.cin*vyin - p.kin*yin + p.min*p.g) / p.min; 
    ax_out = (FHX - p.cout*vxout - p.kout*xout) / p.mout;
    ay_out = (FHY - p.cout*vyout - p.kout*yout + p.mout*p.g) / p.mout;
    
    dydt = [vxin; ax_in; vyin; ay_in; vxout; ax_out; vyout; ay_out];
end