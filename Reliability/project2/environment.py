import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import norm

class UAVMissionEnv(gym.Env):
    """
    基于深度强化学习的无人机(UAV)运输系统动态任务中止策略环境
    包含一个任务载荷(E11，相机传感器)和两个运输子系统组件(E21, E22，齿轮箱组件)
    """
    def __init__(self):
        super(UAVMissionEnv, self).__init__()
        
        # --- 动作空间与状态空间 [4, 5] ---
        # 动作空间: 0 代表 中止任务 (DA), 1 代表 继续任务 (DC)
        self.action_space = spaces.Discrete(2)
        # 状态空间: [X_11, X_21, X_22, n], 分别为三个组件的退化状态和当前检查步数
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        
        # --- 论文 Section 5.1 数值仿真参数 [6-8] ---
        # 固有退化率 mu (drift) 和 扩散系数 sigma (diffusion)
        self.mu = np.array([0.24, 0.45, 0.52]) 
        self.sigma = np.array([0.30, 0.10, 0.12])
        
        # 随机相关性退化交互矩阵 H2 [9]
        self.H = np.array([
            [0.0,   0.0,   0.0  ],
            [0.01,  0.0,   0.072],
            [0.01,  0.063, 0.0  ]
        ])
        
        # 失效阈值 L1, L2, L3 [8]
        self.L = np.array([7.5, 12.0, 13.0])
        
        # 时间与周期设定 [8]
        self.delta = 1.0        # 检查间隔
        self.T1 = 22.0          # 任务执行阶段时长
        self.T2 = 3.0           # 返回阶段时长
        self.N = int(np.ceil(self.T1 / self.delta)) - 1  # 最大决策步数 (21步)
        
        # 成本与收益参数 [8]
        self.r_m = 1200.0       # 任务完成收益
        self.c_m = 500.0        # 任务失败惩罚
        self.c_f = 1500.0       # 系统失效成本
        self.c_I = 2.0          # 日常检查成本
        
        # 计算考虑了相关性后的综合退化参数 tilde_mu 和 tilde_sigma [10]
        self.tilde_mu = self.mu + np.dot(self.H, self.mu)
        self.tilde_sigma = np.sqrt(self.sigma**2 + np.dot(self.H**2, self.sigma**2))
        
        # 内部状态变量
        self.state = None
        self.task_failed = False

    def reset(self, seed=None, options=None):
        """初始化系统状态"""
        super().reset(seed=seed)
        # 初始时，所有组件退化量均为0，时间步为0
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.task_failed = False
        return self.state, {}

    def _phi(self, t):
        """救援时间函数 [8]"""
        return t if t <= 3.0 else 3.0

    def _ig_cdf(self, mu_tilde, sigma_tilde, t, L, x):
        """
        根据逆高斯(IG)分布计算组件在时间t内的失效概率 (Eq. 5) [2, 11]
        """
        if x >= L:
            return 1.0
        if t <= 0:
            return 0.0
            
        term1 = (mu_tilde * t - L + x) / (sigma_tilde * np.sqrt(t))
        term2 = (-mu_tilde * t + L - x) / (sigma_tilde * np.sqrt(t))
        
        prob = norm.cdf(term1) + np.exp((2 * (L - x) * mu_tilde) / (sigma_tilde**2)) * norm.cdf(term2)
        return np.clip(prob, 0.0, 1.0)

    def _check_system_failure(self, t_interval, x_state):
        """
        计算运输子系统（齿轮箱E21, E22）在指定时间段内是否失效。
        根据论文逻辑，任一运输组件失效即导致系统失效。
        """
        # 齿轮箱组件 E21 对应索引 1
        p_fail_21 = self._ig_cdf(self.tilde_mu[4], self.tilde_sigma[4], t_interval, self.L[4], x_state[4])
        # 齿轮箱组件 E22 对应索引 2
        p_fail_22 = self._ig_cdf(self.tilde_mu[5], self.tilde_sigma[5], t_interval, self.L[5], x_state[5])
        
        # 串联系统：系统存活概率 = 各组件存活概率的乘积
        p_sys_survive = (1.0 - p_fail_21) * (1.0 - p_fail_22)
        p_sys_fail = 1.0 - p_sys_survive
        
        return np.random.rand() < p_sys_fail

    def step(self, action):
        """环境步进与奖励计算"""
        x_11, x_21, x_22, n = self.state
        x_state = np.array([x_11, x_21, x_22])
        t_current = n * self.delta
        
        reward = 0.0
        done = False
        info = {}

        if action == 0:  # DA: 中止任务
            t_rescue = self._phi(t_current)
            
            # 若任务尚未失败，中止任务意味着任务强制失败，扣除任务失败惩罚
            if not self.task_failed:
                reward -= self.c_m
                self.task_failed = True
                
            # 检查系统在救援过程中是否失效
            sys_fail_during_rescue = self._check_system_failure(t_rescue, x_state)
            
            if sys_fail_during_rescue:
                reward -= self.c_f
                info['reason'] = 'System failed during rescue'
            else:
                info['reason'] = 'Successfully rescued'
                
            done = True

        elif action == 1:  # DC: 继续任务
            reward -= self.c_I  # 扣除日常检查成本
            
            # 1. 检查当前步长内运输系统是否失效
            sys_fail = self._check_system_failure(self.delta, x_state)
            
            if sys_fail:
                if not self.task_failed:
                    reward -= self.c_m
                reward -= self.c_f
                done = True
                info['reason'] = 'System crashed during mission'
            else:
                # 2. 检查任务载荷(相机E11)是否失效 (相机对应索引 0)
                p_fail_11 = self._ig_cdf(self.tilde_mu, self.tilde_sigma, self.delta, self.L, x_state)
                payload_fail = np.random.rand() < p_fail_11
                
                if payload_fail and not self.task_failed:
                    reward -= self.c_m
                    self.task_failed = True
                    x_state = self.L  # 仅将相机的状态强制置为失效阈值
                
                # 3. 模拟维纳过程自然退化增量
                delta_x = np.random.normal(
                    loc=self.tilde_mu * self.delta, 
                    scale=self.tilde_sigma * np.sqrt(self.delta)
                )
                x_state = x_state + delta_x
                # 退化量非严格单调，但约束下界为当前值
                x_state = np.maximum(x_state, [x_11, x_21, x_22])
                
                n += 1  # 推进时间步
                self.state = np.array([x_state, x_state[4], x_state[5], n], dtype=np.float32)
                
                # 4. 判断是否到达任务完成时间 T1
                if n >= self.N:
                    # 进入返回阶段
                    sys_fail_return = self._check_system_failure(self.T2, x_state)
                    
                    if not self.task_failed:
                        reward += self.r_m  # 获取任务完成收益
                        
                    if sys_fail_return:
                        reward -= self.c_f
                        info['reason'] = 'Task completed but failed during return'
                    else:
                        info['reason'] = 'Task completed and returned safely'
                        
                    done = True
                    
        return self.state, reward, done, False, info