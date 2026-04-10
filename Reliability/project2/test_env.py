import numpy as np
from environment import MissionAbortEnv  # 确保你刚才的代码保存为 environment.py

def test_random_agent(episodes=10):
    # 实例化环境
    env = MissionAbortEnv()
    
    for ep in range(episodes):
        print(f"\n{'='*20} Episode {ep+1} 开始 {'='*20}")
        state, info = env.reset()
        total_reward = 0
        terminated = False
        step_count = 0
        
        while not terminated:
            step_count += 1
            
            # --- 动作掩码 (Action Mask) 逻辑 ---
            # 如果 info 提示载荷已经损坏，智能体失去选择权，必须终止任务
            if info.get('payload_failed', False):
                action = 1  # 强制终止任务 (DA)
                print(f"  [Step {step_count}] ⚠️ 载荷已失效，触发 Action Mask，强制执行 DA")
            else:
                action = env.action_space.sample()  # 随机选择 0 (继续) 或 1 (终止)
            
            action_name = "DC (继续任务)" if action == 0 else "DA (终止并救援)"
            
            # 打印当前状态
            # state 格式: [X_E11 (载荷), X_E21 (运输1), X_E22 (运输2), n (时间)]
            print(f"  [Step {step_count}] 当前状态: 载荷退化={state[0]:.2f}, 运输1退化={state[1]:.2f}, 运输2退化={state[2]:.2f}, 任务时间={state[3]:.0f}")
            print(f"            执行动作: {action_name}")
            
            # 与环境交互
            next_state, reward, terminated, truncated, info = env.step(action)
            
            print(f"            获得奖励: {reward}")
            total_reward += reward
            state = next_state
            
        print(f"--- Episode {ep+1} 结束! 总存活步数: {step_count}, 总累积奖励: {total_reward} ---")

if __name__ == "__main__":
    # 设置 numpy 打印格式，方便阅读
    np.set_printoptions(precision=2, suppress=True)
    test_random_agent()