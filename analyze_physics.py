import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from custom_numpy_env import NumpyAcrobotEnv

def analyze_episode():
    env = NumpyAcrobotEnv(render_mode=None) 
    try:
        model = PPO.load("./models/ppo_brachiation_sample")
    except FileNotFoundError:
        print("エラー: モデルファイルが見つかりません。")
        return

    obs, _ = env.reset()
    done = False
    
    time_steps = []
    velocities_1 = []
    velocities_2 = []
    torques = []
    
    step_count = 0

    print("解析エピソード実行中...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        s = env.state
        
        time_steps.append(step_count * env.dynamics.dt)
        velocities_1.append(s[2])
        velocities_2.append(s[3])
        torques.append(action[0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        step_count += 1
        if step_count > 300:
            break

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    axes[0].plot(time_steps, velocities_1, label=r'$\dot{\theta}_1$', color='blue')
    axes[0].plot(time_steps, velocities_2, label=r'$\dot{\theta}_2$', color='orange')
    axes[0].set_ylabel('Angular Velocity')
    axes[0].grid(True)
    axes[0].legend(loc='upper right')
    
    axes[1].plot(time_steps, torques, label='Torque', color='green')
    axes[1].set_ylabel('Torque')
    axes[1].set_xlabel('Time [s]')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("analysis_result.pdf", format="pdf")
    plt.show()

if __name__ == "__main__":
    analyze_episode()