# analyze_physics_elastic.py

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from custom_numpy_env_elastic import NumpyAcrobotEnv

def analyze_episode():
    env = NumpyAcrobotEnv(render_mode=None) 
    try:
        model = PPO.load("./models/ppo_hybrid_branch")
    except FileNotFoundError:
        return

    obs, info = env.reset()
    done = False
    time_steps, velocities_1, torques = [], [], []
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        s = env.state
        
        time_steps.append(step_count * 0.02)
        velocities_1.append(s[4] if len(s) == 6 else s[4])
        torques.append(action[0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        if step_count > 300: break

    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, torques, label='Torque', color='green')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque')
    plt.grid(True)
    plt.savefig("analysis_result_hybrid.pdf", format="pdf")
    plt.show()

if __name__ == "__main__":
    analyze_episode()