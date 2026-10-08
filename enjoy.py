import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from custom_numpy_env import NumpyAcrobotEnv

if __name__ == "__main__":
    env = NumpyAcrobotEnv(render_mode="human")
    env = TimeLimit(env, max_episode_steps=300)
    
    try:
        model = PPO.load("./models/ppo_brachiation_sample")
    except FileNotFoundError:
        print("モデルが見つかりません。pendulum_leap_train.pyを実行してください。")
        exit()

    obs, info = env.reset()
    
    for episode in range(5):
        done = False
        total_reward = 0
        print(f"--- Episode {episode} Start ---")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            
            if terminated or truncated:
                done = True
                print(f"Episode finished. Reward: {total_reward:.2f}")
                obs, info = env.reset()
    
    env.close()