# enjoy_elastic.py

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from custom_numpy_env_elastic import NumpyAcrobotEnv

if __name__ == "__main__":
    env = NumpyAcrobotEnv(render_mode="human")
    env = TimeLimit(env, max_episode_steps=600)
    
    try:
        model = PPO.load("./models/ppo_hybrid_branch")
    except FileNotFoundError:
        print("モデルが見つかりません。")
        exit()

    obs, info = env.reset()
    for episode in range(3):
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                done = True
                obs, info = env.reset()
    env.close()