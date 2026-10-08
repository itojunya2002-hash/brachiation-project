import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from custom_numpy_env import NumpyAcrobotEnv

if __name__ == "main":
    env = NumpyAcrobotEnv(render_mode=None)
    env = TimeLimit(env, max_episode_steps=300)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    print("Training Start (Public Mock)...")
    model.learn(total_timesteps=50_000, progress_bar=True)
    
    model.save("./models/ppo_brachiation_sample")
    print("Model saved.")
    env.close()