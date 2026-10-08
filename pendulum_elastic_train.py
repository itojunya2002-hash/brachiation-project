# pendulum_elastic_train.py

import os
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from custom_numpy_env_elastic import NumpyAcrobotEnv

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs_hybrid_branch", exist_ok=True)

    env = NumpyAcrobotEnv(render_mode=None)
    env = TimeLimit(env, max_episode_steps=500)
    
    print("=== Start Training (Elastic Branch Model) ===")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs_hybrid_branch/",
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    # progress_bar=True を削除し、エラーを防いでいます
    model.learn(total_timesteps=100_000)
    
    model.save("./models/ppo_hybrid_branch")
    print("Saved model.")
    env.close()