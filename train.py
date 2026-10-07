import os
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from envs.brachiation_env import BrachiationMuJoCoEnv

if __name__ == '__main__':
    LOG_DIR = "./ppo_log/"
    MODEL_PATH = "ppo_brachiation.zip"
    
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)

    HYPERPARAMS = {
        "n_steps": 2048, "batch_size": 512, "gamma": 0.99,
        "learning_rate": 1.5e-4, "ent_coef": 0.001, "n_epochs": 15
    }

    print("学習開始...")
    set_random_seed(42)
    vec_env = make_vec_env(BrachiationMuJoCoEnv, n_envs=4, monitor_dir=LOG_DIR)
    
    model = PPO("MlpPolicy", vec_env, verbose=1, **HYPERPARAMS)
    model.learn(total_timesteps=100_000) # デモ用短縮
    model.save(MODEL_PATH)
    vec_env.close()
    print(f"モデルを保存しました: {MODEL_PATH}")