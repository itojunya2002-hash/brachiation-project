import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from envs.brachiation_transfer_env import BrachiationTransferEnv

if __name__ == '__main__':
    BASE_MODEL = "ppo_rigid_model"
    FINAL_MODEL = "ppo_elastic_model"
    set_random_seed(42)

    HYPERPARAMS = {
        "n_steps": 2048, "batch_size": 512, "gamma": 0.99,
        "learning_rate": 1.5e-4, "ent_coef": 0.001, "n_epochs": 15
    }

    # Phase 1: 剛体モデルでの基礎学習
    print(" Phase 1: 剛体モデルによるベース方策...")
    env_rigid = make_vec_env(BrachiationTransferEnv, n_envs=4, env_kwargs=dict(is_elastic=False))
    model = PPO("MlpPolicy", env_rigid, verbose=1, **HYPERPARAMS)
    model.learn(total_timesteps=50_000) # デモ用
    model.save(BASE_MODEL)
    env_rigid.close()

    # Phase 2: 弾性モデルへの転移学習（Sim-to-Sim Adaptation）
    print("\nPhase 2: 弾性モデルへの転移学習")
    env_elastic = make_vec_env(BrachiationTransferEnv, n_envs=4, env_kwargs=dict(is_elastic=True))
    model = PPO.load(BASE_MODEL, env=env_elastic)
    
    
    model.learning_rate = 5e-5
    model.learn(total_timesteps=30_000) # デモ用
    model.save(FINAL_MODEL)
    env_elastic.close()
    print("\n✅ 転移学習パイプライン完了")