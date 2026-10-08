# run_simulation.py
import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from hardware_consideration.swingup_framework_sample import CurriculumSwingUpEnv
from stable_baselines3 import SAC

def run_evaluation():
    print("🎬 学習済み（または初期）モデルによる挙動のシミュレーション評価を開始します...")
    
    # ステージ5（最大制約・不感帯・ドメインランダマイゼーション有効）で評価環境を構築
    env = CurriculumSwingUpEnv(max_steps=250)
    env.set_curriculum_stage(4) # 最大のハードウェア制約ステージ
    
    # 簡易的なモデルのインスタンス化（※学習済みモデルがある場合は SAC.load("path") に置き換え可能）
    # ここではランダム方策またはダミー動作確認として実行
    obs, _ = env.reset(seed=42)
    
    time_steps = []
    angles = []
    torques = []
    
    for t in range(250):
        # ランダムアクション（または model.predict(obs)）で動作テスト
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        
        time_steps.append(t * 0.01 * 4) # 4ステップスキップを考慮
        angles.append(env.data.qpos[0])
        torques.append(info["actual_torque"])
        
        if truncated:
            break
            
    print(f"✅ シミュレーション終了: 総ステップ数 {len(time_steps)}")
    
    # 簡易プロットで挙動（トルクフィルタや不感帯の効果）を確認
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, angles, color='#003366', lw=2)
    plt.title("Link Angle Trajectory")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.grid(True, linestyle=":")
    
    plt.subplot(1, 2, 2)
    plt.plot(time_steps, torques, color='#008000', lw=1.5)
    plt.title("Filtered Torque with Deadband")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [N·m]")
    plt.grid(True, linestyle=":")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_evaluation()