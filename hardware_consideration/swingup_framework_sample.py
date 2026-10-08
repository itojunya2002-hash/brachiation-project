import os
import warnings
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ============================================================
# Sim-to-Real Consideration Framework (Public Sample)
# ============================================================
class CurriculumSwingUpEnv(gym.Env):
    """
    実機（Sim-to-Real）への転移を見据えた1リンク振子の制御環境。
    モータのトルク応答遅れ、起動時のソフトスタート、および不感帯（Deadband）を模倣し、
    ハードウェア特性に対するロバスト性を高める設計を検証するためのテンプレート。
    """
    def __init__(self, max_steps=250):
        super().__init__()
        self.model_xml = """
        <mujoco model="pendulum_hardware_sample">
            <option timestep="0.01" gravity="0 0 -9.81"/>
            <worldbody>
                <body name="base" pos="0 0 0">
                    <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.005" frictionloss="0.005"/>
                    <body name="link" pos="0 0 0">
                        <inertial pos="0 0 0.09" mass="0.05" diaginertia="0.000135 0.000135 0.000005"/>
                        <geom name="rod" type="capsule" fromto="0 0 0  0 0 0.18" size="0.01" density="1000"/>
                        <body name="tip_weight" pos="0 0 0.18">
                            <inertial pos="0 0 0" mass="0.20" diaginertia="0.00001 0.00001 0.00001"/>
                            <geom name="weight_geom" type="sphere" size="0.02" rgba="0.8 0.2 0.2 1"/>
                        </body>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor name="motor" joint="hinge" ctrlrange="-0.28 0.28" gear="1"/>
            </actuator>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.max_steps = max_steps
        self.step_count = 0
        self.last_torque = 0.0

        self.max_torque_limit = 0.25
        self.dr_level = 0
        self.enable_deadband = False
        self.current_alpha = 0.30

    def set_curriculum_stage(self, stage_num):
        """カリキュラム学習のステージ設定（ハードウェア制約の段階的追加）"""
        if stage_num == 1:
            self.max_torque_limit = 0.25; self.dr_level = 0; self.enable_deadband = False
        elif stage_num == 2:
            self.max_torque_limit = 0.20; self.dr_level = 0; self.enable_deadband = False
        elif stage_num == 3:
            self.max_torque_limit = 0.20; self.dr_level = 1; self.enable_deadband = False
        elif stage_num == 4:
            self.max_torque_limit = 0.20; self.dr_level = 2; self.enable_deadband = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_torque = 0.0

        # ドメインランダマイゼーション（実機の個体差・摩擦変動への適応）
        if self.dr_level == 0:
            self.current_alpha = 0.30
            init_theta, init_vel = np.pi, 0.0
        elif self.dr_level == 1:
            self.current_alpha = np.random.uniform(0.25, 0.35)
            init_theta = np.pi + np.random.uniform(-0.04, 0.04)
            init_vel = np.random.uniform(-0.05, 0.05)
        else:
            self.current_alpha = np.random.uniform(0.20, 0.35)
            init_theta = np.pi + np.random.uniform(-0.08, 0.08)
            init_vel = np.random.uniform(-0.1, 0.1)

        self.data.qpos[0] = init_theta
        self.data.qvel[0] = init_vel
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # 1. アクチュエータのトルク制限と一次遅れフィルタ（応答遅れの模倣）
        raw_torque = np.clip(action[0], -1.0, 1.0) * self.max_torque_limit
        torque = (1.0 - self.current_alpha) * self.last_torque + self.current_alpha * raw_torque

        # 2. 起動時のソフトスタート（急激なトルク跳ね上がり防止）
        if self.step_count <= 15:
            torque = torque * (self.step_count / 15.0)

        # 3. モータドライバ等の不感帯（Deadband）の考慮
        if self.enable_deadband and np.abs(torque) < 0.005:
            applied_torque = 0.0
        else:
            applied_torque = torque

        self.last_torque = applied_torque
        self.data.ctrl[0] = applied_torque

        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

        theta = self.data.qpos[0]
        theta_norm = ((theta + np.pi) % (2.0 * np.pi)) - np.pi
        theta_dot = self.data.qvel[0]

        # ========================================================
        # [PROPRIETARY] 報酬関数のコアロジック
        # ※ 研究機密（係数やハイパーパラメータの詳細）はプレースホルダー化
        # ========================================================
        WEIGHT_BASE = 1.0  # [Proprietary Parameter]
        reward = WEIGHT_BASE * np.cos(theta_norm)

        if np.abs(theta_norm) < 0.5:
            # 目標近傍での収束項およびペナルティ項（係数は非公開）
            reward += 10.0  # [Proprietary Parameter]
            reward -= 1.0 * (theta_norm ** 2)  # [Proprietary Parameter]
            reward -= 0.1 * (theta_dot ** 2)   # [Proprietary Parameter]
            reward -= 0.01 * (applied_torque ** 2) # [Proprietary Parameter]

        info = {"actual_torque": float(applied_torque)}
        return self._get_obs(), float(reward), False, self.step_count >= self.max_steps, info

    def _get_obs(self):
        theta = self.data.qpos[0]
        return np.array([np.cos(theta), np.sin(theta), self.data.qvel[0], self.last_torque], dtype=np.float32)


# ============================================================
# Pipeline Execution Template
# ============================================================
if __name__ == '__main__':
    print("Sim-to-Real 検討用フレームワークの動作テストを開始")
    
    def make_env():
        env = CurriculumSwingUpEnv()
        env.set_curriculum_stage(1)
        return env

    vec_env = SubprocVecEnv([make_env for _ in range(2)])
    model = SAC("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=2000) # デモ用短縮
    print("✅ パイプラインの検証完了（研究のコアロジックはプレースホルダー化されています）")