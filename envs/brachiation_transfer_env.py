import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box  # ← これがインポートされているか確認
from gymnasium.utils import EzPickle

try:
    import mujoco
    from mujoco import mj_name2id, mj_step, mj_resetData
except ImportError:
    import sys
    print("MuJoCoが見つかりません。")
    sys.exit(1)

def wrap_angle(angle):
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

class BrachiationTransferEnv(gym.Env, EzPickle):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    L1, L2 = 0.5, 0.5
    M1, M2 = 0.3, 0.3
    SUCCESS_THRESHOLD = 0.03
    STATIONARY_VELOCITY_THRESHOLD = 0.5
    MAX_TORQUE = 3.0

    def __init__(self, render_mode=None, is_elastic=False, lambda_2=1.0):
        super().__init__()
        self.is_elastic = is_elastic
        self.dt = 0.02
        self.max_episode_steps = 250
        self.accumulated_torque_squared = 0.0
        self.lambda_2 = lambda_2

        # ターゲット位置の定義
        t1t, t2t = np.pi/4.0, np.pi/2.0
        self.target_pos = np.array([
            self.L1 * np.sin(t1t) + self.L2 * np.sin(t1t + t2t),
            -self.L1 * np.cos(t1t) - self.L2 * np.cos(t1t + t2t)
        ])

        xml_content = self._generate_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(self.dt / self.model.opt.timestep) or 1

        self.joint_ids = [mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'joint1'),
                          mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'joint2')]
        self.actuator_id = mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'torque_actuator')

        if self.is_elastic:
            self.joint_y_id = mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'joint_y')

        # 観測空間の次元 (剛体・弾性間の転移学習のため共通化)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        EzPickle.__init__(self)

    def _generate_xml(self):
        joint_y_xml = '<joint name="joint_y" type="slide" axis="0 0 1" stiffness="300" damping="2.0"/>' if self.is_elastic else ''
        return f"""
        <mujoco model="Brachiation">
            <compiler angle="radian" inertiafromgeom="true"/>
            <option timestep="0.005" gravity="0 0 -9.81"/>
            <worldbody>
                <body name="branch" pos="0 0 0">
                    {joint_y_xml}
                    <geom type="sphere" size="0.03" rgba="0.5 0.5 0.5 1" mass="1.0"/>
                    <body name="link1" pos="0 0 0">
                        <joint name="joint1" pos="0 0 0" axis="0 1 0" range="-3.14 3.14"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -{self.L1}" size="0.02" rgba="0 0 1 1" mass="{self.M1}"/>
                        <body name="link2" pos="0 0 -{self.L1}">
                            <joint name="joint2" pos="0 0 0" axis="0 1 0" range="-2.8 2.8" damping="0.05"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -{self.L2}" size="0.02" rgba="1 0 0 1" mass="{self.M2}"/>
                            <site name="tip" pos="0 0 -{self.L2}" size="0.04"/>
                        </body>
                    </body>
                </body>
            </worldbody>
            <actuator><motor name="torque_actuator" joint="joint2" gear="{self.MAX_TORQUE}"/></actuator>
        </mujoco>
        """

    def _get_pos(self):
        y_off = self.data.qpos[self.joint_y_id] if self.is_elastic else 0.0
        t1, t2 = self.data.qpos[self.joint_ids[0]], self.data.qpos[self.joint_ids[1]]
        return np.array([self.L1*np.sin(t1)+self.L2*np.sin(t1+t2), y_off-self.L1*np.cos(t1)-self.L2*np.cos(t1+t2)])

    def _get_obs(self):
        t1, t2 = wrap_angle(self.data.qpos[self.joint_ids[0]]), wrap_angle(self.data.qpos[self.joint_ids[1]])
        d1, d2 = np.clip(self.data.qvel[self.joint_ids[0]], -10, 10), np.clip(self.data.qvel[self.joint_ids[1]], -10, 10)
        y_b = self.data.qpos[self.joint_y_id] if self.is_elastic else 0.0
        dy_b = self.data.qvel[self.joint_y_id] if self.is_elastic else 0.0
        return np.array([np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2),
                         d1, d2, self.target_pos[0], self.target_pos[1], y_b, dy_b], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.accumulated_torque_squared = 0.0
        mj_resetData(self.model, self.data)
        self.data.qpos[self.joint_ids[0]] = -np.pi/4.0
        self.data.qpos[self.joint_ids[1]] = -np.pi/2.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        torque_normalized = np.clip(action[0], -1.0, 1.0)
        self.data.ctrl[self.actuator_id] = torque_normalized
        for _ in range(self.frame_skip): mj_step(self.model, self.data)

        obs = self._get_obs()
        tip_pos = self._get_pos()
        tip_vel = np.linalg.norm(self.data.qvel)
        dist_pos = np.linalg.norm(tip_pos - self.target_pos)
        self.accumulated_torque_squared += (torque_normalized * self.MAX_TORQUE) ** 2

        # 報酬関数のコア部分はプレースホルダー化
        reward = 10.0 * np.exp(-10.0 * dist_pos)

        terminated = False
        if dist_pos < self.SUCCESS_THRESHOLD and tip_vel < self.STATIONARY_VELOCITY_THRESHOLD:
            reward += 100.0
            terminated = True

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass