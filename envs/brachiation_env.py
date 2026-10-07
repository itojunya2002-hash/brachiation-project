import pathlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle

try:
    import mujoco
    from mujoco import mj_name2id, mj_step, mj_resetData
except ImportError:
    import sys
    print("MuJoCoライブラリが見つかりません。")
    sys.exit(1)

def wrap_angle(angle):
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

class BrachiationMuJoCoEnv(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    SUCCESS_THRESHOLD = 0.03
    MAX_TORQUE = 3.0
    L1 = 0.3
    L2 = 0.3

    def __init__(self, render_mode=None, lambda_2=1.0):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 0.02
        self.max_episode_steps = 250
        self.current_step = 0
        self.accumulated_torque_squared = 0.0
        self.lambda_2 = lambda_2

        # ターゲット位置の定義
        self.initial_theta1, self.initial_theta2 = -np.pi/4.0, -np.pi/2.0
        self.target_theta1, self.target_theta2 = np.pi/4.0, np.pi/2.0
        t1t, t2t = self.target_theta1, self.target_theta2
        self.x_target = self.L1 * np.sin(t1t) + self.L2 * np.sin(t1t + t2t)
        self.y_target = -self.L1 * np.cos(t1t) - self.L2 * np.cos(t1t + t2t)
        self.target_pos = np.array([self.x_target, self.y_target])

        # モデルの自動生成
        BR_DIR = pathlib.Path('brachiation')
        BR_DIR.mkdir(exist_ok=True)
        XML_PATH = BR_DIR / 'brachiation_xy_2d.xml'

        with open(XML_PATH, 'w') as f:
            f.write(f"""
            <mujoco model="Brachiation">
                <compiler angle="radian" inertiafromgeom="true"/>
                <option timestep="0.005" gravity="0 0 -9.81"/>
                <worldbody>
                    <body name="link1" pos="0 0 0">
                        <joint name="joint1" pos="0 0 0" axis="0 1 0" range="-3.14 3.14"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -{self.L1}" size="0.025" rgba="0 0 1 1" mass="0.3"/>
                        <body name="link2" pos="0 0 -{self.L1}">
                            <joint name="joint2" pos="0 0 0" axis="0 1 0" range="-2.8 2.8" damping="0.05"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -{self.L2}" size="0.025" rgba="1 0 0 1" mass="0.3"/>
                            <site name="tip" pos="0 0 -{self.L2}" size="0.04"/>
                        </body>
                    </body>
                </worldbody>
                <actuator>
                    <motor name="torque_actuator" joint="joint2" gear="{self.MAX_TORQUE}"/>
                </actuator>
            </mujoco>
            """)

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(self.dt / self.model.opt.timestep) or 1
        self.joint_ids = [mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'joint1'),
                          mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'joint2')]
        self.actuator_id = mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'torque_actuator')

        high = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        EzPickle.__init__(self)

    def _get_pos(self):
        t1, t2 = self.data.qpos[self.joint_ids[0]], self.data.qpos[self.joint_ids[1]]
        x = self.L1 * np.sin(t1) + self.L2 * np.sin(t1 + t2)
        y = -self.L1 * np.cos(t1) - self.L2 * np.cos(t1 + t2)
        return np.array([x, y])

    def _get_obs(self):
        t1, t2 = wrap_angle(self.data.qpos[self.joint_ids[0]]), wrap_angle(self.data.qpos[self.joint_ids[1]])
        d1, d2 = np.clip(self.data.qvel[self.joint_ids[0]], -10, 10), np.clip(self.data.qvel[self.joint_ids[1]], -10, 10)
        return np.array([np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2),
                         d1, d2, self.x_target, self.y_target], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.accumulated_torque_squared = 0.0
        mj_resetData(self.model, self.data)
        self.data.qpos[self.joint_ids[0]] = self.initial_theta1
        self.data.qpos[self.joint_ids[1]] = self.initial_theta2
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        torque_normalized = np.clip(action[0], -1.0, 1.0)
        torque = torque_normalized * self.MAX_TORQUE
        self.data.ctrl[self.actuator_id] = torque_normalized
        for _ in range(self.frame_skip): mj_step(self.model, self.data)

        obs = self._get_obs()
        tip_pos = self._get_pos()
        dist_pos = np.linalg.norm(tip_pos - self.target_pos)
        self.accumulated_torque_squared += (torque ** 2)

        # 報酬設計の抽象化例
        reward = 10.0 * np.exp(-10.0 * dist_pos) 
        
        terminated = False
        info = {'current_torque': torque, 'relative_distance': dist_pos}
        if dist_pos < self.SUCCESS_THRESHOLD:
            reward += 100.0
            terminated = True

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def close(self):
        pass