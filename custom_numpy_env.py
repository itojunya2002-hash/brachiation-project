import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from pendulum_dynamics import PendulumDynamics, FlightDynamics

class NumpyAcrobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, target_angles=None, initial_angles=None, render_mode=None):
        self.render_mode = render_mode
        self.target_pos = np.array([1.4, 0.0], dtype=np.float32) 
        
        self.dynamics = PendulumDynamics() 
        self.flight_dynamics = FlightDynamics() 
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        high = np.inf * np.ones(13, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = None 
        self.is_attached = True
        self.caught_target = False 
        self.steps_on_branch = 0 
        
        self.fig = None
        self.ax = None

    def _get_obs(self):
        th1, th2, v1, v2, x, y, vx, vy = self.state
        if self.caught_target:
            rel_target = np.array([0.0, 0.0], dtype=np.float32)
        else:
            rel_target = self.target_pos - np.array([x, y])
        obs = np.array([
            np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2), v1, v2,
            x, y, vx, vy,
            rel_target[0], rel_target[1],
            1.0 if self.is_attached else -1.0
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        init_th1 = -np.pi/3 
        init_th2 = -np.pi/3
        
        self.state = np.array([init_th1, init_th2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.is_attached = True
        self.caught_target = False
        self.steps_on_branch = 0 
        return self._get_obs(), {}

    def step(self, action):
        ai_torque = action[0]
        grip_cmd = action[1]
        
        th1, th2, dth1, dth2, x, y, vx, vy = self.state
        reward = 0.0
        terminated = False
        
        # 汎用的なペナルティ項
        reward -= 0.001 * (ai_torque ** 2)
        
        if self.is_attached:
            if self.caught_target:
                zero_action = np.array([0.0, 1.0]) 
                next_angles = self.dynamics(self.state[:4], zero_action)
                next_angles[2] *= 0.95 
                next_angles[3] *= 0.95 
                self.state = np.concatenate([next_angles, self.target_pos, [0, 0]])
                
                if (next_angles[2]**2 + next_angles[3]**2) < 0.01:
                    terminated = True
                    reward += 10.0 
            else:
                self.steps_on_branch += 1
                reward -= 0.01 
                
                if self.steps_on_branch > 250:
                    terminated = True
                    reward -= 1.0 
                
                if grip_cmd <= 0: 
                    self.is_attached = False
                    self.state = self.flight_dynamics(self.state, action)
                    reward += 1.0 
                else:
                    next_angles = self.dynamics(self.state[:4], action)
                    self.state = np.concatenate([next_angles, [0, 0, 0, 0]])
        else:
            self.state = self.flight_dynamics(self.state, action)
            current_tip_pos = self.dynamics.get_tip_position(self.state[:2], self.state[4:6])
            dist_to_target = np.linalg.norm(current_tip_pos - self.target_pos)
            
            reward += (1.0 - dist_to_target) * 0.1
            
            if dist_to_target < 0.05 and grip_cmd > 0:
                self.is_attached = True
                self.caught_target = True
                reward += 10.0 
                
                old_th1, old_th2 = self.state[0], self.state[1]
                new_th1 = self.dynamics._normalize_angle(old_th1 + old_th2 + np.pi)
                new_th2 = self.dynamics._normalize_angle(-old_th2)
                
                self.state = np.array([new_th1, new_th2, 0.0, 0.0, 
                                     self.target_pos[0], self.target_pos[1], 0.0, 0.0], dtype=np.float32)

            if self.state[5] < -2.0:
                terminated = True
                reward -= 1.0
        
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human": return
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.clear()
        s = self.state
        base_pos = s[4:6]
        l1, l2 = self.dynamics.l1, self.dynamics.l2
        p1 = base_pos + np.array([l1 * np.sin(s[0]), -l1 * np.cos(s[0])])
        p2 = p1 + np.array([l2 * np.sin(s[0] + s[1]), -l2 * np.cos(s[0] + s[1])])
        self.ax.plot([base_pos[0], p1[0]], [base_pos[1], p1[1]], color='black', lw=5)
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=5)
        self.ax.set_xlim(-2.0, 3.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.fig.canvas.draw()
        plt.pause(0.001)
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None