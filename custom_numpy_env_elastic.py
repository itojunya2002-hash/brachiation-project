# custom_numpy_env_elastic.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from pendulum_dynamics_elastic import PendulumDynamics, ElasticPendulumDynamics, FlightDynamics

class NumpyAcrobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, target_angles=None, initial_angles=None, render_mode=None):
        self.render_mode = render_mode
        self.target_pos = np.array([1.6, 0.0], dtype=np.float32) 
        
        self.rigid_dynamics = PendulumDynamics() 
        self.elastic_dynamics = ElasticPendulumDynamics() 
        self.flight_dynamics = FlightDynamics() 
        
        self.elastic_dynamics.y_anchor = 0.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        high = np.inf * np.ones(15, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = None 
        self.is_attached = True
        self.caught_target = False 
        self.steps_on_branch = 0 
        self.last_action = np.zeros(2)
        
        self.fig = None
        self.ax = None

    def _get_obs(self):
        if self.is_attached:
            y_b, th1, th2, dy_b, dth1, dth2 = self.state
            x_base, y_base = (self.target_pos[0], self.target_pos[1]) if self.caught_target else (0.0, y_b)
            vx_base, vy_base = 0.0, dy_b
        else:
            y_b, th1, th2, dy_b, dth1, dth2, x_base, y_base, vx_base, vy_base = self.state

        rel_target = np.array([0.0, 0.0], dtype=np.float32) if self.caught_target else self.target_pos - np.array([x_base, y_base])

        return np.array([
            np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2), dth1, dth2,
            x_base, y_base, vx_base, vy_base,
            rel_target[0], rel_target[1],
            1.0 if self.is_attached else -1.0,
            y_b, dy_b
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        init_th1, init_th2 = -np.pi/3, -np.pi/3
        init_y_b = -0.1
        
        self.state = np.array([init_y_b, init_th1, init_th2, 0.0, 0.0, 0.0], dtype=np.float32)
        self.is_attached = True
        self.caught_target = False
        self.steps_on_branch = 0 
        self.last_action = np.zeros(2)
        return self._get_obs(), {}

    def step(self, action):
        ai_torque, grip_cmd = action[0], action[1]
        reward = 0.0
        terminated = False
        
        if not self.caught_target:
            reward -= 0.01 * (ai_torque ** 2) 
        self.last_action = action.copy()
        
        if self.is_attached:
            if self.caught_target:
                current_rigid_state = np.array([self.state[1], self.state[2], self.state[4], self.state[5]])
                next_rigid = self.rigid_dynamics(current_rigid_state, np.array([0.0, 1.0]))
                self.state = np.array([0.0, next_rigid[0], next_rigid[1], 0.0, next_rigid[2], next_rigid[3]])
                if (next_rigid[2]**2 + next_rigid[3]**2) < 0.01:
                    terminated = True
                    reward += 10.0 
            else:
                self.steps_on_branch += 1
                reward -= 0.01
                if self.steps_on_branch > 500:
                    terminated = True
                    reward -= 1.0 

                if grip_cmd <= 0: 
                    self.is_attached = False
                    flight_init = np.concatenate([self.state, [0.0, self.state[0], 0.0, self.state[3]]])
                    self.state = self.flight_dynamics(flight_init, action)
                    reward += 1.0 
                else:
                    self.state = self.elastic_dynamics(self.state, action)
        else:
            self.state = self.flight_dynamics(self.state, action)
            th1, th2 = self.state[1], self.state[2]
            base_pos = self.state[6:8]
            current_tip_pos = self.rigid_dynamics.get_tip_position([th1, th2], base_pos)
            dist_to_target = np.linalg.norm(current_tip_pos - self.target_pos)
            
            reward += (1.0 - dist_to_target) * 0.1
            
            if dist_to_target < 0.15 and grip_cmd > 0:
                self.is_attached = True
                self.caught_target = True
                reward += 10.0 
                self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            if not self.is_attached and self.state[7] < -3.0: 
                terminated = True
                reward -= 1.0
        
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human": return
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.clear()
        self.ax.set_xlim(-2.0, 3.5)
        self.ax.set_ylim(-3.0, 2.0)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.fig.canvas.draw()
        plt.pause(0.001)
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None