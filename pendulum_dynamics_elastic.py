# pendulum_dynamics_elastic.py

import numpy as np

class PendulumDynamics:
    """
    [公開用モック] 基本ダイナミクス＆剛体ブランチ
    """
    def __init__(self, dt=0.02, m1=0.6, m2=0.6, l1=0.6, l2=0.6, 
                 lc1=0.3, lc2=0.3, I1=0.01, I2=0.01, g=9.8):
        self.dt = dt
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.lc1, self.lc2 = lc1, lc2
        self.I1, self.I2 = I1, I2
        self.g = g
        
        self.M_branch = 1.0
        self.K_branch = 100.0
        self.C_branch = 2.0
        self.y_anchor = 0.0
        
        self.max_vel_1 = 4 * np.pi
        self.max_vel_2 = 9 * np.pi

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_tip_position(self, angles, base_pos=np.array([0,0])):
        s0, s1 = angles[0], angles[1]
        p1 = base_pos + np.array([self.l1 * np.sin(s0), -self.l1 * np.cos(s0)])
        p2 = p1 + np.array([self.l2 * np.sin(s0 + s1), -self.l2 * np.cos(s0 + s1)])
        return p2
    
    def get_com_position(self, angles, base_pos=np.array([0,0])):
        s0, s1 = angles[0], angles[1]
        c1 = base_pos + np.array([self.lc1 * np.sin(s0), -self.lc1 * np.cos(s0)])
        p1 = base_pos + np.array([self.l1 * np.sin(s0), -self.l1 * np.cos(s0)])
        c2 = p1 + np.array([self.lc2 * np.sin(s0 + s1), -self.lc2 * np.cos(s0 + s1)])
        
        com_x = (self.m1 * c1[0] + self.m2 * c2[0]) / (self.m1 + self.m2)
        com_y = (self.m1 * c1[1] + self.m2 * c2[1]) / (self.m1 + self.m2)
        return np.array([com_x, com_y])

    def get_jacobian_tip(self, theta1, theta2):
        j11 = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        j12 = self.l2 * np.cos(theta1 + theta2)
        j21 = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        j22 = self.l2 * np.sin(theta1 + theta2)
        return np.array([[j11, j12], [j21, j22]])

    def __call__(self, state, action):
        theta1, theta2, dth1, dth2 = state
        new_dth1 = dth1 * 0.99
        new_dth2 = dth2 * 0.99
        new_th1 = self._normalize_angle(theta1 + new_dth1 * self.dt)
        new_th2 = self._normalize_angle(theta2 + new_dth2 * self.dt)
        return np.array([new_th1, new_th2, new_dth1, new_dth2])


class ElasticPendulumDynamics(PendulumDynamics):
    """
    [公開用モック] 弾性ブランチ（スタート用）
    """
    def __call__(self, state, action):
        y, th1, th2, dy, dth1, dth2 = state
        
        new_dy = dy * 0.98
        new_dth1 = dth1 * 0.99
        new_dth2 = dth2 * 0.99
        new_y = y + new_dy * self.dt
        new_th1 = self._normalize_angle(th1 + new_dth1 * self.dt)
        new_th2 = self._normalize_angle(th2 + new_dth2 * self.dt)
        
        return np.array([new_y, new_th1, new_th2, new_dy, new_dth1, new_dth2])


class FlightDynamics(PendulumDynamics):
    """
    [公開用モック] 空中ダイナミクス
    """
    def __call__(self, state, action):
        y_b, th1, th2, dy_b, dth1, dth2 = state[:6]
        base_pos = state[6:8] 
        base_vel = state[8:10] 
        
        new_dy_b = dy_b * 0.98
        new_y_b = y_b + new_dy_b * self.dt
        new_dth1 = dth1 * 0.99
        new_dth2 = dth2 * 0.99
        new_th1 = self._normalize_angle(th1 + new_dth1 * self.dt)
        new_th2 = self._normalize_angle(th2 + new_dth2 * self.dt)
        
        new_base_pos = base_pos + base_vel * self.dt
        new_base_vel = base_vel

        return np.array([new_y_b, new_th1, new_th2, new_dy_b, new_dth1, new_dth2, 
                         new_base_pos[0], new_base_pos[1], new_base_vel[0], new_base_vel[1]])

    def solve_inelastic_collision(self, state_flight, target_pos):
        return 0.0, 0.0