import numpy as np
import matplotlib.pyplot as plt 
# import torch
# from scipy.linalg import expm
        

class DisturbanceEstimator():
    def __init__(self, init_state, env, interval=None):
        self.t_batch = 0.0
        self.env = env
        # self.dynamics = dynamics_model
        self.state_hat = init_state
        self.state_tilde = np.zeros(self.state_hat.shape)
        self.sigma_hat = np.zeros(self.state_hat.shape)

        # use dynamics function directly to make DOB component more portable
        self.get_f, self.get_g = env._get_dynamics()

        # DOB parameters
        self.dt_int = self.env.dt/1.0  # dt for dynamics integration
        self.dt = self.dt_int/1.0  # dt for DOB
        self.interval = interval
        if self.interval is not None:
            self.dt = self.dt_int/self.interval
        self.Ae = -1.0
        self.Mat_expm = np.exp(self.Ae*self.dt)
        self.Phi = (self.Mat_expm - 1.0) / self.Ae
        self.adapt_gain_no_Bm = -self.Mat_expm/self.Phi

    def state_predictor(self, state, action):
        self.state_hat = self.dt * (self.get_f(state) + self.sigma_hat + self.Ae*self.state_tilde + self.get_g(state) @ action) + self.state_hat
        return self.state_hat

    def adaptive_law(self, state):
        self.state_tilde = self.state_hat - state
        self.sigma_hat = self.adapt_gain_no_Bm * self.state_tilde
        return self.sigma_hat
    
    def disturbance_estimator(self, state, action):
        sigma_hat = None
        state_hat = None
        if self.interval is None:
            sigma_hat = self.adaptive_law(state)
            state_hat = self.state_predictor(state, action)
        else:
            sigma_hat = self.adaptive_law(state)
            for i in range(self.interval):
                state_hat = self.state_predictor(state, action)
        return sigma_hat



if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())
    from envs.quadrotor_env import QuadrotorEnv
    env = QuadrotorEnv()
    env.reset()
    state = env.state
    DOB = DisturbanceEstimator(state, env)
    for i in range(1000):
        action = np.array([-0.5, 0.5])
        sigma_est = DOB.disturbance_estimator(state, action)
        sigma_true = env.uncertainty
        xtilde = DOB.state_tilde
        
        state_next, reward, done, info = env._step(action)
        state = state_next

    plt.show()