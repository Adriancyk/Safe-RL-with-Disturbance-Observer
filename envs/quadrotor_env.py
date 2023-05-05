import numpy as np
import gym
from gym import spaces
from scipy.linalg import expm
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class QuadrotorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(QuadrotorEnv, self).__init__()

        self.dynamics_mode = 'Quadrotor'
        self.get_f, self.get_g = self._get_dynamics()
        self.mass = 0.027
        self.Ixx = 1.4e-5
        self.Iyy = 1.4e-5
        self.Izz = 2.17e-5
        self.g = 9.81
        self.L = 0.046
        self.d = self.L/np.sqrt(2)
        self.x_threshold = 2.0
        self.z_threshold = 2.5
        self.z_ground = 0.0
        self.theta_threshold_radians = 85 * np.pi / 180
        self.u_ref = np.ones(2)*self.mass*self.g/2
        self.dt = 0.01
        self.max_episode_steps = 400
        self.state_err_weight = np.diag([16, 16, 0, 7, 7, 0]) # 16 7
        self.act_err_weight = np.diag([0.01, 0.01])
        self.circle_bound_radius = 0.9
        # Define disturbance factors
        self.frac_factor = 1.0 # rotor fraction factor (1.0 for no disturbance)
        self.drag_factor = 0.0 # wind drag factor (0.0 for no disturbance)
        # Disturbance estimator parameters (no need to change)
        self.max_v = 10.0
        self.a = 1.0
        self.lt = 1.0
        self.ld = 2.0 * self.drag_factor * self.max_v
        self.bd = 1.0
        self.L1_theta = self.ld * 5 + self.bd
        self.L1_phi = 1.5 * self.L1_theta
        self.L1_eta = self.lt + self.ld * self.L1_phi
        self.L1_gamma = 2.0 * np.sqrt(6) * self.L1_eta*self.dt + np.sqrt(6) * (1.0 - np.exp(-self.a * self.dt)) * self.L1_theta
        # self.reward_goal = 100
        self.reward_exp = True

        # useful variables for plotting and analysis
        self.overall_disturbance = []
        self.wind_disturbance = []
        self.friction_disturbance = []
        low = np.array([
                -self.x_threshold, 
                self.z_ground, 
                -self.theta_threshold_radians, 
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max
            ])
        high = np.array([
                self.x_threshold, 
                self.z_threshold, 
                self.theta_threshold_radians, 
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max
            ])
        low_ext = np.array([
                -self.x_threshold, 
                self.z_ground, 
                -self.theta_threshold_radians, 
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max
            ])

        high_ext = np.array([
                self.x_threshold, 
                self.z_threshold, 
                self.theta_threshold_radians, 
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max
            ])

        self.bounded_state_space = spaces.Box(low=low, high=high, shape=(6,))
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,)) # rotor can only generate positive force
        self.safe_action_space = spaces.Box(low=0.0, high=2.0, shape=(2,))
        self.observation_space = spaces.Box(low=low_ext, high=high_ext, shape=(12,))

        # Initialize Env
        self.state = np.zeros((6,))
        self.obs = np.zeros((12,))
        self.uncertainty = np.zeros((6,))
        self.episode_step = 0

        self.reset()
        # Get Dynamics
        # self.get_f, self.get_g = self._get_dynamics()
        # Get Reference Trajectory
        self.x_ref_traj, self.z_ref_traj, self.dx_ref_traj, self.dz_ref_traj, self.ddx_ref_traj, self.ddz_ref_traj,  self.x_bound, self.z_bound = self.generate_trajectory_PD()

        self.x_ref_traj, self.z_ref_traj, self.dx_ref_traj, self.dz_ref_traj, self.theta_ref_traj, self.dtheta_ref_traj = self.generate_trajectory()

        
    def generate_trajectory_PD(self, scale_factor=0.5, shape='circle'): 
        rate = 2.0 * np.pi / (self.max_episode_steps*self.dt)
        x_ref_traj = []
        z_ref_traj = []
        dx_ref_traj = []
        dz_ref_traj = []
        ddx_ref_traj = []
        ddz_ref_traj = []
        x_bound = []
        z_bound = []
        height = 0.5
        if shape == 'figure8':
            for t in np.arange(0, self.max_episode_steps*self.dt + self.dt, self.dt):
                x_ref_traj.append(scale_factor * np.sin(rate * t))
                z_ref_traj.append(scale_factor * np.sin(rate * t) * np.cos(rate * t) + height)
                dx_ref_traj.append(scale_factor * rate * np.cos(rate * t))
                dz_ref_traj.append(scale_factor * rate * (np.cos(rate * t)**2 - np.sin(rate * t)**2))
        elif shape == 'circle':
            for t in np.arange(0, self.max_episode_steps * self.dt + self.dt, self.dt):
                x_ref_traj.append(scale_factor * np.sin(rate * t))
                z_ref_traj.append(-scale_factor * np.cos(rate * t) + height)
                dx_ref_traj.append(scale_factor * rate * np.cos(rate * t))
                dz_ref_traj.append(scale_factor * rate * np.sin(rate * t))
                ddx_ref_traj.append(-scale_factor * rate**2 * np.sin(rate * t))
                ddz_ref_traj.append(scale_factor * rate**2 * np.cos(rate * t))
                x_bound.append(scale_factor * self.circle_bound_radius * 2 * np.cos(rate/2 * t))
                z_bound.append(scale_factor * self.circle_bound_radius * 2 * np.sin(rate/2 * t))
        
        return x_ref_traj, z_ref_traj, dx_ref_traj, dz_ref_traj, ddx_ref_traj, ddz_ref_traj, x_bound, z_bound

    def generate_trajectory(self):
        x_ref_traj = []
        z_ref_traj = []
        dx_ref_traj = []
        dz_ref_traj = []
        theta_ref_traj = []
        dtheta_ref_traj = []
        state = self.state
        theta_ref_traj = []
        dtheta_ref_traj = []
        for i in range(0, self.max_episode_steps):
            action, theta, dtheta = self.PD_controller(state, i)
            x_ref_traj.append(state[0])
            z_ref_traj.append(state[1])
            theta_ref_traj.append(state[2])
            dx_ref_traj.append(state[3])
            dz_ref_traj.append(state[4])
            dtheta_ref_traj.append(state[5])
            state, rewards, dones, info = self.step(action, use_reward=False)
        self.overall_disturbance = []
        self.wind_disturbance = []
        self.friction_disturbance = []
        self.reset()
        return x_ref_traj, z_ref_traj, dx_ref_traj, dz_ref_traj, theta_ref_traj, dtheta_ref_traj, 


    def step(self, action, use_reward=True):
        # action = np.clip(action, -1.0, 1.0)
        state, reward, done, info = self._step(action, use_reward)
        return state, reward, done, info

    def _step(self, action, use_reward=True):
        # x z theta dx dz dtheta
        # Start with the prior for continuous time system x' = f(x) + g(x)u
        # Disturbed continuous time system x' = f(x) + g(x)(u(x) + dm(x)) + d(x)

        self.uncertainty_d = np.zeros(self.state.shape)
        self.uncertainty_dm = np.zeros(action.shape)

        self.uncertainty_d[3] = self.drag_factor * self.state[0] * self.state[0] # mimic drag force disturbance in x direction
        self.uncertainty_d[4] = self.drag_factor * self.state[1] * self.state[1] # mimic drag force disturbance in z direction

        self.uncertainty_dm[0] = self.frac_factor * action[0] # mimic rotor fraction force disturbance in T1 direction
        self.uncertainty_dm[1] = self.frac_factor * action[1] # mimic rotor fraction force disturbance in T2 direction
        self.wind_disturbance.append(self.uncertainty_d)
        self.friction_disturbance.append(self.uncertainty_dm)
        
        self.uncertainty = self.get_g(self.state) @ (self.uncertainty_dm) + self.uncertainty_d
        self.overall_disturbance.append(self.uncertainty)

        self.state = self.dt * (self.get_f(self.state) + self.get_g(self.state) @ (action + self.uncertainty_dm)) + self.state
        self.state = self.dt * self.uncertainty_d + self.state
        self.episode_step += 1
        reward = 0.0
        

        info = dict()
        if use_reward:
            reward = self.get_reward(self.state, action)
        if self.get_done():
            info['out_of_bound'] = True
            reward += -100
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps
            info['reach_max_steps'] = True

        return self.state, reward, done, info

    def get_reward(self, state, action):

        idx = min(self.episode_step, self.max_episode_steps-1)
        act_err = action - self.u_ref
        ref_traj = np.array([self.x_ref_traj[idx], self.z_ref_traj[idx], self.theta_ref_traj[idx], self.dx_ref_traj[idx], self.dz_ref_traj[idx], self.dtheta_ref_traj[idx]])
        state_err = state - ref_traj
        dist = np.sum(state_err@self.state_err_weight@state_err)
        dist += np.sum(act_err@self.act_err_weight@act_err)
        reward = -dist
        if self.reward_exp:
            reward = np.exp(reward)
        return reward

    def get_done(self):

        mask = np.array([1, 1, 1, 0, 0, 0])
        out_of_bound = np.logical_or(self.state < self.bounded_state_space.low, self.state > self.bounded_state_space.high)
        out_of_bound = np.any(out_of_bound * mask)
        if out_of_bound:
            return True
        return False

    def reset(self):

        self.episode_step = 0
        self.state = np.zeros((6, ))
        self.state_nom = np.zeros((6, ))
        self.uncertainty = np.zeros((6,))
        self.obs = np.zeros((12,))

        return self.state, self.obs
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

    def _get_dynamics(self):
        """Get affine CBFs for a given environment.

        Parameters
        ----------

        Returns
        -------
        get_f : callable
                Drift dynamics of the continuous system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the continuous system x' = f(x) + g(x)u
        """

        def get_f(state):
            f_x = np.zeros(state.shape)
            f_x[0] = state[3]
            f_x[1] = state[4]
            f_x[2] = state[5]
            f_x[4] = -self.g
            return f_x

        def get_g(state):
            g_x = np.zeros((state.shape[0], 2))
            theta = state[2]
            g_x = np.array([[0, 0], [0, 0], [0, 0],
                        [-np.sin(theta)/self.mass, -np.sin(theta)/self.mass],
                        [np.cos(theta)/self.mass, np.cos(theta)/self.mass],
                        [       -self.d/self.Iyy,         self.d/self.Iyy]])
            return g_x

        return get_f, get_g

    def update_param(self):
        '''update the gamma and theta of L1 parameters
        '''

        self.L1_theta = self.ld * 1 + self.bd
        self.L1_phi = 1.0 + self.theta
        self.L1_eta = self.lt + self.ld * self.L1_phi
        self.L1_gamma = 2.0 * np.sqrt(6) * self.L1_eta*self.dt + np.sqrt(6) * (1.0 - np.exp(-self.a * self.dt)) * self.L1_theta

        return self.L1_gamma, self.L1_theta
    
    def get_obs(self, states, step):
        self.ref = np.array([self.x_ref_traj[step], self.z_ref_traj[step], self.theta_ref_traj[step], self.dx_ref_traj[step], self.dz_ref_traj[step], self.dtheta_ref_traj[step]])
        self.obs = np.zeros((12,))
        self.obs[0:6] = states
        self.obs[6:12] = self.ref
        return self.obs

    def PD_controller(self, states, step):
        Kp = np.array([10, 10, 25])
        Kd = np.array([10, 10, 25])
        Kp_rot = np.array([1000, 10, 10])
        Kd_rot = np.array([100, .10, .10])

        phi_c = -(self.ddx_ref_traj[step] + Kd[1] * (self.dx_ref_traj[step] - states[3]) + Kp[1] * (self.x_ref_traj[step] - states[0])) / self.g
        phi_c_dot = -(Kd[1] * (self.ddx_ref_traj[step] + self.g * states[2]) + Kp[1] * (self.dx_ref_traj[step] - states[3])) / self.g

        F = self.mass * (Kd[2] * (self.dz_ref_traj[step] - states[4]) + Kp[2] * (self.z_ref_traj[step] - states[1]) + self.g + self.ddz_ref_traj[step])

        M = self.Iyy * (Kp_rot[0] * (phi_c - states[2]) + Kd_rot[0] * (phi_c_dot - states[5]))

        T1 = (F * self.d - M) / (2 * self.d)
        T2 = (F * self.d + M) / (2 * self.d)
        u = np.array([T1, T2])
        return u, phi_c, phi_c_dot

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    # import gym
    # import sys, os
    # sys.path.append(os.getcwd())
    # from rcbf_sac.L1_estimator import L1E
    # import torch
    # from torch.optim import Adam
    # from rcbf_sac.utils import to_tensor
    # from rcbf_sac.model import GaussianPolicy

    # device = torch.device("cuda")
    # action_space = spaces.Box(low=0.0, high=0.9, shape=(2,))
    # target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
    # log_alpha = torch.zeros(1, requires_grad=True, device=device)
    # alpha_optim = Adam([log_alpha], lr=0.0003)

    # policy = GaussianPolicy(6, action_space.shape[0], 256, action_space).to(device)
    # policy_optim = Adam(policy.parameters(), lr=0.0003)

    
    # policy.load_state_dict(
    #         torch.load('test/actor_done1.pkl', map_location=torch.device(device))
    #     )

    env = QuadrotorEnv()

    fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(10, 10))
    ax1.plot(env.theta_ref_traj)
    ax1.set_title('theta')
    ax2.plot(env.dtheta_ref_traj)
    ax2.set_title('dtheta')
    ax3.plot(env.x_ref_traj, env.z_ref_traj)
    ax3.set_title('X and Z')
    ax4.plot(env.dx_ref_traj)
    ax4.set_title('dx')
    ax5.plot(env.dz_ref_traj)
    ax5.set_title('dz')


    # L1 = L1E(np.zeros((6,)), env)
    # sigma = []
    # real_state = []
    # # plt.figure()
    # # plt.plot(env.x_ref_traj, env.z_ref_traj)
    # # plt.plot(env.x_bound, env.z_bound)
    # # # plt.plot(env.dx_ref_traj)
    # # # plt.plot(env.dz_ref_traj)
    # # plt.show()

    # for i in range(0, env.max_episode_steps):
    #     real_state.append(env.state)
    #     state = to_tensor(env.state, torch.FloatTensor, device)
    #     action, _,action_mean = policy.sample(state)
    #     action = action.detach().cpu().numpy()
    #     action_mean = action_mean.detach().cpu().numpy()

    #     sigma_hat = L1.disturbance_estimator(env.state, action_mean)
    #     env.step(action_mean)
    #     sigma.append(sigma_hat)


    # wind = np.array(env.wind_disturbance)
    # fric = np.array(env.friction_disturbance)
    # disturb = np.array(env.overall_disturbance)
    # sigma = np.array(sigma)

    # # plot disturbance and sigma along ddx ddz ddtheta
    # fig1, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    # fig1.suptitle('Disturbance and sigma')
    # ax1.plot(disturb[:, 0], '#13EAC9', label='disturbance')
    # ax1.plot(sigma[:, 0], '--', label='sigma')
    # ax1.legend(['disturbance', 'sigma'])
    # ax2.plot(disturb[:, 1], '#13EAC9', label='disturbance')
    # ax2.plot(sigma[:, 1], '--', label='sigma')
    # ax2.legend(['disturbance', 'sigma'])
    # ax3.plot(disturb[:, 2], '#13EAC9', label='disturbance')
    # ax3.plot(sigma[:, 2], '--', label='sigma')
    # ax3.legend(['disturbance', 'sigma'])
    # ax4.plot(disturb[:, 3], '#13EAC9', label='disturbance')
    # ax4.plot(sigma[:, 3], '--', label='sigma')
    # ax4.legend(['disturbance', 'sigma'])
    # ax5.plot(disturb[:, 4], '#13EAC9', label='disturbance')
    # ax5.plot(sigma[:, 4], '--', label='sigma')
    # ax5.legend(['disturbance', 'sigma'])
    # ax6.plot(disturb[:, 5], '#13EAC9', label='disturbance')
    # ax6.plot(sigma[:, 5], '--', label='sigma')
    # ax6.legend(['disturbance', 'sigma'])


    # fig2, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
    # fig2.suptitle('Disturbance Estimation Error')
    # ax1.plot(disturb[0:-1,3] - sigma[1:,3])  # type: ignore
    # ax1.legend('ddx')
    # ax2.plot(disturb[0:-1,4] - sigma[1:,4])  # type: ignore
    # ax2.legend('ddz')
    # ax3.plot(disturb[0:-1,5] - sigma[1:,5])  # type: ignore
    # ax3.legend('ddtheta')


    # # fig3, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    # # ax1.plot(real[:, 0])
    # # ax2.plot(real[:, 1])
    # # ax3.plot(real[:, 2])
    # # ax4.plot(real[:, 3])
    # # ax5.plot(real[:, 4])
    # # ax6.plot(real[:, 5])

    # fig4, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    # fig4.suptitle('Wind Disturbance')
    # ax1.plot(wind[:, 0])
    # ax2.plot(wind[:, 1])
    # ax3.plot(wind[:, 2])
    # ax4.plot(wind[:, 3])
    # ax5.plot(wind[:, 4])
    # ax6.plot(wind[:, 5])

    # fig5, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    # fig4.suptitle('Rotor Friction Disturbance')
    # ax1.plot(fric[:, 0])
    # ax2.plot(fric[:, 1])


    plt.show()