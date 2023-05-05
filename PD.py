import numpy as np
import matplotlib.pyplot as plt
from envs.quadrotor_env import QuadrotorEnv
import numpy as np
from rcbf_sac.dynamics import DYNAMICS_MODE
from quadprog import solve_qp
from rcbf_sac.cbf_qp import CascadeCBFLayer
import sys
import sys, os
sys.path.append(os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def PD_controller(states, env, step):
    Kp = np.array([10, 10, 25])
    Kd = np.array([10, 10, 25])
    Kp_rot = np.array([1000, 10, 10])
    Kd_rot = np.array([100, .10, .10])

    phi = states[2]
    phi_c = -(env.ddx_ref_traj[step] + Kd[1] * (env.dx_ref_traj[step] - states[3]) + Kp[1] * (env.x_ref_traj[step] - states[0])) / env.g
    phi_c_dot = -(Kd[1] * (env.ddx_ref_traj[step] + env.g * states[2]) + Kp[1] * (env.dx_ref_traj[step] - states[3])) / env.g

    F = env.mass * (Kd[2] * (env.dz_ref_traj[step] - states[4]) + Kp[2] * (env.z_ref_traj[step] - states[1]) + env.g + env.ddz_ref_traj[step])
    # F_x = env.mass * (Kd[2] * (env.dx_ref_traj[step] - states[3]) + Kp[2] * (env.x_ref_traj[step] - states[0]) + env.ddx_ref_traj[step])
    # F = np.sqrt(F_x**2 + F_z**2)

    M = env.Iyy * (Kp_rot[0] * (phi_c - states[2]) + Kd_rot[0] * (phi_c_dot - states[5]))
    # print('z====')
    # print(env.z_ref_traj[step] - states[1])
    # print('x====')
    # print(env.x_ref_traj[step] - states[0])

    T1 = (F * env.d - M) / (2 * env.d)
    T2 = (F * env.d + M) / (2 * env.d)
    u = np.array([T1, T2])
    return u, phi_c, phi_c_dot


if __name__ == '__main__':
    env = QuadrotorEnv()
    state, obs = env.reset()

    state_list = []
    h_list = []
    safe_action_list = []
    sigma = []
    theta_ref = []
    dtheta_ref = []
    T1_list = []
    T2_list = []
    rew = 0.0
    sigma_hat = np.zeros(state.shape[0])
    for i in range(0, env.max_episode_steps):
        sigma.append(sigma_hat)

        action, theta, dtheta = PD_controller(state, env, i)
        theta_ref.append(theta)
        dtheta_ref.append(dtheta)
        T1_list.append(action[0])
        T2_list.append(action[1])
        state, rewards, dones, info = env.step(action)
        rew += rewards
        state_list.append(state)


    # plot trajectory
    import matplotlib.pyplot as plt
    x_ref_traj = np.array(env.x_ref_traj)
    z_ref_traj = np.array(env.z_ref_traj)
    x_ref_v = np.array(env.dx_ref_traj)
    z_ref_v = np.array(env.dz_ref_traj)
    x_temp = np.array(env.temp1)
    z_temp = np.array(env.temp2)
    x_bound = np.array(env.x_bound)
    z_bound = np.array(env.z_bound)
    state_list = np.array(state_list)
    h_list = np.array(h_list)
    theta_ref = np.array(theta_ref)
    dtheta_ref = np.array(dtheta_ref)
    safe_action_list = np.array(safe_action_list)



    disturb = np.array(env.overall_disturbance)
    sigma = np.array(sigma)


    fig1, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(state_list[:,0])
    ax1.plot(state_list[:,1])
    ax1.plot(x_ref_traj)
    ax1.plot(z_ref_traj)
    ax1.legend(["x", "z", "x_ref", "z_ref"])

    ax2.plot(state_list[:,3])
    ax2.plot(state_list[:,4])
    ax2.plot(x_ref_v)
    ax2.plot(z_ref_v)
    ax2.legend(["dx", "dz", "dx_ref", "dz_ref"])

    fig2 = plt.figure()
    plt.plot(state_list[:,0], state_list[:,1], color='goldenrod')
    plt.plot(x_ref_traj, z_ref_traj, color='blue', linestyle='dashed')
    plt.plot(x_bound, z_bound, color='green')
    plt.scatter(state_list[0,0], state_list[0,1], color='red', marker='*')
    plt.scatter(state_list[-1,0], state_list[-1,1], color='red', marker='*')
    plt.plot(np.array([x_bound[0], x_bound[-1]]), np.array([z_bound[0], z_bound[-1]]), color='green')
    plt.legend(["Real Trajectory", "Ref Trajetory", "Safe Bound"])
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('PD Trajectory Tracking without CBF')

    # theta_ref = theta_ref / np.pi * 180
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax1.plot(theta_ref)
    # ax1.plot(state_list[:,2] / np.pi * 180)
    ax2.plot(dtheta_ref)
    # ax2.plot(state_list[:,5] / np.pi * 180)
    ax1.legend(["theta_ref"])
    ax2.legend(["dtheta_ref"])

    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax1.plot(T1_list)
    ax2.plot(T2_list)
    ax1.legend(["T1"])
    ax2.legend(["T2"])


    plt.show()